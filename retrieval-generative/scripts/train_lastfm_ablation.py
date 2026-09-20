"""Run controlled LastFM two-tower negative-sampling and loss ablations."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.lastfm import LastFMSplit, load_lastfm_split
from src.models.user_item_tower import UserItemTwoTower


@dataclass(frozen=True)
class Trial:
    loss: str
    negatives: int
    tail_weighting: bool = False

    @property
    def name(self) -> str:
        suffix = "-tail" if self.tail_weighting else ""
        return f"{self.loss}-neg{self.negatives}{suffix}"


DEFAULT_TRIALS = (
    Trial("bce", 0),
    Trial("bce", 1),
    Trial("bce", 3),
    Trial("bce", 5),
    Trial("bpr", 1),
    Trial("bpr", 3),
    Trial("bpr", 5),
    Trial("bce", 5, True),
    Trial("bpr", 5, True),
)


class InteractionDataset(Dataset):
    def __init__(self, split: LastFMSplit, negatives: int, seed: int, tail_weighting: bool):
        self.split = split
        self.negatives = negatives
        self.rng = np.random.default_rng(seed)
        popularity = np.maximum(split.item_popularity, 1)
        weights = np.sqrt(popularity.max() / popularity)
        weights = np.clip(weights / weights.mean(), 0.5, 5.0)
        self.weights = weights.astype(np.float32) if tail_weighting else np.ones_like(weights, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.split.train_users)

    def __getitem__(self, index: int):
        user = int(self.split.train_users[index])
        positive = int(self.split.train_items[index])
        negatives: list[int] = []
        while len(negatives) < self.negatives:
            candidate = int(self.rng.integers(0, self.split.num_items))
            if candidate not in self.split.user_seen[user] and candidate not in negatives:
                negatives.append(candidate)
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(positive, dtype=torch.long),
            torch.tensor(negatives, dtype=torch.long),
            torch.tensor(self.weights[positive], dtype=torch.float32),
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, device, loss_name: str) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for users, positives, negatives, weights in loader:
        users = users.to(device)
        positives = positives.to(device)
        negatives = negatives.to(device)
        weights = weights.to(device)
        positive_scores = model(users, positives)

        if loss_name == "bce":
            positive_loss = F.binary_cross_entropy_with_logits(
                positive_scores, torch.ones_like(positive_scores), reduction="none"
            ) * weights
            if negatives.shape[1]:
                expanded_users = users[:, None].expand_as(negatives)
                negative_scores = model(expanded_users.reshape(-1), negatives.reshape(-1)).reshape_as(negatives)
                negative_loss = F.binary_cross_entropy_with_logits(
                    negative_scores, torch.zeros_like(negative_scores), reduction="none"
                ).mean(dim=1)
                loss = (positive_loss + negative_loss).mean()
            else:
                loss = positive_loss.mean()
        elif loss_name == "bpr":
            expanded_users = users[:, None].expand_as(negatives)
            negative_scores = model(expanded_users.reshape(-1), negatives.reshape(-1)).reshape_as(negatives)
            loss = (F.softplus(negative_scores - positive_scores[:, None]).mean(dim=1) * weights).mean()
        else:
            raise ValueError(f"unsupported loss: {loss_name}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(users)
        total_examples += len(users)
    return total_loss / total_examples


@torch.no_grad()
def evaluate(
    model,
    split: LastFMSplit,
    targets: np.ndarray,
    device,
    topk: int,
    extra_seen: np.ndarray | None = None,
    batch_size: int = 128,
) -> dict[str, float]:
    model.eval()
    item_ids = torch.arange(split.num_items, device=device)
    item_vectors = model.encode_item(item_ids)
    hits = 0
    ndcg = 0.0
    tail_hits = 0
    tail_count = 0
    warm_hits = 0
    warm_count = 0
    cold_count = 0
    covered: set[int] = set()
    nonzero_popularity = split.item_popularity[split.item_popularity > 0]
    tail_threshold = float(np.median(nonzero_popularity))

    for start in range(0, split.num_users, batch_size):
        stop = min(start + batch_size, split.num_users)
        users = torch.arange(start, stop, device=device)
        scores = model.encode_user(users) @ item_vectors.T
        for row, user in enumerate(range(start, stop)):
            seen = split.user_seen[user]
            if extra_seen is not None:
                seen = seen | {int(extra_seen[user])}
            if seen:
                scores[row, torch.tensor(sorted(seen), device=device)] = -torch.inf
        ranking = scores.topk(min(topk, split.num_items), dim=1).indices.cpu().tolist()
        for offset, recommendations in enumerate(ranking):
            user = start + offset
            target = int(targets[user])
            covered.update(recommendations)
            # A target never observed in training has no learned item signal;
            # report tail recall only for rare but evaluable artists.
            is_tail = 0 < split.item_popularity[target] <= tail_threshold
            is_warm = split.item_popularity[target] > 0
            if is_tail:
                tail_count += 1
            if is_warm:
                warm_count += 1
            else:
                cold_count += 1
            if target in recommendations:
                hits += 1
                rank = recommendations.index(target) + 1
                ndcg += 1.0 / math.log2(rank + 1.0)
                if is_tail:
                    tail_hits += 1
                if is_warm:
                    warm_hits += 1

    return {
        f"recall@{topk}": hits / split.num_users,
        f"ndcg@{topk}": ndcg / split.num_users,
        f"item_coverage@{topk}": len(covered) / split.num_items,
        f"warm_recall@{topk}": warm_hits / max(warm_count, 1),
        f"tail_recall@{topk}": tail_hits / max(tail_count, 1),
        "warm_target_count": warm_count,
        "tail_target_count": tail_count,
        "cold_target_count": cold_count,
        "cold_target_rate": cold_count / split.num_users,
    }


def run_trial(args, split: LastFMSplit, trial: Trial, device) -> dict[str, object]:
    if trial.loss == "bpr" and trial.negatives < 1:
        raise ValueError("BPR requires at least one negative")
    set_seed(args.seed)
    dataset = InteractionDataset(split, trial.negatives, args.seed, trial.tail_weighting)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = UserItemTwoTower(split.num_users, split.num_items, args.embedding_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history: list[dict[str, float]] = []
    best_val: dict[str, float] | None = None
    best_state = None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, loader, optimizer, device, trial.loss)
        metrics = evaluate(model, split, split.val_items, device, args.topk)
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(row)
        if best_val is None or metrics[f"recall@{args.topk}"] > best_val[f"recall@{args.topk}"]:
            best_val = row
            best_state = deepcopy(model.state_dict())
        print(
            f"[{trial.name}] epoch={epoch}/{args.epochs} loss={train_loss:.6f} "
            f"recall@{args.topk}={metrics[f'recall@{args.topk}']:.6f} "
            f"ndcg@{args.topk}={metrics[f'ndcg@{args.topk}']:.6f}",
            flush=True,
        )
    model.load_state_dict(best_state)
    test_metrics = evaluate(
        model,
        split,
        split.test_items,
        device,
        args.topk,
        extra_seen=split.val_items,
    )
    return {
        "trial": asdict(trial),
        "name": trial.name,
        "best_epoch": best_val["epoch"],
        "best_val": best_val,
        "test": test_metrics,
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
    }


def write_report(payload: dict[str, object], output_dir: Path, topk: int) -> None:
    rows = payload["results"]
    lines = [
        "# LastFM Two-Tower Ablation",
        "",
        "HetRec LastFM implicit feedback with a fixed seeded per-user holdout. "
        "Because the dataset has no timestamps, the split is not chronological.",
        "",
        f"Interactions: {payload['dataset']['num_interactions']:,}; users: {payload['dataset']['num_users']:,}; "
        f"artists: {payload['dataset']['num_items']:,}; epochs per trial: {payload['config']['epochs']}.",
        "",
        f"| Trial | Best epoch | Val Recall@{topk} | Test Recall@{topk} | Test NDCG@{topk} | "
        f"Coverage@{topk} | Warm Recall@{topk} | Tail Recall@{topk} |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in rows:
        val = result["best_val"]
        test = result["test"]
        lines.append(
            f"| {result['name']} | {result['best_epoch']} | {val[f'recall@{topk}']:.6f} | "
            f"{test[f'recall@{topk}']:.6f} | {test[f'ndcg@{topk}']:.6f} | "
            f"{test[f'item_coverage@{topk}']:.6f} | {test[f'warm_recall@{topk}']:.6f} | "
            f"{test[f'tail_recall@{topk}']:.6f} |"
        )
    output_dir.joinpath("report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    resolved_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(resolved_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    split = load_lastfm_split(args.data_file, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [run_trial(args, split, trial, device) for trial in DEFAULT_TRIALS]
    payload = {
        "dataset": {
            "name": "HetRec 2011 LastFM 2K",
            "num_interactions": split.num_interactions,
            "num_users": split.num_users,
            "num_items": split.num_items,
        },
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "learning_rate": args.learning_rate,
            "topk": args.topk,
            "seed": args.seed,
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "results": results,
    }
    args.output_dir.joinpath("summary.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    write_report(payload, args.output_dir, args.topk)


if __name__ == "__main__":
    main()
