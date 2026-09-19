"""训练 MovieLens 1M 上的第一个 DSSM 基线。

Example:
  python scripts/train_dssm.py --epochs 2 --max-users 1000
"""

import argparse
import random
import sys
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.movielens import load_sequences
from src.evaluation.retrieval import evaluate_retrieval
from src.models.dssm import DSSM


class RetrievalDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        history, target = self.examples[index]
        return torch.tensor(history, dtype=torch.long), target


def collate(batch):
    histories, targets = zip(*batch)
    lengths = torch.tensor([len(history) for history in histories], dtype=torch.long)
    padded = pad_sequence(histories, batch_first=True, padding_value=0)
    # mask=True 表示真实行为，False 表示补齐位置。
    mask = torch.arange(padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    return padded, mask, torch.tensor(targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw/ml-1m")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--negative-mode", choices=("in_batch", "random"), default="in_batch")
    parser.add_argument("--num-negatives", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_sequences(args.data_dir, max_users=args.max_users, max_train_samples=args.max_train_samples)
    model = DSSM(data.num_items, args.embedding_dim).to(device)
    loader = DataLoader(RetrievalDataset(data.train), batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print(f"device={device} users={data.num_users} items={data.num_items} train={len(data.train)} test={len(data.test)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for histories, mask, targets in loader:
            histories, mask, targets = histories.to(device), mask.to(device), targets.to(device)
            optimizer.zero_grad()
            # 两种负采样策略共用同一个训练入口，便于公平对照。
            if args.negative_mode == "in_batch":
                loss = model.in_batch_loss(histories, targets, mask, args.temperature)
            else:
                loss = model.random_negative_loss(
                    histories, targets, data.num_items, mask, args.num_negatives
                )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
        model.eval()
        metrics = evaluate_retrieval(model, data.test, data.num_items, device=device)
        metric_text = " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
        print(f"epoch={epoch} loss={total_loss / max(len(data.train), 1):.4f} {metric_text}")


if __name__ == "__main__":
    main()
