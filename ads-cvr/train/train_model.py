"""Generic streaming trainer for the current Criteo sponsored-search data."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from model.registry import build_model
from prepare.reader import DenseStats, SponsoredSearchReader, feature_columns
from result.evaluate import evaluate_cvr, row_split_mask
from result.storage import save_training_result


SUPPORTED_MODELS = ("lr", "fm", "wide_deep", "deepfm")


@dataclass
class TrainConfig:
    model: str = "deepfm"
    task: str = "cvr"
    target: str = "Sale"
    run_name: str | None = None
    run_tag: str | None = None
    data_path: str = "data/criteo-sponsored-search/data.tsv"
    cache_dir: str | None = None
    epochs: int = 8
    batch_size: int = 65_536
    max_rows: int | None = 200_000
    stats_rows: int = 1_000_000
    num_sparse_bins: int = 100_000
    embedding_dim: int = 16
    learning_rate: float = 1e-3
    val_fraction: float = 0.2
    train_fraction: float = 0.7
    split_mode: str = "time"
    feature_set: str = "full"
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-5
    seed: int = 42
    device: str = "auto"


def _to_tensors(batch, indices, device):
    index = torch.from_numpy(indices)
    sparse = torch.from_numpy(batch.sparse).index_select(0, index).to(device)
    dense = torch.from_numpy(batch.dense).index_select(0, index).to(device)
    labels = torch.from_numpy(batch.labels).index_select(0, index).to(device)
    return sparse, dense, labels


def train_model(
    config: TrainConfig | None = None,
    *,
    reader: SponsoredSearchReader | None = None,
    stats: DenseStats | None = None,
    save_artifacts: bool = True,
    **overrides,
):
    """Train one registered tabular model and persist its artifacts.

    The function intentionally streams the input file. ``max_rows`` can be
    lowered for a quick smoke run; use ``None`` to process the full dataset.
    The project entry point can pass a prepared ``reader`` and ``stats`` and
    set ``save_artifacts=False`` so result/output persistence stays in the
    orchestration layer.
    """
    config = config or TrainConfig()
    values = asdict(config)
    values.update(overrides)
    config = TrainConfig(**values)
    model_name = config.model.lower().strip()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"supported streaming models: {', '.join(SUPPORTED_MODELS)}")
    if config.task.lower() != "cvr":
        raise ValueError("the current Sponsored Search trainer supports task='cvr' only")
    if config.target != "Sale":
        raise ValueError("the current Sponsored Search trainer uses target='Sale'")
    if not 0.0 < config.val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if config.split_mode not in {"time", "row"}:
        raise ValueError("split_mode must be 'time' or 'row'")
    if config.split_mode == "time" and config.train_fraction + config.val_fraction >= 1.0:
        raise ValueError("train_fraction + val_fraction must be below 1")
    if config.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be at least 1")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(
        "cuda" if config.device == "auto" and torch.cuda.is_available() else
        "cpu" if config.device == "auto" else config.device
    )
    if reader is None:
        sparse_columns, dense_columns = feature_columns(config.feature_set)
        reader = SponsoredSearchReader(
            config.data_path,
            num_sparse_bins=config.num_sparse_bins,
            chunksize=config.batch_size,
            max_rows=config.max_rows,
            sparse_columns=sparse_columns,
            dense_columns=dense_columns,
        )
    if config.split_mode == "time" and reader.time_split is None:
        reader.fit_time_split(config.train_fraction, config.val_fraction)
    started_at = time.perf_counter()
    if stats is None:
        stats = reader.fit_dense_stats(
            config.stats_rows,
            split="train" if config.split_mode == "time" else None,
        )
    model = build_model(
        model_name,
        num_sparse_bins=reader.num_sparse_bins,
        num_sparse_fields=reader.num_sparse_fields,
        num_dense=reader.num_dense,
        embedding_dim=config.embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history = []

    print(
        f"device={device} model={model_name} rows={config.max_rows or 'all'} "
        f"sparse_fields={reader.num_sparse_fields} dense_fields={reader.num_dense}"
    )
    best_logloss = float("inf")
    best_state = None
    stale_epochs = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        offset = 0
        total_loss = 0.0
        seen = 0
        train_split = "train" if config.split_mode == "time" else None
        for batch in reader.iter_batches(stats, split=train_split):
            if not len(batch.labels):
                continue
            if config.split_mode == "row":
                selected = row_split_mask(offset, len(batch.labels), 1.0 - 2.0 * config.val_fraction, config.val_fraction, "train")
                offset += len(batch.labels)
                if not selected.any():
                    continue
                indices = torch.from_numpy(np.flatnonzero(selected))
                sparse = torch.from_numpy(batch.sparse).index_select(0, indices).to(device)
                dense = torch.from_numpy(batch.dense).index_select(0, indices).to(device)
                labels = torch.from_numpy(batch.labels).index_select(0, indices).to(device)
            else:
                sparse = torch.from_numpy(batch.sparse).to(device)
                dense = torch.from_numpy(batch.dense).to(device)
                labels = torch.from_numpy(batch.labels).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(sparse, dense, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            seen += len(labels)
        metrics = evaluate_cvr(model, reader, stats, device, config.val_fraction, split="valid")
        train_logloss = total_loss / max(seen, 1)
        row = {"epoch": epoch, "train_loss": train_logloss, "train_logloss": train_logloss, **metrics}
        history.append(row)
        print(
            f"epoch={epoch} train_loss={row['train_loss']:.5f} "
            f"val_logloss={row['val_logloss']:.5f} val_auc={row['val_auc']:.5f} "
            f"val_pr_auc={row['val_pr_auc']:.5f}"
        )

        current_logloss = float(row["val_logloss"])
        if current_logloss < best_logloss - config.early_stopping_min_delta:
            best_logloss = current_logloss
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.early_stopping_patience:
                print(f"early_stop epoch={epoch} best_val_logloss={best_logloss:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    run_name = config.run_name or f"sponsored-search-{model_name}"
    values["run_name"] = run_name
    values["resolved_device"] = str(device)
    if save_artifacts:
        save_training_result(ROOT, run_name, model, history, values, started_at)
    return model, history
