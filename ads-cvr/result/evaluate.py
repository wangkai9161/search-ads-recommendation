"""Evaluation routines for the click-level Criteo CVR task."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .binary import (
    binary_auc,
    binary_average_precision,
    binary_logloss,
    brier_score,
    expected_calibration_error,
)


def evaluate_cvr(
    model,
    reader,
    stats,
    device: str | torch.device,
    val_fraction: float = 0.2,
    split: str = "valid",
) -> dict[str, float]:
    """Evaluate ``P(Sale=1 | click context)`` on a named data split."""
    model.eval()
    losses: list[float] = []
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    offset = 0
    with torch.no_grad():
        use_time_split = reader.time_split is not None
        for batch in reader.iter_batches(stats, split=split if use_time_split else None):
            if use_time_split:
                selected = np.ones(len(batch.labels), dtype=bool)
            elif split in {"train", "valid", "test"}:
                selected = row_split_mask(offset, len(batch.labels), 1.0 - 2.0 * val_fraction, val_fraction, split)
            else:
                raise ValueError("row split mode supports only train and valid evaluation")
            offset += len(batch.labels)
            if not selected.any():
                continue
            index = torch.from_numpy(np.flatnonzero(selected))
            sparse = torch.from_numpy(batch.sparse).index_select(0, index).to(device)
            dense = torch.from_numpy(batch.dense).index_select(0, index).to(device)
            target = torch.from_numpy(batch.labels).index_select(0, index).to(device)
            logits = model(sparse, dense)
            losses.append(
                float(torch.nn.functional.binary_cross_entropy_with_logits(logits, target).item())
                * len(target)
            )
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(target.cpu().numpy())
    if not labels:
        prefix = "val" if split == "valid" else split
        return {f"{prefix}_{name}": float("nan") for name in (
            "loss", "logloss", "auc", "pr_auc", "brier", "ece", "positive_rate"
        )}
    y_prob = np.concatenate(probabilities)
    y_true = np.concatenate(labels)
    prefix = "val" if split == "valid" else split
    return {
        f"{prefix}_loss": float(sum(losses) / len(y_true)),
        f"{prefix}_logloss": binary_logloss(y_true, y_prob),
        f"{prefix}_auc": binary_auc(y_true, y_prob),
        f"{prefix}_pr_auc": binary_average_precision(y_true, y_prob),
        f"{prefix}_brier": brier_score(y_true, y_prob),
        f"{prefix}_ece": expected_calibration_error(y_true, y_prob),
        f"{prefix}_positive_rate": float(y_true.mean()),
    }


def final_metrics(history: list[dict[str, Any]]) -> dict[str, float]:
    """Return the numeric metrics from the last training epoch."""
    if not history:
        return {}
    metrics: dict[str, float] = {}
    for key, value in history[-1].items():
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
            metrics[key] = float(value)
    return metrics


def split_mask(start: int, size: int, val_fraction: float) -> np.ndarray:
    period = 100
    val_width = max(1, min(period - 1, round(period * val_fraction)))
    return (np.arange(start, start + size) % period) >= val_width


def row_split_mask(start: int, size: int, train_fraction: float, val_fraction: float, split: str) -> np.ndarray:
    """Legacy deterministic row split with explicit train/valid/test masks."""
    if not 0.0 < train_fraction < 1.0 or not 0.0 < val_fraction < 1.0:
        raise ValueError("row split fractions must be between 0 and 1")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("row split fractions must sum to less than 1")
    bucket = np.arange(start, start + size) % 100 / 100.0
    if split == "train":
        return bucket < train_fraction
    if split == "valid":
        return (bucket >= train_fraction) & (bucket < train_fraction + val_fraction)
    if split == "test":
        return bucket >= train_fraction + val_fraction
    raise ValueError("split must be train, valid, or test")


__all__ = ["evaluate_cvr", "final_metrics", "split_mask", "row_split_mask"]


@torch.no_grad()
def predict_cvr(model, reader, stats, device: str | torch.device, split: str = "valid"):
    """Return labels, logits and probabilities for a chronological split."""
    model.eval()
    logits = []
    labels = []
    for batch in reader.iter_batches(stats, split=split):
        sparse = torch.from_numpy(batch.sparse).to(device)
        dense = torch.from_numpy(batch.dense).to(device)
        logits.append(model(sparse, dense).detach().cpu().numpy())
        labels.append(batch.labels.astype(np.float32, copy=False))
    if not labels:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    y_true = np.concatenate(labels)
    y_logit = np.concatenate(logits)
    return y_true, y_logit, 1.0 / (1.0 + np.exp(-np.clip(y_logit, -40.0, 40.0)))


__all__.append("predict_cvr")
