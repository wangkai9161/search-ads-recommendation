"""CVR training and metric visualizations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_cvr_visualizations(
    root: str | Path,
    run_name: str,
    history: list[dict[str, Any]],
    metrics: dict[str, Any],
) -> Path:
    """Render loss curves and final CVR metrics under ``output/<run_name>``."""
    output_dir = Path(root) / "output" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_history(output_dir / "training_curves.png", history)
    _save_metrics(output_dir / "cvr_metrics.png", metrics)
    return output_dir


def save_tuning_comparison(
    root: str | Path,
    rows: list[dict[str, Any]],
    output_dir: str | Path | None = None,
) -> Path | None:
    """Render comparable CVR metrics for all successful tuning trials."""
    valid_rows = [row for row in rows if row.get("status") == "ok"]
    if not valid_rows:
        return None
    path = Path(output_dir) / "cvr_tuning_comparison.png" if output_dir else Path(root) / "output" / "cvr_tuning_comparison.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [str(row.get("run_name", f"trial-{index + 1}")) for index, row in enumerate(valid_rows)]
    panels = [
        ("val_logloss", "CVR LogLoss (lower is better)"),
        ("val_pr_auc", "CVR PR-AUC (higher is better)"),
        ("val_auc", "CVR ROC-AUC (higher is better)"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 5), squeeze=False)
    x = np.arange(len(names))
    for axis, (key, title) in zip(axes[0], panels):
        values = [float("nan") if _number(row.get(key)) is None else float(row[key]) for row in valid_rows]
        axis.bar(x, values)
        axis.set_xticks(x, names, rotation=45, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _save_history(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    epochs = [row.get("epoch", index + 1) for index, row in enumerate(history)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), squeeze=False)
    train_loss = [_number(row.get("train_loss")) for row in history]
    val_loss = [_number(row.get("val_loss")) for row in history]
    val_auc = [_number(row.get("val_auc")) for row in history]
    val_pr_auc = [_number(row.get("val_pr_auc")) for row in history]
    if any(value is not None for value in train_loss):
        axes[0, 0].plot(epochs, train_loss, marker="o", label="train_loss")
    if any(value is not None for value in val_loss):
        axes[0, 0].plot(epochs, val_loss, marker="o", label="val_loss")
    if any(value is not None for value in val_auc):
        axes[0, 1].plot(epochs, val_auc, marker="o", label="val_auc")
    if any(value is not None for value in val_pr_auc):
        axes[0, 1].plot(epochs, val_pr_auc, marker="o", label="val_pr_auc")
    for axis, title in zip(axes[0], ("CVR loss", "CVR ranking metrics")):
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(True, alpha=0.25)
        if axis.lines:
            axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_metrics(path: Path, metrics: dict[str, Any]) -> None:
    keys = [
        ("val_logloss", "LogLoss (lower)"),
        ("val_pr_auc", "PR-AUC (higher)"),
        ("val_auc", "ROC-AUC (higher)"),
        ("val_brier", "Brier (lower)"),
        ("val_ece", "ECE (lower)"),
    ]
    available = [(key, label, _number(metrics.get(key))) for key, label in keys]
    available = [(key, label, value) for key, label, value in available if value is not None]
    if not available:
        return
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar([label for _, label, _ in available], [value for _, _, value in available])
    axis.set_title("Final CVR validation metrics")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


__all__ = ["save_cvr_visualizations", "save_tuning_comparison"]
