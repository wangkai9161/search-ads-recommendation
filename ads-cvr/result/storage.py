"""Persistence for model checkpoints and numeric CVR results."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def save_training_result(
    root: str | Path,
    run_name: str,
    model: torch.nn.Module,
    history: list[dict[str, Any]],
    config: dict[str, Any],
    started_at: float,
    metrics: dict[str, Any] | None = None,
):
    """Save one run under ``result/<run_name>`` without rendering plots."""
    root = Path(root)
    result_dir = root / "result" / run_name
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = result_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_path = checkpoint_dir / "model_final.pth"
    torch.save(model.state_dict(), model_path)

    history_path = result_dir / "train_history.csv"
    fields = sorted({key for row in history for key in row}) if history else []
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(history)

    result_metrics = dict(history[-1] if history else {})
    if history and any("val_logloss" in row for row in history):
        best_row = min(
            (row for row in history if isinstance(row.get("val_logloss"), (int, float))),
            key=lambda row: float(row["val_logloss"]),
        )
        result_metrics["last_epoch"] = history[-1].get("epoch")
        result_metrics["best_epoch"] = best_row.get("epoch")
        result_metrics["best_val_logloss"] = best_row.get("val_logloss")
        result_metrics["epoch"] = best_row.get("epoch")
    result_metrics.update(metrics or {})
    result_metrics.update(
        {
            "run_name": run_name,
            "model": config.get("model", ""),
            "task": config.get("task", "cvr"),
            "target": config.get("target", "Sale"),
            "elapsed_seconds": time.perf_counter() - started_at,
            "checkpoint": str(model_path),
        }
    )
    (result_dir / "metrics.json").write_text(
        json.dumps(result_metrics, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    (result_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )
    return result_dir, result_metrics


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


__all__ = ["save_training_result"]
