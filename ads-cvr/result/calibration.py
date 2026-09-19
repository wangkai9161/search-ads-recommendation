"""Probability calibration and lift reports for CVR predictions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .binary import binary_logloss, brier_score, expected_calibration_error


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit a scalar temperature on validation logits using a deterministic grid."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if not len(logits):
        return 1.0
    temperatures = np.exp(np.linspace(np.log(0.25), np.log(4.0), 161))
    losses = []
    for temperature in temperatures:
        probabilities = _sigmoid(logits / temperature)
        losses.append(binary_logloss(labels, probabilities))
    return float(temperatures[int(np.argmin(losses))])


def calibration_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "logloss": binary_logloss(labels, probabilities),
        "brier": brier_score(labels, probabilities),
        "ece": expected_calibration_error(labels, probabilities),
        "positive_rate": float(np.asarray(labels).mean()) if len(labels) else float("nan"),
    }


def calibration_table(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for index in range(bins):
        selected = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1] if index == bins - 1 else probabilities < edges[index + 1]
        )
        rows.append({
            "bin": index,
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": int(selected.sum()),
            "mean_predicted": float(probabilities[selected].mean()) if selected.any() else float("nan"),
            "empirical_rate": float(labels[selected].mean()) if selected.any() else float("nan"),
        })
    return rows


def lift_table(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not len(labels):
        return []
    order = np.argsort(-probabilities, kind="mergesort")
    ordered_labels = labels[order]
    ordered_probabilities = probabilities[order]
    overall = float(labels.mean())
    rows = []
    for index, indices in enumerate(np.array_split(np.arange(len(labels)), bins), start=1):
        if not len(indices):
            continue
        rate = float(ordered_labels[indices].mean())
        rows.append({
            "decile": index,
            "count": int(len(indices)),
            "mean_predicted": float(ordered_probabilities[indices].mean()),
            "conversion_rate": rate,
            "lift": rate / overall if overall > 0 else float("nan"),
            "cumulative_gain": float(ordered_labels[: indices[-1] + 1].sum() / max(ordered_labels.sum(), 1.0)),
        })
    return rows


def save_calibration_report(
    output_dir: str | Path,
    validation_labels: np.ndarray,
    validation_logits: np.ndarray,
    test_labels: np.ndarray,
    test_logits: np.ndarray,
) -> Path:
    """Fit on validation only and save tables, plots and summary JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temperature = fit_temperature(validation_logits, validation_labels)
    report = {"temperature": temperature, "splits": {}}
    for name, labels, logits in (("validation", validation_labels, validation_logits), ("test", test_labels, test_logits)):
        raw = _sigmoid(logits)
        calibrated = _sigmoid(logits / temperature)
        report["splits"][name] = {
            "raw": calibration_metrics(labels, raw),
            "temperature_scaled": calibration_metrics(labels, calibrated),
        }
        _write_csv(output_dir / f"{name}_calibration.csv", calibration_table(labels, calibrated))
        _write_csv(output_dir / f"{name}_lift.csv", lift_table(labels, calibrated))

    _plot_reliability(output_dir / "reliability_diagram.png", validation_labels, validation_logits, test_labels, test_logits, temperature)
    _plot_lift(output_dir / "decile_lift.png", test_labels, test_logits, temperature)
    path = output_dir / "calibration_metrics.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    return path


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = np.clip(np.asarray(logits, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    import csv

    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def _plot_reliability(path, val_labels, val_logits, test_labels, test_logits, temperature):
    fig, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], "k--", label="perfect")
    for name, labels, logits in (("validation", val_labels, val_logits), ("test", test_labels, test_logits)):
        rows = calibration_table(labels, _sigmoid(logits / temperature))
        valid = [row for row in rows if np.isfinite(row["mean_predicted"])]
        axis.plot([row["mean_predicted"] for row in valid], [row["empirical_rate"] for row in valid], marker="o", label=name)
    axis.set_xlabel("mean predicted CVR")
    axis.set_ylabel("empirical conversion rate")
    axis.set_title(f"Reliability diagram (temperature={temperature:.3f})")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_lift(path, labels, logits, temperature):
    rows = lift_table(labels, _sigmoid(logits / temperature))
    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot([row["decile"] for row in rows], [row["lift"] for row in rows], marker="o")
    axis.axhline(1.0, color="k", linestyle="--", linewidth=1)
    axis.set_xlabel("prediction decile (high to low)")
    axis.set_ylabel("lift")
    axis.set_title("Test decile lift")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


__all__ = [
    "fit_temperature",
    "calibration_metrics",
    "calibration_table",
    "lift_table",
    "save_calibration_report",
]
