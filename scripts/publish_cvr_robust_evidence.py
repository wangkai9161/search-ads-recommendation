"""Combine CVR baseline and feature-ablation runs into publishable evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("test_logloss", "test_pr_auc", "test_auc", "test_ece")


def read_successful(path: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in payload["runs"] if row.get("status") == "ok"]
    return payload, rows


def aggregate(rows: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    output = []
    for name, group in groups.items():
        summary: dict[str, object] = {key: name, "runs": len(group)}
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=0))
        output.append(summary)
    return output


def compact(row: dict[str, object]) -> dict[str, object]:
    keys = (
        "model", "feature_set", "seed", "best_epoch", "last_epoch",
        "val_logloss", "val_pr_auc", "val_auc", *METRICS,
    )
    return {key: row.get(key) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--ablation-summary", type=Path, nargs="+", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline_payload, baseline_rows = read_successful(args.baseline_summary)
    ablation_payloads = []
    ablation_rows = []
    for path in args.ablation_summary:
        current_payload, current_rows = read_successful(path)
        ablation_payloads.append(current_payload)
        ablation_rows.extend(current_rows)
    ablation_payload = ablation_payloads[0]
    labels = np.load(args.cache_dir / "labels.float32.npy", mmap_mode="r")
    split = np.load(args.cache_dir / "split.uint8.npy", mmap_mode="r")
    split_stats = {}
    for name, split_id in (("train", 0), ("validation", 1), ("test", 2)):
        selected = split == split_id
        split_stats[name] = {
            "rows": int(selected.sum()),
            "positive_rate": float(labels[selected].mean()),
        }

    payload = {
        "dataset": {
            "name": "Criteo Sponsored Search Conversion Log",
            "target": "Sale after click",
            "rows": int(len(labels)),
            "splits": split_stats,
        },
        "protocol": {
            "split": "chronological 70/15/15",
            "selection": baseline_payload["objective"],
            "epochs_max": baseline_payload["parameters"]["epochs"],
            "categorical_hashing": "field-namespaced hashing; bucket 0 reserved for missing values",
            "post_conversion_fields_excluded": ["SalesAmountInEuro", "time_delay_for_conversion"],
            "baseline_seeds": baseline_payload["parameters"]["seeds"],
            "ablation_seeds": ablation_payload["parameters"]["seeds"],
        },
        "four_model_baseline": [compact(row) for row in baseline_rows],
        "deepfm_feature_ablation_runs": [compact(row) for row in ablation_rows],
        "deepfm_feature_ablation_aggregate": aggregate(ablation_rows, "feature_set"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.joinpath("cvr-robust-summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Criteo CVR Robustness Report",
        "",
        f"Rows: {len(labels):,}. Target: post-click `Sale`. Split: chronological 70/15/15. "
        "Post-conversion revenue and delay fields are excluded.",
        "",
        "## Four-model baseline (seed 42)",
        "",
        "| Model | Test LogLoss | Test PR-AUC | Test AUC | Test ECE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in baseline_rows:
        lines.append(
            f"| {row['model']} | {float(row['test_logloss']):.6f} | "
            f"{float(row['test_pr_auc']):.6f} | {float(row['test_auc']):.6f} | "
            f"{float(row['test_ece']):.6f} |"
        )
    lines.extend([
        "",
        "## DeepFM feature robustness (three seeds)",
        "",
        "| Features | Runs | Test LogLoss | Test PR-AUC | Test AUC |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for row in payload["deepfm_feature_ablation_aggregate"]:
        lines.append(
            f"| {row['feature_set']} | {row['runs']} | "
            f"{row['test_logloss_mean']:.6f} +/- {row['test_logloss_std']:.6f} | "
            f"{row['test_pr_auc_mean']:.6f} +/- {row['test_pr_auc_std']:.6f} | "
            f"{row['test_auc_mean']:.6f} +/- {row['test_auc_std']:.6f} |"
        )
    args.output_dir.joinpath("cvr-robust-report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
