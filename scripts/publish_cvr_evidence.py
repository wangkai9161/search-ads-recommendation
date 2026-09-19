"""Build a path-free, lightweight CVR evidence snapshot from raw run output."""

import argparse
import json
from pathlib import Path


METRICS = (
    "val_logloss",
    "val_auc",
    "val_pr_auc",
    "val_brier",
    "val_ece",
    "test_logloss",
    "test_auc",
    "test_pr_auc",
    "test_brier",
    "test_ece",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    args = parser.parse_args()

    raw = json.loads(args.summary.read_text(encoding="utf-8"))
    parameters = raw["parameters"]
    results = []
    for row in raw["runs"]:
        result = {
            "model": row["model"],
            "feature_set": row["feature_set"],
            "embedding_dim": row["embedding_dim"],
            "learning_rate": row["learning_rate"],
            "seed": row["seed"],
            "status": row["status"],
            "best_epoch": row.get("best_epoch"),
            "last_epoch": row.get("last_epoch"),
        }
        result.update({metric: row.get(metric) for metric in METRICS})
        results.append(result)

    calibration_file = args.summary.parent / "calibration/calibration_metrics.json"
    calibration = json.loads(calibration_file.read_text(encoding="utf-8")) if calibration_file.is_file() else None
    payload = {
        "dataset": {
            "name": "Criteo Sponsored Search Conversion Log",
            "rows": args.rows,
            "target": "Sale after click",
        },
        "protocol": {
            "split_mode": parameters["split_mode"],
            "train_fraction": parameters["train_fraction"],
            "validation_fraction": parameters["val_fraction"],
            "models": parameters["models"],
            "feature_sets": parameters["feature_sets"],
            "epochs_max": parameters["epochs"],
            "early_stopping_patience": parameters["early_stopping_patience"],
            "embedding_dims": parameters["embedding_dims"],
            "learning_rates": parameters["learning_rates"],
            "seeds": parameters["seeds"],
        },
        "selection": raw["objective"],
        "best_model": raw["best_run"]["model"] if raw.get("best_run") else None,
        "results": results,
        "calibration": calibration,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.joinpath("cvr-full-four-model-summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Criteo Full-Data Four-Model Comparison",
        "",
        f"Rows: {args.rows:,}. Target: post-click `Sale`. Split: chronological 70/15/15. "
        f"Training budget: at most {parameters['epochs']} epochs with validation LogLoss early stopping.",
        "",
        "| Model | Best/last epoch | Test LogLoss | Test PR-AUC | Test AUC | Test ECE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['model']} | {row['best_epoch']}/{row['last_epoch']} | "
            f"{row['test_logloss']:.6f} | {row['test_pr_auc']:.6f} | "
            f"{row['test_auc']:.6f} | {row['test_ece']:.6f} |"
        )
    lines.extend(["", f"Selected by validation objective: **{payload['best_model']}**."])
    args.output_dir.joinpath("cvr-full-four-model-report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
