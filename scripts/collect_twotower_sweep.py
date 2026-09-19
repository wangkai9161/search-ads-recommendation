"""Collect controlled Two-Tower negative-sampling runs into publishable evidence."""

import argparse
import json
from pathlib import Path

import pandas as pd


CONFIGS = (
    ("neg1-lr1e3", 1, 1e-3),
    ("neg4-lr1e3", 4, 1e-3),
    ("neg10-lr1e3", 10, 1e-3),
    ("neg10-lr5e4", 10, 5e-4),
    ("neg20-lr1e3", 20, 1e-3),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for name, negatives, learning_rate in CONFIGS:
        result_file = args.input_root / name / "result.csv"
        if not result_file.is_file():
            continue
        result = pd.read_csv(result_file).iloc[0]
        rows.append(
            {
                "negatives": negatives,
                "learning_rate": learning_rate,
                "epochs": 10,
                "best_epoch": int(result["epoch"]),
                "recall@20": float(result["recall@20"]),
                "ndcg@20": float(result["ndcg@20"]),
            }
        )
    if not rows:
        raise FileNotFoundError("no Two-Tower result.csv files found")
    rows.sort(key=lambda row: (row["recall@20"], row["ndcg@20"]), reverse=True)
    payload = {
        "dataset": "MovieLens-1M ratings >= 3",
        "protocol": "10 epochs, per-user final holdout, full-catalog evaluation with history filtering",
        "results": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.joinpath("twotower-sweep-summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# MovieLens Two-Tower Sampling Sweep",
        "",
        "| Negatives | Learning rate | Best epoch | Recall@20 | NDCG@20 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['negatives']} | {row['learning_rate']:.4g} | {row['best_epoch']} | "
            f"{row['recall@20']:.6f} | {row['ndcg@20']:.6f} |"
        )
    args.output_dir.joinpath("twotower-sweep-report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
