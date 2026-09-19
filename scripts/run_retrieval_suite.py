"""Run the frozen-resume LastFM and MovieLens retrieval evidence suite."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"[run] cwd={cwd} command={' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def result_row(path: Path, epochs: int) -> dict[str, object]:
    frame = pd.read_csv(path)
    row = {
        key: value.item() if hasattr(value, "item") else value
        for key, value in frame.iloc[0].to_dict().items()
    }
    row["training_epochs"] = epochs
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--movielens-dir", type=Path, required=True)
    parser.add_argument("--lastfm-file", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    movie_csv = args.runtime_dir / "movielens-1m.csv"
    run(
        [args.python, str(ROOT / "sequential-ranking/prepare_movielens.py"), "--ratings", str(args.movielens_dir / "ratings.dat"), "--output", str(movie_csv)],
        ROOT,
        env,
    )

    lastfm_output = args.evidence_dir / "lastfm"
    run(
        [
            args.python,
            str(ROOT / "retrieval-generative/scripts/train_lastfm_ablation.py"),
            "--data-file",
            str(args.lastfm_file),
            "--output-dir",
            str(lastfm_output),
            "--epochs",
            str(args.epochs),
            "--device",
            "cuda",
        ],
        ROOT / "retrieval-generative",
        env,
    )

    movie_output = args.runtime_dir / "movielens-output"
    twotower_output = movie_output / "twotower"
    sequence_output = movie_output / "sequence"
    run(
        [
            args.python,
            "train_twotower.py",
            "--data_path",
            str(movie_csv),
            "--save_dir",
            str(twotower_output),
            "--epochs",
            str(args.epochs),
            "--topk",
            "20",
        ],
        ROOT / "sequential-ranking/recall-twotower",
        env,
    )
    run(
        [
            args.python,
            "train_sequence_models.py",
            "--data_path",
            str(movie_csv),
            "--save_root",
            str(sequence_output),
            "--model",
            "all",
            "--epochs",
            str(args.epochs),
            "--topk",
            "20",
        ],
        ROOT / "sequential-ranking/sequence-sasrec",
        env,
    )

    rows = [
        {"model": "twotower", **result_row(twotower_output / "result.csv", args.epochs)},
        result_row(sequence_output / "poprec/result.csv", 0),
        result_row(sequence_output / "gru4rec/result.csv", args.epochs),
        result_row(sequence_output / "sasrec/result.csv", args.epochs),
    ]
    payload = {
        "dataset": "MovieLens-1M ratings >= 3",
        "split": "per-user final interaction holdout",
        "evaluation": "full catalog with prior interactions filtered",
        "rows": rows,
    }
    summary_path = args.evidence_dir / "movielens-10epoch-summary.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# MovieLens-1M Unified Retrieval Comparison",
        "",
        "All learned models use 10 training epochs. Evaluation holds out each user's final "
        "positive interaction, ranks the full catalog, and filters prior interactions.",
        "",
        "| Model | Training epochs | Best epoch | Recall@20 | NDCG@20 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['training_epochs']} | {int(row['epoch'])} | "
            f"{row['recall@20']:.6f} | {row['ndcg@20']:.6f} |"
        )
    args.evidence_dir.joinpath("movielens-10epoch-report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
