"""Run leakage-free, multi-seed LastFM and MovieLens experiments."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"[run] cwd={cwd} command={' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def csv_row(path: Path) -> dict[str, object]:
    row = pd.read_csv(path).iloc[0].to_dict()
    return {key: value.item() if hasattr(value, "item") else value for key, value in row.items()}


def aggregate(rows: list[dict[str, object]], group_key: str, metrics: list[str]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row[group_key]), []).append(row)
    output = []
    for name, group in groups.items():
        summary: dict[str, object] = {group_key: name, "runs": len(group)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=0))
        output.append(summary)
    return output


def write_movie_evidence(
    rows: list[dict[str, object]],
    output_dir: Path,
    epochs: int,
    seeds: list[int],
    sequence_train_sample_cap: int,
) -> None:
    metrics = ["val_recall@20", "val_ndcg@20", "test_recall@20", "test_ndcg@20"]
    payload = {
        "dataset": "MovieLens-1M ratings >= 3; users with at least 3 positives",
        "protocol": {
            "split": "per-user chronological train/validation/test; final two positives held out",
            "selection": "best epoch by validation Recall@20; test evaluated once",
            "evaluation": "full catalog with known positives filtered",
            "epochs": epochs,
            "seeds": seeds,
            "sequence_train_sample_cap": sequence_train_sample_cap,
        },
        "runs": rows,
        "aggregate": aggregate(rows, "model", metrics),
    }
    output_dir.joinpath("movielens-10epoch-summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# MovieLens-1M Leakage-Free Multi-Seed Comparison",
        "",
        "The penultimate positive selects the epoch and the final positive is evaluated once. "
        "Reported values are population mean and standard deviation across seeds.",
        "",
        "| Model | Runs | Test Recall@20 | Test NDCG@20 | Val Recall@20 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["aggregate"]:
        lines.append(
            f"| {row['model']} | {row['runs']} | "
            f"{row['test_recall@20_mean']:.6f} +/- {row['test_recall@20_std']:.6f} | "
            f"{row['test_ndcg@20_mean']:.6f} +/- {row['test_ndcg@20_std']:.6f} | "
            f"{row['val_recall@20_mean']:.6f} +/- {row['val_recall@20_std']:.6f} |"
        )
    output_dir.joinpath("movielens-10epoch-report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_lastfm_evidence(rows: list[dict[str, object]], output_dir: Path, epochs: int, seeds: list[int]) -> None:
    metrics = [
        "val_recall@20",
        "test_recall@20",
        "test_ndcg@20",
        "test_item_coverage@20",
        "test_warm_recall@20",
        "test_tail_recall@20",
        "cold_target_rate",
    ]
    payload = {
        "dataset": "HetRec 2011 LastFM 2K implicit feedback",
        "protocol": {
            "split": "seeded per-user train/validation/test; two distinct artists held out",
            "selection": "best epoch by validation Recall@20; test evaluated once",
            "evaluation": "full catalog with known positives filtered",
            "epochs": epochs,
            "seeds": seeds,
        },
        "runs": rows,
        "aggregate": aggregate(rows, "trial", metrics),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# LastFM Leakage-Free Multi-Seed Ablation",
        "",
        "Because LastFM has no timestamps, each seed changes both the random holdout and model initialization. "
        "Cold targets have no training interaction and are separated from warm and tail recall.",
        "",
        "| Trial | Runs | Test Recall@20 | NDCG@20 | Coverage@20 | Warm Recall@20 | Tail Recall@20 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["aggregate"]:
        lines.append(
            f"| {row['trial']} | {row['runs']} | "
            f"{row['test_recall@20_mean']:.6f} +/- {row['test_recall@20_std']:.6f} | "
            f"{row['test_ndcg@20_mean']:.6f} +/- {row['test_ndcg@20_std']:.6f} | "
            f"{row['test_item_coverage@20_mean']:.6f} | {row['test_warm_recall@20_mean']:.6f} | "
            f"{row['test_tail_recall@20_mean']:.6f} |"
        )
    output_dir.joinpath("report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--movielens-dir", type=Path, required=True)
    parser.add_argument("--lastfm-file", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=parse_seeds, default=[41, 42, 43])
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--sequence-train-samples", type=int, default=200_000)
    args = parser.parse_args()
    python_path = Path(args.python).expanduser()
    if not python_path.is_absolute():
        python_path = Path.cwd() / python_path
    args.python = str(python_path.absolute())

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

    movie_rows: list[dict[str, object]] = []
    lastfm_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        seed_dir = args.runtime_dir / f"seed-{seed}"
        lastfm_output = seed_dir / "lastfm"
        run(
            [
                args.python,
                str(ROOT / "retrieval-generative/scripts/train_lastfm_ablation.py"),
                "--data-file", str(args.lastfm_file),
                "--output-dir", str(lastfm_output),
                "--epochs", str(args.epochs),
                "--seed", str(seed),
                "--device", "cuda",
            ],
            ROOT / "retrieval-generative",
            env,
        )
        lastfm_payload = json.loads(lastfm_output.joinpath("summary.json").read_text(encoding="utf-8"))
        for result in lastfm_payload["results"]:
            test = result["test"]
            val = result["best_val"]
            lastfm_rows.append(
                {
                    "trial": result["name"],
                    "seed": seed,
                    "best_epoch": result["best_epoch"],
                    "val_recall@20": val["recall@20"],
                    "test_recall@20": test["recall@20"],
                    "test_ndcg@20": test["ndcg@20"],
                    "test_item_coverage@20": test["item_coverage@20"],
                    "test_warm_recall@20": test["warm_recall@20"],
                    "test_tail_recall@20": test["tail_recall@20"],
                    "warm_target_count": test["warm_target_count"],
                    "tail_target_count": test["tail_target_count"],
                    "cold_target_count": test["cold_target_count"],
                    "cold_target_rate": test["cold_target_rate"],
                }
            )

        movie_output = seed_dir / "movielens"
        twotower_output = movie_output / "twotower"
        sequence_output = movie_output / "sequence"
        run(
            [args.python, "train_twotower.py", "--data_path", str(movie_csv), "--save_dir", str(twotower_output), "--epochs", str(args.epochs), "--topk", "20", "--seed", str(seed)],
            ROOT / "sequential-ranking/recall-twotower",
            env,
        )
        run(
            [args.python, "train_sequence_models.py", "--data_path", str(movie_csv), "--save_root", str(sequence_output), "--model", "all", "--epochs", str(args.epochs), "--topk", "20", "--seed", str(seed), "--max_train_samples", str(args.sequence_train_samples)],
            ROOT / "sequential-ranking/sequence-sasrec",
            env,
        )
        model_paths = {
            "twotower": twotower_output / "result.csv",
            "poprec": sequence_output / "poprec/result.csv",
            "gru4rec": sequence_output / "gru4rec/result.csv",
            "sasrec": sequence_output / "sasrec/result.csv",
        }
        for model, path in model_paths.items():
            movie_rows.append({"model": model, "seed": seed, "training_epochs": 0 if model == "poprec" else args.epochs, **csv_row(path)})

    write_movie_evidence(
        movie_rows,
        args.evidence_dir,
        args.epochs,
        args.seeds,
        args.sequence_train_samples,
    )
    write_lastfm_evidence(lastfm_rows, args.evidence_dir / "lastfm", args.epochs, args.seeds)


if __name__ == "__main__":
    main()
