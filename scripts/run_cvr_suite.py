"""Run the full-data four-model Criteo Sponsored Search comparison."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--experiment-id", default="rtx5080-full-four-model-10ep")
    args = parser.parse_args()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    command = [
        args.python,
        "main.py",
        "--models",
        "lr,fm,wide_deep,deepfm",
        "--embedding-dims",
        "16",
        "--learning-rates",
        "0.001",
        "--seeds",
        "42",
        "--epochs",
        str(args.epochs),
        "--max-rows",
        "0",
        "--stats-rows",
        "0",
        "--data-path",
        str(args.data_file),
        "--cache-dir",
        str(args.cache_dir),
        "--split-mode",
        "time",
        "--feature-sets",
        "full",
        "--device",
        "cuda",
        "--experiment-id",
        args.experiment_id,
        "--calibration",
    ]
    print(f"[run] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT / "ads-cvr", env=env, check=True)


if __name__ == "__main__":
    main()
