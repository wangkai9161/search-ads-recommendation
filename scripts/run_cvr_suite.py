"""Run a configurable full-data Criteo Sponsored Search experiment."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def absolute_path(path: Path) -> Path:
    return path.absolute() if path.is_absolute() else (Path.cwd() / path).absolute()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--experiment-id", default="rtx5080-full-four-model-10ep")
    parser.add_argument("--models", default="lr,fm,wide_deep,deepfm")
    parser.add_argument("--feature-sets", default="full")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    python_path = Path(args.python).expanduser()
    if not python_path.is_absolute():
        python_path = Path.cwd() / python_path
    args.python = str(python_path.absolute())
    args.data_file = absolute_path(args.data_file)
    args.cache_dir = absolute_path(args.cache_dir)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    command = [
        args.python,
        "main.py",
        "--models",
        args.models,
        "--embedding-dims",
        "16",
        "--learning-rates",
        "0.001",
        "--seeds",
        args.seeds,
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
        args.feature_sets,
        "--device",
        "cuda",
        "--experiment-id",
        args.experiment_id,
    ]
    if args.calibration:
        command.append("--calibration")
    if args.dry_run:
        command.append("--dry-run")
    print(f"[run] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT / "ads-cvr", env=env, check=True)


if __name__ == "__main__":
    main()
