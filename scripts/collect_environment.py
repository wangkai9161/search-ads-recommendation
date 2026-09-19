"""Capture reproducibility metadata without storing datasets or credentials."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, action="append", default=[])
    args = parser.parse_args()

    payload = {
        "recorded_at": command("date", "-Is"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "git_commit": command("git", "rev-parse", "HEAD"),
        "datasets": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in args.dataset
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
