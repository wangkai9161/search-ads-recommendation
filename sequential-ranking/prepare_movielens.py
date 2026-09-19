"""Convert the official MovieLens-1M ratings file to the shared CSV schema."""

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(
        args.ratings,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(
        f"wrote {len(frame)} interactions, {frame.user_id.nunique()} users, "
        f"{frame.item_id.nunique()} items to {args.output}"
    )


if __name__ == "__main__":
    main()
