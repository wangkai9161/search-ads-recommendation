"""LastFM HetRec implicit-feedback loading and deterministic holdout splits."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class LastFMSplit:
    train_users: np.ndarray
    train_items: np.ndarray
    test_items: np.ndarray
    user_seen: list[set[int]]
    item_popularity: np.ndarray
    num_users: int
    num_items: int
    num_interactions: int


def load_lastfm_split(data_file: str | Path, seed: int = 42) -> LastFMSplit:
    """Load ``user_artists.dat`` and hold out one artist per user.

    HetRec does not provide interaction timestamps. The test item is therefore
    sampled once with a fixed seed instead of being described as chronological.
    """

    frame = pd.read_csv(data_file, sep="\t")
    required = {"userID", "artistID", "weight"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"LastFM file is missing columns: {sorted(missing)}")

    frame = frame.drop_duplicates(["userID", "artistID"]).copy()
    counts = frame.groupby("userID")["artistID"].transform("size")
    frame = frame[counts >= 2].copy()
    frame["user"] = pd.factorize(frame["userID"], sort=True)[0]
    frame["item"] = pd.factorize(frame["artistID"], sort=True)[0]
    frame = frame.sort_values(["user", "item"]).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    test_items = np.full(frame["user"].nunique(), -1, dtype=np.int64)
    holdout_indices: list[int] = []
    for user, group in frame.groupby("user", sort=True):
        chosen = int(rng.choice(group.index.to_numpy()))
        holdout_indices.append(chosen)
        test_items[int(user)] = int(frame.at[chosen, "item"])

    train = frame.drop(index=holdout_indices)
    train_users = train["user"].to_numpy(dtype=np.int64)
    train_items = train["item"].to_numpy(dtype=np.int64)
    num_users = int(frame["user"].nunique())
    num_items = int(frame["item"].nunique())
    user_seen = [set() for _ in range(num_users)]
    for user, item in zip(train_users, train_items):
        user_seen[int(user)].add(int(item))
    item_popularity = np.bincount(train_items, minlength=num_items).astype(np.int64)

    return LastFMSplit(
        train_users=train_users,
        train_items=train_items,
        test_items=test_items,
        user_seen=user_seen,
        item_popularity=item_popularity,
        num_users=num_users,
        num_items=num_items,
        num_interactions=len(frame),
    )
