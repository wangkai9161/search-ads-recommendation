from pathlib import Path

import pandas as pd

from src.data.lastfm import load_lastfm_split


def test_lastfm_split_is_deterministic_and_leak_free(tmp_path: Path):
    data_file = tmp_path / "user_artists.dat"
    pd.DataFrame(
        {
            "userID": [1, 1, 1, 2, 2, 2],
            "artistID": [10, 11, 12, 10, 13, 14],
            "weight": [4, 3, 2, 5, 2, 1],
        }
    ).to_csv(data_file, sep="\t", index=False)

    first = load_lastfm_split(data_file, seed=7)
    second = load_lastfm_split(data_file, seed=7)

    assert first.test_items.tolist() == second.test_items.tolist()
    assert first.num_interactions == 6
    assert first.num_users == 2
    for user, target in enumerate(first.test_items):
        assert int(target) not in first.user_seen[user]
