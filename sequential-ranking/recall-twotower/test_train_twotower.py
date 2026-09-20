import pandas as pd

from train_twotower import leave_two_out_split


def test_leave_two_out_split_is_chronological():
    frame = pd.DataFrame(
        {
            "user_id": [0, 0, 0, 0, 1, 1, 1],
            "item_id": [10, 11, 12, 13, 20, 21, 22],
            "timestamp": [1, 2, 3, 4, 1, 2, 3],
        }
    )

    train, valid, test = leave_two_out_split(frame)

    assert valid.sort_values("user_id")["item_id"].tolist() == [12, 21]
    assert test.sort_values("user_id")["item_id"].tolist() == [13, 22]
    assert len(train) == 3
