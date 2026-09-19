"""MovieLens 1M 数据读取，以及按时间构造下一物品训练样本。"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class SequenceData:
    train: list[tuple[list[int], int]]
    test: list[tuple[list[int], int]]
    num_users: int
    num_items: int


def load_sequences(
    data_dir: str | Path,
    min_history: int = 3,
    max_users: int | None = None,
    max_train_samples: int | None = None,
) -> SequenceData:
    """按用户时间顺序切分行为：最后一次行为做测试，其余前缀做训练。"""
    data_dir = Path(data_dir)
    columns = ["user_id", "item_id", "rating", "timestamp"]
    ratings = pd.read_csv(
        data_dir / "ratings.dat",
        sep="::",
        engine="python",
        names=columns,
    ).sort_values(["user_id", "timestamp"])

    user_ids = sorted(ratings["user_id"].unique())
    if max_users is not None:
        user_ids = user_ids[:max_users]

    item_ids = sorted(ratings["item_id"].unique())
    item_to_index = {item_id: index for index, item_id in enumerate(item_ids)}
    train: list[tuple[list[int], int]] = []
    test: list[tuple[list[int], int]] = []

    for user_id in user_ids:
        sequence = [
            item_to_index[item_id]
            for item_id in ratings.loc[ratings["user_id"] == user_id, "item_id"].tolist()
        ]
        if len(sequence) < min_history:
            continue
        # 最后一个物品只出现在测试目标中，避免把未来行为泄露给训练。
        test.append((sequence[:-1], sequence[-1]))
        for position in range(1, len(sequence) - 1):
            # 每个前缀都预测下一个行为，形成多个 next-item 训练样本。
            train.append((sequence[:position], sequence[position]))

    if max_train_samples is not None:
        train = train[:max_train_samples]

    return SequenceData(
        train=train,
        test=test,
        num_users=len(user_ids),
        num_items=len(item_ids),
    )
