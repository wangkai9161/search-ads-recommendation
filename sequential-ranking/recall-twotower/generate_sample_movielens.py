# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import pandas as pd


def generate_sample_movielens(save_path, num_users=1000, num_items=2000, min_inter=20, max_inter=60, seed=42):
    rng = np.random.default_rng(seed)

    item_pop = rng.zipf(a=1.4, size=num_items).astype(np.float64)
    item_pop = item_pop / item_pop.sum()

    item_genre = rng.integers(0, 20, size=num_items)

    rows = []
    ts = 1000000000

    for u in range(num_users):
        preferred = rng.choice(20, size=3, replace=False)
        n = rng.integers(min_inter, max_inter + 1)

        item_score = item_pop.copy()
        genre_boost = np.isin(item_genre, preferred).astype(np.float64)
        item_score = item_score + 0.003 * genre_boost
        item_score = item_score / item_score.sum()

        items = rng.choice(num_items, size=n, replace=False, p=item_score)
        for item in items:
            rating = rng.choice([3, 4, 5], p=[0.2, 0.45, 0.35])
            rows.append([u, item, rating, ts])
            ts += rng.integers(1, 1000)

    df = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[OK] saved: {save_path}")
    print(f"[Info] shape: {df.shape}, users={df.user_id.nunique()}, items={df.item_id.nunique()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="./data/sample_movielens.csv")
    parser.add_argument("--num_users", type=int, default=1000)
    parser.add_argument("--num_items", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_sample_movielens(args.save_path, args.num_users, args.num_items, seed=args.seed)


if __name__ == "__main__":
    main()
