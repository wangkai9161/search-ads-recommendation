# -*- coding: utf-8 -*-
import os
import argparse
import random
from copy import deepcopy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def leave_two_out_split(df):
    df = df.sort_values(["user_id", "timestamp"])
    counts = df.groupby("user_id")["item_id"].transform("size")
    df = df[counts >= 3].copy()
    test_rows = df.groupby("user_id").tail(1)
    remaining = df.drop(test_rows.index)
    val_rows = remaining.groupby("user_id").tail(1)
    train_rows = remaining.drop(val_rows.index)
    return (
        train_rows.reset_index(drop=True),
        val_rows.reset_index(drop=True),
        test_rows.reset_index(drop=True),
    )


class PairDataset(Dataset):
    def __init__(self, train_df, num_items, num_neg=4, seed=42, all_positive_df=None):
        self.users = train_df["user_id"].values.astype(np.int64)
        self.pos_items = train_df["item_id"].values.astype(np.int64)
        self.num_items = int(num_items)
        self.num_neg = int(num_neg)
        self.rng = np.random.default_rng(seed)
        positives = train_df if all_positive_df is None else all_positive_df
        self.user_pos = positives.groupby("user_id")["item_id"].apply(set).to_dict()

    def __len__(self):
        return len(self.users)

    def sample_negative(self, user):
        while True:
            item = int(self.rng.integers(0, self.num_items))
            if item not in self.user_pos.get(int(user), set()):
                return item

    def __getitem__(self, idx):
        user = int(self.users[idx])
        pos = int(self.pos_items[idx])

        users = [user]
        items = [pos]
        labels = [1.0]

        for _ in range(self.num_neg):
            users.append(user)
            items.append(self.sample_negative(user))
            labels.append(0.0)

        return (
            torch.tensor(users, dtype=torch.long),
            torch.tensor(items, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float32)
        )


class TwoTower(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)

        self.user_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)

    def encode_user(self, user_ids):
        z = self.user_mlp(self.user_emb(user_ids))
        return nn.functional.normalize(z, dim=-1)

    def encode_item(self, item_ids):
        z = self.item_mlp(self.item_emb(item_ids))
        return nn.functional.normalize(z, dim=-1)

    def forward(self, user_ids, item_ids):
        u = self.encode_user(user_ids)
        v = self.encode_item(item_ids)
        logits = torch.sum(u * v, dim=-1)
        return logits


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss, total_num = 0.0, 0

    for users, items, labels in loader:
        b, m = users.shape
        users = users.reshape(-1).to(device)
        items = items.reshape(-1).to(device)
        labels = labels.reshape(-1).to(device)

        optimizer.zero_grad()
        logits = model(users, items)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.numel()
        total_num += labels.numel()

    return total_loss / total_num


@torch.no_grad()
def evaluate_topk(model, history_df, target_df, num_items, device, topk=20):
    model.eval()

    all_items = torch.arange(num_items, dtype=torch.long, device=device)
    item_vecs = model.encode_item(all_items)

    user_train_pos = history_df.groupby("user_id")["item_id"].apply(set).to_dict()

    recall_sum = 0.0
    ndcg_sum = 0.0
    users = target_df["user_id"].to_numpy(dtype=np.int64)
    targets = target_df["item_id"].to_numpy(dtype=np.int64)
    batch_size = 256

    for start in range(0, len(users), batch_size):
        end = min(start + batch_size, len(users))
        batch_users = torch.from_numpy(users[start:end]).to(device)
        batch_targets = torch.from_numpy(targets[start:end]).to(device)
        scores = torch.matmul(model.encode_user(batch_users), item_vecs.t())

        # 过滤训练集中已交互物品，避免推荐用户已经看过的物品。
        for row_index, user in enumerate(users[start:end]):
            seen = user_train_pos.get(int(user), set())
            if seen:
                scores[row_index, torch.tensor(sorted(seen), dtype=torch.long, device=device)] = -torch.inf

        top_items = torch.topk(scores, k=topk, dim=1).indices
        matches = top_items.eq(batch_targets[:, None])
        recall_sum += float(matches.any(dim=1).sum().item())
        for row_index in range(len(batch_users)):
            positions = matches[row_index].nonzero(as_tuple=False)
            if positions.numel():
                rank = int(positions[0, 0].item()) + 1
                ndcg_sum += 1.0 / np.log2(rank + 1.0)

    count = len(users)
    return recall_sum / count, ndcg_sum / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/sample_movielens.csv")
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_neg", type=int, default=4)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.data_path)
    df = df[df["rating"] >= 3].copy()
    eligible = df.groupby("user_id")["item_id"].transform("size") >= 3
    df = df[eligible].copy()

    # 重新编码，保证 id 从 0 连续
    df["user_id"] = pd.factorize(df["user_id"])[0]
    df["item_id"] = pd.factorize(df["item_id"])[0]
    num_users = int(df["user_id"].nunique())
    num_items = int(df["item_id"].nunique())

    train_df, val_df, test_df = leave_two_out_split(df)

    train_dataset = PairDataset(
        train_df,
        num_items=num_items,
        num_neg=args.num_neg,
        seed=args.seed,
        all_positive_df=df,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = TwoTower(num_users, num_items, args.embed_dim, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"[Info] users={num_users}, items={num_items}, train={len(train_df)}, val={len(val_df)}, device={device}")

    best_recall = -1.0
    best_state = None
    best_epoch = None
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        recall, ndcg = evaluate_topk(model, train_df, val_df, num_items, device, topk=args.topk)

        print(f"Epoch {epoch:03d}/{args.epochs:03d} | TrainLoss={train_loss:.6f} | Recall@{args.topk}={recall:.6f} | NDCG@{args.topk}={ndcg:.6f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            f"val_recall@{args.topk}": recall,
            f"val_ndcg@{args.topk}": ndcg,
        })

        if recall > best_recall:
            best_recall = recall
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))

    model.load_state_dict(best_state)
    test_history = pd.concat([train_df, val_df], ignore_index=True)
    test_recall, test_ndcg = evaluate_topk(
        model, test_history, test_df, num_items, device, topk=args.topk
    )
    best_history = history[best_epoch - 1]
    result = {
        "epoch": best_epoch,
        "train_loss": best_history["train_loss"],
        f"val_recall@{args.topk}": best_history[f"val_recall@{args.topk}"],
        f"val_ndcg@{args.topk}": best_history[f"val_ndcg@{args.topk}"],
        f"test_recall@{args.topk}": test_recall,
        f"test_ndcg@{args.topk}": test_ndcg,
    }
    pd.DataFrame(history).to_csv(os.path.join(args.save_dir, "train_history.csv"), index=False)
    pd.DataFrame([result]).to_csv(os.path.join(args.save_dir, "result.csv"), index=False)

    print(
        f"[Done] best_val_recall@{args.topk}={best_recall:.6f} "
        f"test_recall@{args.topk}={test_recall:.6f}"
    )
    print(f"[Done] save_dir={args.save_dir}")


if __name__ == "__main__":
    main()
