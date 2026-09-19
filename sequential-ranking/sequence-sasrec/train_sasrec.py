# -*- coding: utf-8 -*-
import os
import argparse
import random
import math
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


def build_sequences(df):
    df = df.sort_values(["user_id", "timestamp"])
    user_seqs = df.groupby("user_id")["item_id"].apply(list).to_dict()
    return user_seqs


class SASRecDataset(Dataset):
    def __init__(self, user_seqs, num_items, max_len=50, mode="train", seed=42):
        self.user_seqs = []
        self.num_items = num_items
        self.max_len = max_len
        self.mode = mode
        self.rng = np.random.default_rng(seed)

        for _, seq in user_seqs.items():
            if len(seq) < 3:
                continue
            self.user_seqs.append(seq)

    def __len__(self):
        return len(self.user_seqs)

    def sample_negative(self, seq_set):
        while True:
            item = int(self.rng.integers(1, self.num_items + 1))
            if item not in seq_set:
                return item

    def __getitem__(self, idx):
        seq = self.user_seqs[idx]
        seq_set = set(seq)

        if self.mode == "train":
            # 使用倒数第二个之前的历史预测倒数第二个，最后一个留作验证
            input_seq = seq[:-2]
            pos_item = seq[-2]
        else:
            input_seq = seq[:-1]
            pos_item = seq[-1]

        input_seq = input_seq[-self.max_len:]

        padded = np.zeros(self.max_len, dtype=np.int64)
        padded[-len(input_seq):] = np.array(input_seq, dtype=np.int64)

        neg_item = self.sample_negative(seq_set)

        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long)
        )


class SASRec(nn.Module):
    def __init__(self, num_items, max_len=50, embed_dim=64, num_heads=2, num_layers=2, dropout=0.2):
        super().__init__()

        self.num_items = num_items
        self.max_len = max_len
        self.embed_dim = embed_dim

        self.item_emb = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.pos_emb.weight)

    def forward(self, seq):
        batch_size, seq_len = seq.shape
        pos = torch.arange(seq_len, device=seq.device).unsqueeze(0).expand(batch_size, seq_len)

        x = self.item_emb(seq) + self.pos_emb(pos)
        x = self.dropout(x)

        padding_mask = seq.eq(0)

        # causal mask: 后面的位置不能看前面未来信息
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=seq.device), diagonal=1).bool()

        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        x = self.layer_norm(x)

        # 取最后一个非 padding 位置作为用户当前兴趣向量
        lengths = seq.ne(0).sum(dim=1).clamp(min=1)
        last_idx = lengths - 1
        user_vec = x[torch.arange(batch_size, device=seq.device), last_idx]

        return user_vec

    def score(self, seq, item_ids):
        user_vec = self.forward(seq)
        item_vec = self.item_emb(item_ids)
        return torch.sum(user_vec * item_vec, dim=-1)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_num = 0.0, 0

    for seq, pos_item, neg_item in loader:
        seq = seq.to(device)
        pos_item = pos_item.to(device)
        neg_item = neg_item.to(device)

        optimizer.zero_grad()

        pos_logits = model.score(seq, pos_item)
        neg_logits = model.score(seq, neg_item)

        loss = -torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-8).mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * seq.size(0)
        total_num += seq.size(0)

    return total_loss / total_num


@torch.no_grad()
def evaluate_topk(model, dataset, device, topk=20):
    model.eval()

    all_items = torch.arange(1, model.num_items + 1, dtype=torch.long, device=device)
    item_vecs = model.item_emb(all_items)

    recall_sum = 0.0
    ndcg_sum = 0.0
    count = 0

    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    for seq, pos_item, _ in loader:
        seq = seq.to(device)
        pos_item = pos_item.to(device)

        user_vec = model.forward(seq)
        scores = torch.matmul(user_vec, item_vecs.t())

        top_idx = torch.topk(scores, k=topk, dim=1).indices + 1

        for i in range(seq.size(0)):
            target = int(pos_item[i].item())
            recs = top_idx[i].detach().cpu().numpy().tolist()

            if target in recs:
                recall_sum += 1.0
                rank = recs.index(target) + 1
                ndcg_sum += 1.0 / math.log2(rank + 1.0)

            count += 1

    return recall_sum / count, ndcg_sum / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/sample_movielens.csv")
    parser.add_argument("--save_dir", type=str, default="./outputs")
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.data_path)
    df = df[df["rating"] >= 3].copy()

    df["user_id"] = pd.factorize(df["user_id"])[0]
    df["item_id"] = pd.factorize(df["item_id"])[0] + 1  # 0 留给 padding

    num_items = int(df["item_id"].max())
    user_seqs = build_sequences(df)

    train_dataset = SASRecDataset(user_seqs, num_items=num_items, max_len=args.max_len, mode="train", seed=args.seed)
    val_dataset = SASRecDataset(user_seqs, num_items=num_items, max_len=args.max_len, mode="val", seed=args.seed + 1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = SASRec(
        num_items=num_items,
        max_len=args.max_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"[Info] users={len(user_seqs)}, items={num_items}, train_users={len(train_dataset)}, device={device}")

    best_recall = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        recall, ndcg = evaluate_topk(model, val_dataset, device, topk=args.topk)

        print(f"Epoch {epoch:03d}/{args.epochs:03d} | TrainLoss={train_loss:.6f} | Recall@{args.topk}={recall:.6f} | NDCG@{args.topk}={ndcg:.6f}")

        row = {"epoch": epoch, "train_loss": train_loss, f"recall@{args.topk}": recall, f"ndcg@{args.topk}": ndcg}
        history.append(row)

        if recall > best_recall:
            best_recall = recall
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))

    pd.DataFrame(history).to_csv(os.path.join(args.save_dir, "train_history.csv"), index=False)
    best_row = history[int(np.argmax([h[f"recall@{args.topk}"] for h in history]))]
    pd.DataFrame([best_row]).to_csv(os.path.join(args.save_dir, "result.csv"), index=False)

    print(f"[Done] best_recall@{args.topk}={best_recall:.6f}")
    print(f"[Done] save_dir={args.save_dir}")


if __name__ == "__main__":
    main()
