# -*- coding: utf-8 -*-
# =============================================================================
# train_sequence_models.py
# Sequence Recommendation: PopRec / GRU4Rec / SASRec
# =============================================================================

import os
import math
import argparse
import random
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


class SeqDataset(Dataset):
    def __init__(self, user_seqs, num_items, max_len=50, mode="train", seed=42):
        self.user_seqs = []
        self.num_items = int(num_items)
        self.max_len = int(max_len)
        self.mode = mode
        self.rng = np.random.default_rng(seed)

        for _, seq in user_seqs.items():
            if len(seq) >= 3:
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
            input_seq = seq[:-2]
            pos_item = seq[-2]
        else:
            input_seq = seq[:-1]
            pos_item = seq[-1]

        input_seq = input_seq[-self.max_len:]

        padded = np.zeros(self.max_len, dtype=np.int64)

        if len(input_seq) > 0:
            padded[-len(input_seq):] = np.array(input_seq, dtype=np.int64)

        neg_item = self.sample_negative(seq_set)

        return (
            torch.tensor(padded, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long)
        )


class GRU4Rec(nn.Module):
    def __init__(self, num_items, embed_dim=64, hidden_dim=64, num_layers=1, dropout=0.2):
        super().__init__()

        self.num_items = int(num_items)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)

        self.item_emb = nn.Embedding(num_items + 1, embed_dim, padding_idx=0)

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.proj = nn.Linear(hidden_dim, embed_dim)

        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, seq):
        emb = self.item_emb(seq)

        lengths = seq.ne(0).sum(dim=1).clamp(min=1).cpu()

        packed = nn.utils.rnn.pack_padded_sequence(
            emb,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False
        )

        _, h = self.gru(packed)
        h = h[-1]

        user_vec = self.proj(h)
        user_vec = nn.functional.normalize(user_vec, dim=-1)

        return user_vec

    def score(self, seq, item_ids):
        user_vec = self.forward(seq)
        item_vec = self.item_emb(item_ids)
        item_vec = nn.functional.normalize(item_vec, dim=-1)
        logits = torch.sum(user_vec * item_vec, dim=-1)
        return logits


class SASRec(nn.Module):
    def __init__(self, num_items, max_len=50, embed_dim=64, num_heads=2, num_layers=2, dropout=0.2):
        super().__init__()

        self.num_items = int(num_items)
        self.max_len = int(max_len)
        self.embed_dim = int(embed_dim)

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

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

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

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=seq.device),
            diagonal=1
        ).bool()

        x = self.encoder(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        x = self.layer_norm(x)

        lengths = seq.ne(0).sum(dim=1).clamp(min=1)
        last_idx = lengths - 1

        user_vec = x[torch.arange(batch_size, device=seq.device), last_idx]
        user_vec = nn.functional.normalize(user_vec, dim=-1)

        return user_vec

    def score(self, seq, item_ids):
        user_vec = self.forward(seq)
        item_vec = self.item_emb(item_ids)
        item_vec = nn.functional.normalize(item_vec, dim=-1)
        logits = torch.sum(user_vec * item_vec, dim=-1)
        return logits


def train_one_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_num = 0

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
def evaluate_deep_model(model, dataset, device, topk=20):
    model.eval()

    all_items = torch.arange(1, model.num_items + 1, dtype=torch.long, device=device)
    item_vecs = model.item_emb(all_items)
    item_vecs = nn.functional.normalize(item_vecs, dim=-1)

    recall_sum = 0.0
    ndcg_sum = 0.0
    count = 0

    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    for seq, pos_item, _ in loader:
        seq = seq.to(device)
        pos_item = pos_item.to(device)

        user_vec = model.forward(seq)
        scores = torch.matmul(user_vec, item_vecs.t())

        top_items = torch.topk(scores, k=topk, dim=1).indices + 1

        for i in range(seq.size(0)):
            target = int(pos_item[i].item())
            recs = top_items[i].cpu().numpy().tolist()

            if target in recs:
                recall_sum += 1.0
                rank = recs.index(target) + 1
                ndcg_sum += 1.0 / math.log2(rank + 1.0)

            count += 1

    return recall_sum / count, ndcg_sum / count


def evaluate_poprec(train_df, val_df, topk=20):
    popular_items = train_df["item_id"].value_counts().index.tolist()
    top_items = popular_items[:topk]

    recall_sum = 0.0
    ndcg_sum = 0.0
    count = 0

    for _, row in val_df.iterrows():
        target = int(row["item_id"])

        if target in top_items:
            recall_sum += 1.0
            rank = top_items.index(target) + 1
            ndcg_sum += 1.0 / math.log2(rank + 1.0)

        count += 1

    return recall_sum / count, ndcg_sum / count


def prepare_data(data_path):
    df = pd.read_csv(data_path)
    df = df[df["rating"] >= 3].copy()

    df["user_id"] = pd.factorize(df["user_id"])[0]
    df["item_id"] = pd.factorize(df["item_id"])[0] + 1

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    val_df = df.groupby("user_id").tail(1)
    train_df = df.drop(val_df.index).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    num_items = int(df["item_id"].max())
    user_seqs = build_sequences(df)

    return df, train_df, val_df, user_seqs, num_items


def build_model(model_name, num_items, max_len, embed_dim, hidden_dim, num_heads, num_layers, dropout):
    model_name = model_name.lower()

    if model_name == "gru4rec":
        return GRU4Rec(
            num_items=num_items,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=1,
            dropout=dropout
        )

    if model_name == "sasrec":
        return SASRec(
            num_items=num_items,
            max_len=max_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )

    raise ValueError(f"Unknown model_name: {model_name}")


def train_deep_model(args, model_name, user_seqs, num_items, device):
    save_dir = os.path.join(args.save_root, model_name)
    os.makedirs(save_dir, exist_ok=True)

    train_dataset = SeqDataset(
        user_seqs=user_seqs,
        num_items=num_items,
        max_len=args.max_len,
        mode="train",
        seed=args.seed
    )

    val_dataset = SeqDataset(
        user_seqs=user_seqs,
        num_items=num_items,
        max_len=args.max_len,
        mode="val",
        seed=args.seed + 1
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    model = build_model(
        model_name=model_name,
        num_items=num_items,
        max_len=args.max_len,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_recall = -1.0
    best_row = None
    history = []

    print("=" * 80)
    print(model)
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        recall, ndcg = evaluate_deep_model(
            model=model,
            dataset=val_dataset,
            device=device,
            topk=args.topk
        )

        row = {
            "model": model_name,
            "epoch": epoch,
            "train_loss": train_loss,
            f"recall@{args.topk}": recall,
            f"ndcg@{args.topk}": ndcg
        }

        history.append(row)

        print(
            f"[{model_name}] Epoch {epoch:03d}/{args.epochs:03d} | "
            f"TrainLoss={train_loss:.6f} | "
            f"Recall@{args.topk}={recall:.6f} | "
            f"NDCG@{args.topk}={ndcg:.6f}"
        )

        if recall > best_recall:
            best_recall = recall
            best_row = row
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))

    pd.DataFrame(history).to_csv(os.path.join(save_dir, "train_history.csv"), index=False)
    pd.DataFrame([best_row]).to_csv(os.path.join(save_dir, "result.csv"), index=False)

    print(f"[Done] {model_name}, best_recall@{args.topk}={best_recall:.6f}")

    return best_row


def run_poprec(args, train_df, val_df):
    save_dir = os.path.join(args.save_root, "poprec")
    os.makedirs(save_dir, exist_ok=True)

    recall, ndcg = evaluate_poprec(
        train_df=train_df,
        val_df=val_df,
        topk=args.topk
    )

    row = {
        "model": "poprec",
        "epoch": 0,
        "train_loss": np.nan,
        f"recall@{args.topk}": recall,
        f"ndcg@{args.topk}": ndcg
    }

    pd.DataFrame([row]).to_csv(os.path.join(save_dir, "result.csv"), index=False)
    pd.DataFrame([row]).to_csv(os.path.join(save_dir, "train_history.csv"), index=False)

    print(
        f"[poprec] Recall@{args.topk}={recall:.6f} | "
        f"NDCG@{args.topk}={ndcg:.6f}"
    )

    return row


def collect_summary(save_root, models):
    rows = []

    for model in models:
        result_path = os.path.join(save_root, model, "result.csv")

        if os.path.exists(result_path):
            rows.append(pd.read_csv(result_path))

    if len(rows) == 0:
        return

    summary = pd.concat(rows, axis=0, ignore_index=True)
    summary_path = os.path.join(save_root, "summary_results.csv")
    summary.to_csv(summary_path, index=False)

    print("=" * 80)
    print("[Summary]")
    print(summary)
    print(f"[Save] {summary_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default="./data/sample_movielens.csv")
    parser.add_argument("--save_root", type=str, default="./outputs_compare")
    parser.add_argument("--model", type=str, default="all", choices=["poprec", "gru4rec", "sasrec", "all"])

    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
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

    os.makedirs(args.save_root, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df, train_df, val_df, user_seqs, num_items = prepare_data(args.data_path)

    print(
        f"[Info] users={df['user_id'].nunique()}, "
        f"items={num_items}, "
        f"interactions={len(df)}, "
        f"device={device}"
    )

    run_models = ["poprec", "gru4rec", "sasrec"] if args.model == "all" else [args.model]

    for model_name in run_models:
        if model_name == "poprec":
            run_poprec(args, train_df, val_df)
        else:
            train_deep_model(args, model_name, user_seqs, num_items, device)

    collect_summary(args.save_root, run_models)


if __name__ == "__main__":
    main()
