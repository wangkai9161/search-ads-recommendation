# -*- coding: utf-8 -*-

import os
import json
import random
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss


# ============================================================
# 1. Basic Utils
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    return device


def make_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_config(config, save_dir):
    path = os.path.join(save_dir, "din_v2_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"[Save Config] {path}")


# ============================================================
# 2. Data Processing
# ============================================================

def load_movielens(data_path):
    df = pd.read_csv(
        data_path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"]
    )
    return df


def add_label_and_time(df, threshold=4):
    df = df.copy()
    df["label"] = (df["rating"] >= threshold).astype(np.float32)
    df["hour"] = pd.to_datetime(df["timestamp"], unit="s").dt.hour.astype(np.int64)
    return df


def encode_user_item(df):
    df = df.copy()

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df["user_id"] = user_encoder.fit_transform(df["user_id"])

    # Reserve movie_id = 0 for padding.
    df["movie_id"] = item_encoder.fit_transform(df["movie_id"]) + 1

    num_users = int(df["user_id"].max() + 1)
    num_items = int(df["movie_id"].max() + 1)

    return df, num_users, num_items, user_encoder, item_encoder


def build_din_samples(
    df,
    max_history_len=20,
    positive_history_only=True,
    min_history_len=1
):
    """
    Build DIN samples by user temporal order.

    Each sample:
        user_id
        target_movie_id
        hist_movie_ids
        hist_len
        hour
        label
        timestamp

    If positive_history_only=True, only rating>=4 historical movies are added into history.
    """

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    rows = []

    for user_id, group in df.groupby("user_id"):
        history = []

        for _, row in group.iterrows():
            target_movie_id = int(row["movie_id"])
            hour = int(row["hour"])
            label = float(row["label"])
            timestamp = int(row["timestamp"])

            if len(history) >= min_history_len:
                hist = history[-max_history_len:]
                hist_len = len(hist)

                if hist_len < max_history_len:
                    hist = hist + [0] * (max_history_len - hist_len)

                rows.append({
                    "user_id": int(user_id),
                    "target_movie_id": target_movie_id,
                    "hist_movie_ids": hist,
                    "hist_len": hist_len,
                    "hour": hour,
                    "label": label,
                    "timestamp": timestamp
                })

            if positive_history_only:
                if label > 0.5:
                    history.append(target_movie_id)
            else:
                history.append(target_movie_id)

    samples = pd.DataFrame(rows)
    return samples


def split_by_user_time(samples, train_ratio=0.8, valid_ratio=0.1):
    """
    Split samples by each user's temporal order.

    This is more suitable for user behavior prediction:
        train: earlier interactions of each user
        valid: middle interactions of each user
        test : later interactions of each user
    """

    train_parts = []
    valid_parts = []
    test_parts = []

    samples = samples.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    for user_id, group in samples.groupby("user_id"):
        group = group.sort_values("timestamp").reset_index(drop=True)
        n = len(group)

        if n < 3:
            train_parts.append(group)
            continue

        n_train = max(int(train_ratio * n), 1)
        n_valid = max(int(valid_ratio * n), 1)

        if n_train + n_valid >= n:
            n_train = max(n - 2, 1)
            n_valid = 1

        train_parts.append(group.iloc[:n_train])
        valid_parts.append(group.iloc[n_train:n_train + n_valid])
        test_parts.append(group.iloc[n_train + n_valid:])

    train_df = pd.concat(train_parts, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
    valid_df = pd.concat(valid_parts, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
    test_df = pd.concat(test_parts, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)

    return train_df, valid_df, test_df


def prepare_data(
    data_path,
    max_history_len=20,
    positive_history_only=True,
    min_history_len=1
):
    raw_df = load_movielens(data_path)
    raw_df = add_label_and_time(raw_df, threshold=4)
    raw_df, num_users, num_items, user_encoder, item_encoder = encode_user_item(raw_df)

    samples = build_din_samples(
        df=raw_df,
        max_history_len=max_history_len,
        positive_history_only=positive_history_only,
        min_history_len=min_history_len
    )

    train_df, valid_df, test_df = split_by_user_time(samples)

    info = {
        "num_users": num_users,
        "num_items": num_items,
        "num_hours": 24,
        "max_history_len": max_history_len,
        "positive_history_only": positive_history_only,
        "min_history_len": min_history_len,
        "user_encoder": user_encoder,
        "item_encoder": item_encoder
    }

    return raw_df, samples, train_df, valid_df, test_df, info


def print_data_info(raw_df, samples, train_df, valid_df, test_df, info):
    print("\n[Raw Data]")
    print(raw_df.head())

    print("\n[Sequence Samples]")
    print(samples.head())

    print("\n[Data Info]")
    print(f"Raw samples          : {len(raw_df)}")
    print(f"DIN samples          : {len(samples)}")
    print(f"Positive ratio       : {samples['label'].mean():.6f}")
    print(f"Number of users      : {info['num_users']}")
    print(f"Number of items      : {info['num_items']} including padding id 0")
    print(f"Max history length   : {info['max_history_len']}")
    print(f"Positive history only: {info['positive_history_only']}")

    print("\n[Split]")
    print(f"Train: {len(train_df)}, pos ratio = {train_df['label'].mean():.6f}")
    print(f"Valid: {len(valid_df)}, pos ratio = {valid_df['label'].mean():.6f}")
    print(f"Test : {len(test_df)}, pos ratio = {test_df['label'].mean():.6f}")


# ============================================================
# 3. Dataset and Dataloader
# ============================================================

class DINDataset(Dataset):
    def __init__(self, df):
        self.user_id = df["user_id"].values.astype(np.int64)
        self.target_movie_id = df["target_movie_id"].values.astype(np.int64)
        self.hist_movie_ids = np.stack(df["hist_movie_ids"].values).astype(np.int64)
        self.hist_len = df["hist_len"].values.astype(np.int64)
        self.hour = df["hour"].values.astype(np.int64)
        self.label = df["label"].values.astype(np.float32)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            "user_id": torch.tensor(self.user_id[idx], dtype=torch.long),
            "target_movie_id": torch.tensor(self.target_movie_id[idx], dtype=torch.long),
            "hist_movie_ids": torch.tensor(self.hist_movie_ids[idx], dtype=torch.long),
            "hist_len": torch.tensor(self.hist_len[idx], dtype=torch.long),
            "hour": torch.tensor(self.hour[idx], dtype=torch.long),
            "label": torch.tensor(self.label[idx], dtype=torch.float32)
        }


def build_dataloaders(
    train_df,
    valid_df,
    test_df,
    train_batch_size=1024,
    eval_batch_size=2048
):
    train_loader = DataLoader(
        DINDataset(train_df),
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0
    )

    valid_loader = DataLoader(
        DINDataset(valid_df),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        DINDataset(test_df),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, valid_loader, test_loader


# ============================================================
# 4. DIN Model
# ============================================================

class DINAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dims=(32, 16), dropout=0.2):
        super().__init__()

        input_dim = 4 * embed_dim

        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        self.att_mlp = nn.Sequential(*layers)

    def forward(self, hist_emb, target_emb, mask):
        """
        hist_emb:   [B, L, D]
        target_emb: [B, D]
        mask:       [B, L], True for valid history
        """

        _, seq_len, _ = hist_emb.shape

        target_expand = target_emb.unsqueeze(1).expand(-1, seq_len, -1)

        att_input = torch.cat(
            [
                hist_emb,
                target_expand,
                hist_emb - target_expand,
                hist_emb * target_expand
            ],
            dim=-1
        )

        att_scores = self.att_mlp(att_input).squeeze(-1)
        att_scores = att_scores.masked_fill(~mask, -1e9)

        att_weights = torch.softmax(att_scores, dim=1)
        interest = torch.sum(hist_emb * att_weights.unsqueeze(-1), dim=1)

        return interest, att_weights


class DIN(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        num_hours=24,
        embed_dim=16,
        att_hidden_dims=(32, 16),
        mlp_hidden_dims=(64, 32),
        dropout=0.4
    ):
        super().__init__()

        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.item_embedding = nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.hour_embedding = nn.Embedding(num_hours, embed_dim)

        self.attention = DINAttention(
            embed_dim=embed_dim,
            hidden_dims=att_hidden_dims,
            dropout=dropout
        )

        mlp_input_dim = embed_dim * 4

        layers = []
        input_dim = mlp_input_dim

        for hidden_dim in mlp_hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        self.mlp = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_embedding.weight.data)
        nn.init.xavier_uniform_(self.item_embedding.weight.data)
        nn.init.xavier_uniform_(self.hour_embedding.weight.data)

        with torch.no_grad():
            self.item_embedding.weight.data[0].fill_(0.0)

    def forward(self, batch, return_attention=False):
        user_id = batch["user_id"]
        target_movie_id = batch["target_movie_id"]
        hist_movie_ids = batch["hist_movie_ids"]
        hour = batch["hour"]

        user_emb = self.user_embedding(user_id)
        target_emb = self.item_embedding(target_movie_id)
        hist_emb = self.item_embedding(hist_movie_ids)
        hour_emb = self.hour_embedding(hour)

        mask = hist_movie_ids > 0

        interest_emb, att_weights = self.attention(
            hist_emb=hist_emb,
            target_emb=target_emb,
            mask=mask
        )

        x = torch.cat(
            [
                user_emb,
                target_emb,
                interest_emb,
                hour_emb
            ],
            dim=-1
        )

        logits = self.mlp(x).squeeze(1)
        pred = torch.sigmoid(logits)

        if return_attention:
            return pred, att_weights

        return pred


def build_model(info, device, config):
    model = DIN(
        num_users=info["num_users"],
        num_items=info["num_items"],
        num_hours=info["num_hours"],
        embed_dim=config["embed_dim"],
        att_hidden_dims=config["att_hidden_dims"],
        mlp_hidden_dims=config["mlp_hidden_dims"],
        dropout=config["dropout"]
    ).to(device)

    print("\n[Model]")
    print(model)

    return model


# ============================================================
# 5. Training and Evaluation
# ============================================================

def move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        labels = batch["label"]

        pred = model(batch)
        loss = criterion(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    preds = []
    labels = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        y = batch["label"]

        pred = model(batch)
        loss = criterion(pred, y)

        total_loss += loss.item() * y.size(0)

        preds.append(pred.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)

    preds_clip = np.clip(preds, 1e-7, 1.0 - 1e-7)

    auc = roc_auc_score(labels, preds)
    logloss = log_loss(labels, preds_clip)
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, auc, logloss, preds, labels


class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-5, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
            return True

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
            return True

        self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return False


def train_model(
    model,
    train_loader,
    valid_loader,
    device,
    save_dir,
    num_epochs=30,
    lr=1e-3,
    weight_decay=1e-5,
    patience=3
):
    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    stopper = EarlyStopping(
        patience=patience,
        min_delta=1e-5,
        mode="max"
    )

    best_auc = -1.0
    best_epoch = -1
    best_path = os.path.join(save_dir, "din_v2_movielens_best.pth")

    history = {
        "train_loss": [],
        "valid_loss": [],
        "valid_auc": [],
        "valid_logloss": []
    }

    print("\n[Training]")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device
        )

        valid_loss, valid_auc, valid_logloss, _, _ = evaluate(
            model=model,
            loader=valid_loader,
            criterion=criterion,
            device=device
        )

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        history["valid_auc"].append(valid_auc)
        history["valid_logloss"].append(valid_logloss)

        print(
            f"Epoch [{epoch:02d}/{num_epochs}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Valid Loss: {valid_loss:.6f} | "
            f"Valid AUC: {valid_auc:.6f} | "
            f"Valid LogLoss: {valid_logloss:.6f}"
        )

        improved = stopper.step(valid_auc)

        if improved:
            best_auc = valid_auc
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
            print(f"[Save Best] epoch={epoch}, valid_auc={valid_auc:.6f}")

        if stopper.should_stop:
            print(f"[Early Stop] epoch={epoch}, best_epoch={best_epoch}, best_auc={best_auc:.6f}")
            break

    print(f"\n[Best Valid AUC] {best_auc:.6f}")
    print(f"[Best Epoch] {best_epoch}")
    print(f"[Save Best Model] {best_path}")

    return history, best_path, best_auc, best_epoch


def test_model(model, test_loader, device, best_path):
    criterion = nn.BCELoss()

    model.load_state_dict(torch.load(best_path, map_location=device))

    test_loss, test_auc, test_logloss, test_preds, test_labels = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device
    )

    print("\n[Test Result]")
    print(f"Test Loss    : {test_loss:.6f}")
    print(f"Test AUC     : {test_auc:.6f}")
    print(f"Test LogLoss : {test_logloss:.6f}")

    return {
        "test_loss": test_loss,
        "test_auc": test_auc,
        "test_logloss": test_logloss,
        "test_preds": test_preds,
        "test_labels": test_labels
    }


# ============================================================
# 6. Recommendation
# ============================================================

@torch.no_grad()
def recommend_topk(
    model,
    user_id,
    history_movie_ids,
    candidate_movie_ids,
    hour,
    device,
    max_history_len=20,
    topk=10
):
    model.eval()

    hist = history_movie_ids[-max_history_len:]
    hist_len = len(hist)

    if hist_len < max_history_len:
        hist = hist + [0] * (max_history_len - hist_len)

    batch = {
        "user_id": torch.tensor([user_id] * len(candidate_movie_ids), dtype=torch.long).to(device),
        "target_movie_id": torch.tensor(candidate_movie_ids, dtype=torch.long).to(device),
        "hist_movie_ids": torch.tensor([hist] * len(candidate_movie_ids), dtype=torch.long).to(device),
        "hist_len": torch.tensor([hist_len] * len(candidate_movie_ids), dtype=torch.long).to(device),
        "hour": torch.tensor([hour] * len(candidate_movie_ids), dtype=torch.long).to(device)
    }

    scores = model(batch).detach().cpu().numpy()

    result = pd.DataFrame({
        "user_id": user_id,
        "target_movie_id": candidate_movie_ids,
        "ctr_score": scores
    })

    result = result.sort_values("ctr_score", ascending=False).reset_index(drop=True)

    return result.head(topk), result


def run_recommendation(model, samples, info, device, save_dir):
    row = samples.iloc[-1]

    user_id = int(row["user_id"])
    history_movie_ids = [int(x) for x in row["hist_movie_ids"] if int(x) > 0]
    hour = int(row["hour"])

    candidate_movie_ids = list(range(1, min(300, info["num_items"])))

    top10, full_rank = recommend_topk(
        model=model,
        user_id=user_id,
        history_movie_ids=history_movie_ids,
        candidate_movie_ids=candidate_movie_ids,
        hour=hour,
        device=device,
        max_history_len=info["max_history_len"],
        topk=10
    )

    top10_path = os.path.join(save_dir, "top10_din_v2_movielens.csv")
    full_rank_path = os.path.join(save_dir, "full_rank_din_v2_movielens.csv")

    top10.to_csv(top10_path, index=False, encoding="utf-8-sig")
    full_rank.to_csv(full_rank_path, index=False, encoding="utf-8-sig")

    print("\n[Top-10 Recommendation]")
    print(top10)

    print(f"\n[Save] {top10_path}")
    print(f"[Save] {full_rank_path}")

    return top10, full_rank


# ============================================================
# 7. Visualization
# ============================================================

def plot_training_curves(history, save_dir):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    plt.plot(epochs, history["valid_loss"], marker="s", label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DIN V2 Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    path = os.path.join(save_dir, "din_v2_loss_curve.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Save Figure] {path}")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["valid_auc"], marker="o", label="Valid AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.title("DIN V2 Validation AUC")
    plt.legend()
    plt.grid(True)
    path = os.path.join(save_dir, "din_v2_valid_auc_curve.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Save Figure] {path}")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["valid_logloss"], marker="o", label="Valid LogLoss")
    plt.xlabel("Epoch")
    plt.ylabel("LogLoss")
    plt.title("DIN V2 Validation LogLoss")
    plt.legend()
    plt.grid(True)
    path = os.path.join(save_dir, "din_v2_valid_logloss_curve.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Save Figure] {path}")


def plot_prediction_distribution(test_preds, test_labels, save_dir):
    pos_preds = test_preds[test_labels == 1]
    neg_preds = test_preds[test_labels == 0]

    plt.figure(figsize=(8, 5))
    plt.hist(neg_preds, bins=50, alpha=0.6, label="Label 0")
    plt.hist(pos_preds, bins=50, alpha=0.6, label="Label 1")
    plt.xlabel("Predicted CTR")
    plt.ylabel("Count")
    plt.title("DIN V2 Prediction Distribution")
    plt.legend()
    plt.grid(True)

    path = os.path.join(save_dir, "din_v2_prediction_distribution.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Save Figure] {path}")


def plot_prediction_samples(test_preds, test_labels, save_dir, num_samples=200):
    n = min(num_samples, len(test_preds))

    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(n), test_labels[:n], marker="o", linestyle="none", label="True Label")
    plt.plot(np.arange(n), test_preds[:n], marker="x", linestyle="none", label="Predicted CTR")
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    plt.title("DIN V2 Predicted CTR vs True Label")
    plt.legend()
    plt.grid(True)

    path = os.path.join(save_dir, "din_v2_prediction_samples.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Save Figure] {path}")


def plot_topk_recommendation(top10, save_dir):
    plt.figure(figsize=(8, 5))
    plt.bar(
        top10["target_movie_id"].astype(str),
        top10["ctr_score"]
    )
    plt.xlabel("Movie ID")
    plt.ylabel("Predicted CTR")
    plt.title("DIN V2 Top-10 Recommendation Scores")
    plt.xticks(rotation=45)
    plt.grid(axis="y")

    path = os.path.join(save_dir, "din_v2_top10_recommendation_scores.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Save Figure] {path}")


@torch.no_grad()
def plot_attention_example(model, samples, device, save_dir):
    model.eval()

    row = samples.iloc[-1]

    batch = {
        "user_id": torch.tensor([int(row["user_id"])], dtype=torch.long).to(device),
        "target_movie_id": torch.tensor([int(row["target_movie_id"])], dtype=torch.long).to(device),
        "hist_movie_ids": torch.tensor([row["hist_movie_ids"]], dtype=torch.long).to(device),
        "hist_len": torch.tensor([int(row["hist_len"])], dtype=torch.long).to(device),
        "hour": torch.tensor([int(row["hour"])], dtype=torch.long).to(device)
    }

    pred, att_weights = model(batch, return_attention=True)

    hist_items = np.array(row["hist_movie_ids"])
    att = att_weights.detach().cpu().numpy().reshape(-1)

    valid_mask = hist_items > 0
    hist_items = hist_items[valid_mask]
    att = att[valid_mask]

    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(att)), att)
    plt.xticks(np.arange(len(att)), hist_items.astype(str), rotation=45)
    plt.xlabel("History Movie ID")
    plt.ylabel("Attention Weight")
    plt.title(f"DIN V2 Attention Weights, Target Movie ID = {int(row['target_movie_id'])}")
    plt.grid(axis="y")

    path = os.path.join(save_dir, "din_v2_attention_example.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[Save Figure] {path}")


def save_all_figures(history, test_result, top10, model, samples, device, save_dir):
    plot_training_curves(history, save_dir)

    plot_prediction_distribution(
        test_preds=test_result["test_preds"],
        test_labels=test_result["test_labels"],
        save_dir=save_dir
    )

    plot_prediction_samples(
        test_preds=test_result["test_preds"],
        test_labels=test_result["test_labels"],
        save_dir=save_dir,
        num_samples=200
    )

    plot_topk_recommendation(top10, save_dir)

    plot_attention_example(
        model=model,
        samples=samples,
        device=device,
        save_dir=save_dir
    )


# ============================================================
# 8. Save Results
# ============================================================

def save_experiment_results(best_auc, best_epoch, test_result, save_dir):
    result_path = os.path.join(save_dir, "din_v2_movielens_result.txt")

    with open(result_path, "w", encoding="utf-8") as f:
        f.write(f"Best Valid AUC: {best_auc:.6f}\n")
        f.write(f"Best Epoch    : {best_epoch}\n")
        f.write(f"Test Loss     : {test_result['test_loss']:.6f}\n")
        f.write(f"Test AUC      : {test_result['test_auc']:.6f}\n")
        f.write(f"Test LogLoss  : {test_result['test_logloss']:.6f}\n")

    print(f"\n[Save Result] {result_path}")

    return result_path


# ============================================================
# 9. Main
# ============================================================

def main():
    set_seed(42)

    root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root, "data/movielens/ml-1m/ratings.dat")
    save_dir = make_dir(os.path.join(root, "outputs", "din-v2"))

    config = {
        "max_history_len": 20,
        "positive_history_only": True,
        "min_history_len": 1,
        "embed_dim": 16,
        "att_hidden_dims": (32, 16),
        "mlp_hidden_dims": (64, 32),
        "dropout": 0.4,
        "num_epochs": 30,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "patience": 3,
        "train_batch_size": 1024,
        "eval_batch_size": 2048
    }

    print("=" * 80)
    print("[Project] MovieLens 1M DIN V2 CTR Baseline")
    print(f"[Data] {data_path}")
    print(f"[Save Dir] {save_dir}")
    print("=" * 80)

    save_config(config, save_dir)

    device = get_device()

    raw_df, samples, train_df, valid_df, test_df, info = prepare_data(
        data_path=data_path,
        max_history_len=config["max_history_len"],
        positive_history_only=config["positive_history_only"],
        min_history_len=config["min_history_len"]
    )

    print_data_info(
        raw_df=raw_df,
        samples=samples,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        info=info
    )

    train_loader, valid_loader, test_loader = build_dataloaders(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        train_batch_size=config["train_batch_size"],
        eval_batch_size=config["eval_batch_size"]
    )

    model = build_model(
        info=info,
        device=device,
        config=config
    )

    history, best_path, best_auc, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        save_dir=save_dir,
        num_epochs=config["num_epochs"],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        patience=config["patience"]
    )

    test_result = test_model(
        model=model,
        test_loader=test_loader,
        device=device,
        best_path=best_path
    )

    save_experiment_results(
        best_auc=best_auc,
        best_epoch=best_epoch,
        test_result=test_result,
        save_dir=save_dir
    )

    top10, _ = run_recommendation(
        model=model,
        samples=samples,
        info=info,
        device=device,
        save_dir=save_dir
    )

    save_all_figures(
        history=history,
        test_result=test_result,
        top10=top10,
        model=model,
        samples=samples,
        device=device,
        save_dir=save_dir
    )

    print("\n[Done]")


if __name__ == "__main__":
    main()
