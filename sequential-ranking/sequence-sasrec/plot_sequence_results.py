# -*- coding: utf-8 -*-
# =============================================================================
# plot_sequence_results.py
# Visualize PopRec / GRU4Rec / SASRec results
# =============================================================================

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def load_summary(save_root):
    summary_path = os.path.join(save_root, "summary_results.csv")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Not found: {summary_path}")

    return pd.read_csv(summary_path)


def plot_bar(df, metric, save_root):
    if metric not in df.columns:
        print(f"[Skip] metric not found: {metric}")
        return

    plot_df = df.sort_values(metric, ascending=False)

    plt.figure(figsize=(7, 5))
    plt.bar(plot_df["model"], plot_df[metric])

    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.title(f"Model Comparison by {metric}")

    for i, v in enumerate(plot_df[metric].values):
        plt.text(i, v, f"{v:.4f}", ha="center", va="bottom")

    plt.tight_layout()

    save_path = os.path.join(save_root, f"{metric.replace('@', '_at_')}_bar.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Save] {save_path}")


def plot_curve(save_root, models, metric):
    plt.figure(figsize=(8, 5))

    has_data = False

    for model in models:
        history_path = os.path.join(save_root, model, "train_history.csv")

        if not os.path.exists(history_path):
            continue

        df = pd.read_csv(history_path)

        if metric not in df.columns:
            continue

        if model == "poprec":
            continue

        plt.plot(df["epoch"], df[metric], marker="o", label=model)
        has_data = True

    if not has_data:
        return

    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(metric)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    save_path = os.path.join(save_root, f"{metric.replace('@', '_at_')}_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"[Save] {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", type=str, default="./outputs_compare")
    parser.add_argument("--topk", type=int, default=20)
    args = parser.parse_args()

    df = load_summary(args.save_root)

    recall_metric = f"recall@{args.topk}"
    ndcg_metric = f"ndcg@{args.topk}"

    plot_bar(df, recall_metric, args.save_root)
    plot_bar(df, ndcg_metric, args.save_root)

    models = df["model"].tolist()

    plot_curve(args.save_root, models, recall_metric)
    plot_curve(args.save_root, models, ndcg_metric)
    plot_curve(args.save_root, models, "train_loss")


if __name__ == "__main__":
    main()
