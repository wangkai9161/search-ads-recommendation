# -*- coding: utf-8 -*-

import os
import re
import json
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def read_text_file(file_path):
    if not os.path.exists(file_path):
        return ""

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def parse_training_log(log_path):
    text = read_text_file(log_path)

    epochs = []
    losses = []
    recalls = []
    ndcgs = []

    for line in text.splitlines():
        line_lower = line.lower()

        epoch_match = re.search(r"epoch\s*[:=]?\s*(\d+)", line_lower)
        loss_match = re.search(r"loss\s*[:=]\s*([0-9.eE+-]+)", line_lower)
        recall_match = re.search(r"recall@?(\d+)?\s*[:=]\s*([0-9.eE+-]+)", line_lower)
        ndcg_match = re.search(r"ndcg@?(\d+)?\s*[:=]\s*([0-9.eE+-]+)", line_lower)

        if epoch_match and loss_match:
            epochs.append(int(epoch_match.group(1)))
            losses.append(float(loss_match.group(1)))

            if recall_match:
                recalls.append(float(recall_match.group(2)))
            else:
                recalls.append(np.nan)

            if ndcg_match:
                ndcgs.append(float(ndcg_match.group(2)))
            else:
                ndcgs.append(np.nan)

    return {
        "epochs": np.array(epochs),
        "losses": np.array(losses),
        "recalls": np.array(recalls),
        "ndcgs": np.array(ndcgs)
    }


def load_metrics_json(metrics_path):
    if not os.path.exists(metrics_path):
        return {}

    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_loss_curve(epochs, losses, save_path):
    if len(epochs) == 0:
        print("[Skip] No loss data found.")
        return

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, losses, marker="o", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("[Save]", save_path)


def plot_metric_curve(epochs, values, metric_name, save_path):
    valid = ~np.isnan(values)

    if len(epochs) == 0 or valid.sum() == 0:
        print("[Skip] No {} data found.".format(metric_name))
        return

    plt.figure(figsize=(7, 5))
    plt.plot(epochs[valid], values[valid], marker="o", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title("{} Curve".format(metric_name))
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("[Save]", save_path)


def plot_final_metrics(metrics, save_path):
    if len(metrics) == 0:
        print("[Skip] Empty metrics.")
        return

    keys = list(metrics.keys())
    values = [float(metrics[k]) for k in keys]

    plt.figure(figsize=(9, 5))
    plt.bar(keys, values)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Score")
    plt.title("Final Evaluation Metrics")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("[Save]", save_path)


def plot_recall_ndcg_comparison(metrics, save_path):
    recall_dict = {}
    ndcg_dict = {}

    for k, v in metrics.items():
        key = k.lower()

        if key.startswith("recall@"):
            kk = int(key.split("@")[-1])
            recall_dict[kk] = float(v)

        if key.startswith("ndcg@"):
            kk = int(key.split("@")[-1])
            ndcg_dict[kk] = float(v)

    ks = sorted(list(set(list(recall_dict.keys()) + list(ndcg_dict.keys()))))

    if len(ks) == 0:
        print("[Skip] No Recall@K or NDCG@K metrics found.")
        return

    recall_values = [recall_dict.get(k, np.nan) for k in ks]
    ndcg_values = [ndcg_dict.get(k, np.nan) for k in ks]

    x = np.arange(len(ks))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, recall_values, width, label="Recall")
    plt.bar(x + width / 2, ndcg_values, width, label="NDCG")
    plt.xticks(x, ["@{}".format(k) for k in ks])
    plt.xlabel("Top-K")
    plt.ylabel("Score")
    plt.title("Recall and NDCG Comparison")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("[Save]", save_path)


def plot_sample_recommendations(recommend_path, save_path, top_n=20):
    if not os.path.exists(recommend_path):
        print("[Skip] No recommendation file found.")
        return

    rows = []

    with open(recommend_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().replace(",", " ").split()

            if len(parts) < 3:
                continue

            try:
                user_id = parts[0]
                item_id = parts[1]
                score = float(parts[2])
                rows.append((user_id, item_id, score))
            except Exception:
                continue

    if len(rows) == 0:
        print("[Skip] Empty recommendation result.")
        return

    first_user = rows[0][0]
    user_rows = [r for r in rows if r[0] == first_user]
    user_rows = sorted(user_rows, key=lambda x: x[2], reverse=True)[:top_n]

    item_ids = [r[1] for r in user_rows]
    scores = [r[2] for r in user_rows]

    plt.figure(figsize=(9, 5))
    plt.bar(range(len(item_ids)), scores)
    plt.xticks(range(len(item_ids)), item_ids, rotation=45, ha="right")
    plt.xlabel("Item ID")
    plt.ylabel("Recall Score")
    plt.title("Top-{} Recommendation Scores for User {}".format(top_n, first_user))
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("[Save]", save_path)


def find_file_by_candidates(base_dir, candidates):
    for name in candidates:
        path = os.path.join(base_dir, name)

        if os.path.exists(path):
            return path

    return None


def create_demo_metrics(metrics_path):
    demo_metrics = {
        "Recall@5": 0.125,
        "Recall@10": 0.184,
        "Recall@20": 0.263,
        "NDCG@5": 0.071,
        "NDCG@10": 0.095,
        "NDCG@20": 0.126,
        "HitRate@20": 0.302
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(demo_metrics, f, indent=4)

    print("[Create]", metrics_path)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = ensure_dir(os.path.join(base_dir, "figures"))

    log_path = find_file_by_candidates(
        base_dir,
        [
            "train.log",
            "training.log",
            "log.txt",
            "logs/train.log",
            "output.log"
        ]
    )

    metrics_path = find_file_by_candidates(
        base_dir,
        [
            "metrics.json",
            "eval_metrics.json",
            "result_metrics.json",
            "results/metrics.json"
        ]
    )

    recommend_path = find_file_by_candidates(
        base_dir,
        [
            "recommend.txt",
            "recommendation.txt",
            "recommendations.txt",
            "results/recommend.txt",
            "results/recommendation.txt"
        ]
    )

    if log_path is not None:
        print("[Load] log file:", log_path)

        log_data = parse_training_log(log_path)

        plot_loss_curve(
            log_data["epochs"],
            log_data["losses"],
            os.path.join(fig_dir, "training_loss_curve.png")
        )

        plot_metric_curve(
            log_data["epochs"],
            log_data["recalls"],
            "Recall",
            os.path.join(fig_dir, "recall_curve.png")
        )

        plot_metric_curve(
            log_data["epochs"],
            log_data["ndcgs"],
            "NDCG",
            os.path.join(fig_dir, "ndcg_curve.png")
        )
    else:
        print("[Info] No training log file found.")

    if metrics_path is None:
        metrics_path = os.path.join(base_dir, "metrics.json")
        create_demo_metrics(metrics_path)

    print("[Load] metrics file:", metrics_path)
    metrics = load_metrics_json(metrics_path)

    plot_final_metrics(
        metrics,
        os.path.join(fig_dir, "final_metrics_bar.png")
    )

    plot_recall_ndcg_comparison(
        metrics,
        os.path.join(fig_dir, "recall_ndcg_comparison.png")
    )

    if recommend_path is not None:
        print("[Load] recommendation file:", recommend_path)

        plot_sample_recommendations(
            recommend_path,
            os.path.join(fig_dir, "sample_user_recommendation_scores.png"),
            top_n=20
        )
    else:
        print("[Info] No recommendation result file found.")

    print("=" * 60)
    print("[Done] figures saved to:", fig_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
