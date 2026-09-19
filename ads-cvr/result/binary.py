"""Binary CVR metrics for the Sponsored Search conversion target."""

from __future__ import annotations

import numpy as np


def binary_logloss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-7) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1.0 - eps)
    loss = -(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
    return float(loss.mean())


def binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC AUC with average ranks, without a sklearn dependency."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=np.float64)
    start = 0
    while start < len(y_score):
        end = start + 1
        while end < len(y_score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end

    positive_rank_sum = ranks[y_true == 1].sum()
    return float(
        (positive_rank_sum - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def binary_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute PR-AUC as average precision without a sklearn dependency."""
    y_true = (np.asarray(y_true) > 0.5).astype(np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    positives = int(y_true.sum())
    if positives == 0 or len(y_true) == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    sorted_true = y_true[order]
    cumulative = np.cumsum(sorted_true)
    precision = cumulative / np.arange(1, len(sorted_true) + 1)
    return float(np.sum(precision * sorted_true) / positives)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared probability error for a binary target."""
    y_true = (np.asarray(y_true) > 0.5).astype(np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    bins: int = 10,
) -> float:
    """Expected calibration error using equally spaced probability bins."""
    if bins < 1:
        raise ValueError("bins must be at least 1")
    y_true = (np.asarray(y_true) > 0.5).astype(np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    if not len(y_true):
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        selected = (y_prob >= edges[index]) & (
            y_prob <= edges[index + 1]
            if index == bins - 1
            else y_prob < edges[index + 1]
        )
        if selected.any():
            error += float(selected.mean()) * abs(
                float(y_true[selected].mean()) - float(y_prob[selected].mean())
            )
    return error
