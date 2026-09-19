"""Pure helpers for constructing and ranking CVR training trials."""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import replace
from typing import Iterable

import numpy as np

from .train_model import SUPPORTED_MODELS, TrainConfig


# The root ``main.py`` can override these values from the command line.
DEFAULT_MODELS = ("deepfm",)
DEFAULT_EMBEDDING_DIMS = (8, 16)
DEFAULT_LEARNING_RATES = (5e-4, 1e-3)
DEFAULT_SEEDS = (42,)
DEFAULT_FEATURE_SETS = ("full",)


def build_trial_configs(
    *,
    base_config: TrainConfig | None = None,
    models: Iterable[str] = DEFAULT_MODELS,
    embedding_dims: Iterable[int] = DEFAULT_EMBEDDING_DIMS,
    learning_rates: Iterable[float] = DEFAULT_LEARNING_RATES,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    feature_sets: Iterable[str] = DEFAULT_FEATURE_SETS,
    run_tag: str | None = None,
    max_runs: int | None = None,
) -> list[TrainConfig]:
    """Expand a grid into independent ``TrainConfig`` objects."""
    base = base_config or TrainConfig()
    models = tuple(str(model).lower().strip() for model in models)
    unsupported = sorted(set(models) - set(SUPPORTED_MODELS))
    if unsupported:
        raise ValueError(f"unsupported model(s): {', '.join(unsupported)}")
    embedding_dims = tuple(int(value) for value in embedding_dims)
    learning_rates = tuple(float(value) for value in learning_rates)
    seeds = tuple(int(value) for value in seeds)
    feature_sets = tuple(str(value).lower().strip() for value in feature_sets)
    if not models or not embedding_dims or not learning_rates or not seeds or not feature_sets:
        raise ValueError("every tuning dimension must contain at least one value")
    if any(value < 1 for value in embedding_dims):
        raise ValueError("embedding dimensions must be positive")
    if any(value <= 0 for value in learning_rates):
        raise ValueError("learning rates must be positive")
    combinations = list(itertools.product(models, embedding_dims, learning_rates, seeds, feature_sets))
    if max_runs is not None:
        if max_runs < 1:
            raise ValueError("max_runs must be at least 1")
        combinations = combinations[:max_runs]
    configs = []
    for model, embedding_dim, learning_rate, seed, feature_set in combinations:
        configs.append(
            replace(
                base,
                model=model,
                embedding_dim=embedding_dim,
                learning_rate=learning_rate,
                seed=seed,
                feature_set=feature_set,
                run_name=_trial_name(model, embedding_dim, learning_rate, seed, feature_set, run_tag),
                run_tag=run_tag,
            )
        )
    return configs


def select_best(results: Iterable[dict[str, object]]) -> dict[str, object] | None:
    """Select the best successful trial by LogLoss, PR-AUC, then ROC-AUC."""
    valid = [
        row
        for row in results
        if row.get("status") == "ok" and _number(row.get("val_logloss")) is not None
    ]
    return min(valid, key=_selection_key) if valid else None


def _trial_name(
    model: str,
    embedding_dim: int,
    learning_rate: float,
    seed: int,
    feature_set: str = "full",
    run_tag: str | None = None,
) -> str:
    suffix = "" if feature_set == "full" else f"-f{feature_set}"
    tag = f"-r{run_tag}" if run_tag else ""
    return f"sponsored-search-tune-{model}-e{embedding_dim}-lr{_token(learning_rate)}-s{seed}{suffix}{tag}"


def _token(value: float | int) -> str:
    text = str(value).lower()
    return re.sub(r"[^a-zA-Z0-9]+", "p", text).strip("p") or "value"


def _selection_key(row: dict[str, object]) -> tuple[float, float, float]:
    return (
        _number(row.get("val_logloss")) if _number(row.get("val_logloss")) is not None else math.inf,
        -_number(row.get("val_pr_auc")) if _number(row.get("val_pr_auc")) is not None else math.inf,
        -_number(row.get("val_auc")) if _number(row.get("val_auc")) is not None else math.inf,
    )


def _number(value: object) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


__all__ = [
    "DEFAULT_MODELS",
    "DEFAULT_EMBEDDING_DIMS",
    "DEFAULT_LEARNING_RATES",
    "DEFAULT_SEEDS",
    "DEFAULT_FEATURE_SETS",
    "build_trial_configs",
    "select_best",
]
