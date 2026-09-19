"""Project entry point for Criteo Sponsored Search CVR experiments."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from model.registry import build_model
from prepare.reader import FEATURE_SETS, read_sponsored_search
from result.calibration import save_calibration_report
from result.evaluate import evaluate_cvr, predict_cvr
from result.storage import save_training_result
from result.visualize import save_cvr_visualizations, save_tuning_comparison
from train.train_model import SUPPORTED_MODELS, TrainConfig, train_model
from train.tuning import (
    build_trial_configs,
    select_best,
)


ROOT = Path(__file__).resolve().parent

# Experiment defaults. Command-line arguments override these values.
# Implemented models: "lr", "fm", "wide_deep", "deepfm", "dssm", "din",
# "dien", "esmm", "transformer".
# Models currently trainable with the click-level Criteo CVR reader:
# "lr", "fm", "wide_deep", "deepfm".
DEFAULT_MODELS = ("deepfm",)
DEFAULT_EMBEDDING_DIMS = (8, 16)
DEFAULT_LEARNING_RATES = (5e-4, 1e-3)
DEFAULT_SEEDS = (42,)
DEFAULT_EPOCHS = 8
DEFAULT_BATCH_SIZE = 65_536
DEFAULT_MAX_ROWS = 200_000  # Set to 0 in the CLI for the full dataset.
DEFAULT_STATS_ROWS = 1_000_000
DEFAULT_NUM_SPARSE_BINS = 100_000
DEFAULT_TRAIN_FRACTION = 0.7
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_SPLIT_MODE = "time"
DEFAULT_FEATURE_SETS = ("full",)
DEFAULT_EARLY_STOPPING_PATIENCE = 3
DEFAULT_EARLY_STOPPING_MIN_DELTA = 1e-5
DEFAULT_DEVICE = "auto"
DEFAULT_MAX_RUNS = None
DEFAULT_DATA_PATH = ROOT / "data/criteo-sponsored-search/data.tsv"


def _parse_list(raw: str, cast, name: str):
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(cast(token))
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(f"invalid {name} value: {token!r}") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{name} must contain at least one value")
    return tuple(values)


def _parse_models(raw: str) -> tuple[str, ...]:
    models = tuple(item.lower().strip() for item in raw.split(",") if item.strip())
    unsupported = sorted(set(models) - set(SUPPORTED_MODELS))
    if unsupported:
        raise argparse.ArgumentTypeError(
            f"unsupported model(s): {', '.join(unsupported)}; choose from {', '.join(SUPPORTED_MODELS)}"
        )
    if not models:
        raise argparse.ArgumentTypeError("models must contain at least one model")
    return models


def _parse_limit(value: str) -> int | None:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("row limits must be >= 0 (use 0 for all rows)")
    return None if parsed == 0 else parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate all experiment parameters in the project entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", type=_parse_models, default=DEFAULT_MODELS, help="comma-separated CVR models")
    parser.add_argument("--embedding-dims", type=lambda value: _parse_list(value, int, "embedding dimension"), default=DEFAULT_EMBEDDING_DIMS, help="comma-separated embedding dimensions")
    parser.add_argument("--learning-rates", type=lambda value: _parse_list(value, float, "learning rate"), default=DEFAULT_LEARNING_RATES, help="comma-separated learning rates")
    parser.add_argument("--seeds", type=lambda value: _parse_list(value, int, "seed"), default=DEFAULT_SEEDS, help="comma-separated random seeds")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="epochs per trial")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="streaming batch size")
    parser.add_argument("--max-rows", type=_parse_limit, default=DEFAULT_MAX_ROWS, help="rows per trial; 0 means all rows")
    parser.add_argument("--stats-rows", type=_parse_limit, default=DEFAULT_STATS_ROWS, help="rows for normalization statistics; 0 means all rows")
    parser.add_argument("--num-sparse-bins", type=int, default=DEFAULT_NUM_SPARSE_BINS, help="hash vocabulary size")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH, help="Criteo TSV path")
    parser.add_argument("--cache-dir", type=Path, default=None, help="optional mmap cache directory for repeated full-data experiments")
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION, help="validation fraction")
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION, help="chronological training fraction")
    parser.add_argument("--split-mode", choices=("time", "row"), default=DEFAULT_SPLIT_MODE, help="data split strategy")
    parser.add_argument("--feature-sets", type=lambda value: _parse_list(value, str, "feature set"), default=DEFAULT_FEATURE_SETS, help="comma-separated feature ablations")
    parser.add_argument("--early-stopping-patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE, help="validation LogLoss patience")
    parser.add_argument("--early-stopping-min-delta", type=float, default=DEFAULT_EARLY_STOPPING_MIN_DELTA, help="minimum validation LogLoss improvement")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=DEFAULT_DEVICE, help="torch device")
    parser.add_argument("--max-runs", type=int, default=DEFAULT_MAX_RUNS, help="cap expanded grid size")
    parser.add_argument("--experiment-id", type=str, default=None, help="unique result namespace; defaults to UTC timestamp")
    parser.add_argument("--calibration", action="store_true", help="fit validation temperature and save calibration/lift reports for the selected trial")
    parser.add_argument("--dry-run", action="store_true", help="validate and print trials without reading data")
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    return args


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_sparse_bins < 2:
        parser.error("--num-sparse-bins must be at least 2")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be between 0 and 1")
    if args.split_mode == "time" and not 0.0 < args.train_fraction < 1.0:
        parser.error("--train-fraction must be between 0 and 1")
    if args.split_mode == "time" and args.train_fraction + args.val_fraction >= 1.0:
        parser.error("--train-fraction + --val-fraction must be below 1")
    unknown_feature_sets = sorted(set(args.feature_sets) - set(FEATURE_SETS))
    if unknown_feature_sets:
        parser.error(f"unknown feature set(s): {', '.join(unknown_feature_sets)}")
    if args.early_stopping_patience < 1:
        parser.error("--early-stopping-patience must be at least 1")
    if args.early_stopping_min_delta < 0:
        parser.error("--early-stopping-min-delta must be >= 0")
    if args.experiment_id is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.experiment_id):
        parser.error("--experiment-id may contain only letters, numbers, '.', '_' and '-'")
    if args.max_runs is not None and args.max_runs < 1:
        parser.error("--max-runs must be at least 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is not available")
    if not args.dry_run and not args.data_path.is_file():
        parser.error(f"data file not found: {args.data_path}")


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _number(value: object) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _write_summary(
    rows: list[dict[str, object]],
    best: dict[str, object] | None,
    args: argparse.Namespace,
    experiment_dir: Path,
    calibration_path: Path | None = None,
) -> None:
    result_dir = experiment_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "trial", "run_name", "model", "feature_set", "task", "target", "embedding_dim",
        "learning_rate", "seed", "status", "error", "epoch", "train_loss", "train_logloss",
        "val_loss", "val_logloss", "val_auc", "val_pr_auc", "val_brier",
        "val_ece", "val_positive_rate", "best_epoch", "last_epoch", "elapsed_seconds",
        "test_loss", "test_logloss", "test_auc", "test_pr_auc", "test_brier",
        "test_ece", "test_positive_rate",
        "checkpoint",
    ]
    with (result_dir / "cvr_tuning_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "task": "cvr",
        "target": "Sale",
        "experiment_id": args.experiment_id,
        "objective": "minimize val_logloss; tie-break maximize val_pr_auc then val_auc",
        "parameters": {
            "models": list(args.models),
            "embedding_dims": list(args.embedding_dims),
            "learning_rates": list(args.learning_rates),
            "seeds": list(args.seeds),
            "feature_sets": list(args.feature_sets),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_rows": args.max_rows,
            "stats_rows": args.stats_rows,
            "num_sparse_bins": args.num_sparse_bins,
            "data_path": str(args.data_path),
            "cache_dir": str(args.cache_dir) if args.cache_dir else None,
            "val_fraction": args.val_fraction,
            "train_fraction": args.train_fraction,
            "split_mode": args.split_mode,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "device": args.device,
            "calibration": args.calibration,
        },
        "best_run": best,
        "calibration_report": str(calibration_path) if calibration_path else None,
        "runs": rows,
    }
    (result_dir / "cvr_tuning_summary.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )

    lines = [
        "# CVR Hyper-parameter Search",
        "",
        "Target: `Sale` after click. Objective: minimize `val_logloss`, then maximize `val_pr_auc` and `val_auc`.",
        "",
        "| Trial | Model | Features | Embedding | Learning rate | Seed | Status | Val LogLoss | Val PR-AUC | Val AUC | Test LogLoss | Test PR-AUC | Test AUC |",
        "| ---: | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        def fmt(key: str) -> str:
            value = _number(row.get(key))
            return "-" if value is None else f"{value:.6f}"

        lines.append(
            f"| {row.get('trial', '-')} | {row.get('model', '-')} | {row.get('feature_set', '-')} | "
            f"{row.get('embedding_dim', '-')} | {row.get('learning_rate', '-')} | {row.get('seed', '-')} | "
            f"{row.get('status', '-')} | {fmt('val_logloss')} | {fmt('val_pr_auc')} | {fmt('val_auc')} | "
            f"{fmt('test_logloss')} | {fmt('test_pr_auc')} | {fmt('test_auc')} |"
        )
    lines.extend(["", "## Selected Trial", ""])
    if best:
        lines.extend([
            f"- Run: `{best.get('run_name', '-')}`",
            f"- Model: `{best.get('model', '-')}`",
            f"- Validation LogLoss: `{_number(best.get('val_logloss')):.6f}`",
            f"- Validation PR-AUC: `{_number(best.get('val_pr_auc')):.6f}`",
            f"- Validation ROC-AUC: `{_number(best.get('val_auc')):.6f}`",
        ])
    else:
        lines.append("No successful trial with finite validation LogLoss.")
    (result_dir / "cvr_tuning_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    """Execute prepare -> train -> result -> output for every grid trial."""
    experiment_id = args.experiment_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.experiment_id = experiment_id
    experiment_dir = ROOT / "result" / "experiments" / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    base_config = TrainConfig(
        task="cvr",
        target="Sale",
        data_path=str(args.data_path),
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        stats_rows=args.stats_rows,
        num_sparse_bins=args.num_sparse_bins,
        val_fraction=args.val_fraction,
        train_fraction=args.train_fraction,
        split_mode=args.split_mode,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        run_tag=experiment_id,
        device=args.device,
    )
    configs = build_trial_configs(
        base_config=base_config,
        models=args.models,
        embedding_dims=args.embedding_dims,
        learning_rates=args.learning_rates,
        seeds=args.seeds,
        feature_sets=args.feature_sets,
        run_tag=experiment_id,
        max_runs=args.max_runs,
    )
    print(
        f"confirmed experiment={experiment_id} task=cvr target=Sale trials={len(configs)} data={args.data_path} "
        f"device={_resolve_device(args.device)}"
    )
    for index, config in enumerate(configs, start=1):
        print(
            f"trial={index}/{len(configs)} model={config.model} feature_set={config.feature_set} "
            f"embedding_dim={config.embedding_dim} learning_rate={config.learning_rate:g} seed={config.seed}"
        )
    if args.dry_run:
        return {"trials": len(configs), "dry_run": True}

    prepared: dict[str, tuple[object, object]] = {}
    resolved_device = _resolve_device(args.device)
    rows: list[dict[str, object]] = []
    for index, config in enumerate(configs, start=1):
        started_at = perf_counter()
        row: dict[str, object] = {
            "trial": index,
            "run_name": config.run_name,
            "model": config.model,
            "feature_set": config.feature_set,
            "task": config.task,
            "target": config.target,
            "embedding_dim": config.embedding_dim,
            "learning_rate": config.learning_rate,
            "seed": config.seed,
            "status": "failed",
        }
        try:
            if config.feature_set not in prepared:
                prepared[config.feature_set] = read_sponsored_search(
                    args.data_path,
                    num_sparse_bins=args.num_sparse_bins,
                    chunksize=args.batch_size,
                    max_rows=args.max_rows,
                    stats_rows=args.stats_rows,
                    split_mode=args.split_mode,
                    train_fraction=args.train_fraction,
                    val_fraction=args.val_fraction,
                    feature_set=config.feature_set,
                    cache_dir=args.cache_dir,
                )
                prepared_reader, _ = prepared[config.feature_set]
                print(
                    f"prepared feature_set={config.feature_set} rows={args.max_rows or 'all'} "
                    f"sparse_fields={prepared_reader.num_sparse_fields} dense_fields={prepared_reader.num_dense} "
                    f"sparse_bins={prepared_reader.num_sparse_bins}"
                )
            reader, stats = prepared[config.feature_set]
            model, history = train_model(
                config,
                reader=reader,
                stats=stats,
                save_artifacts=False,
            )
            valid_metrics = evaluate_cvr(model, reader, stats, resolved_device, config.val_fraction, split="valid")
            test_metrics = evaluate_cvr(model, reader, stats, resolved_device, config.val_fraction, split="test")
            best_epoch = min(history, key=lambda item: float(item["val_logloss"]))["epoch"]
            metrics = {**valid_metrics, **test_metrics, "best_epoch": best_epoch, "last_epoch": history[-1]["epoch"]}
            config_values = asdict(config)
            config_values["resolved_device"] = str(resolved_device)
            if reader.time_split is not None:
                config_values["time_split_boundaries"] = asdict(reader.time_split)
            _, saved_metrics = save_training_result(
                ROOT,
                config.run_name or f"sponsored-search-{config.model}",
                model,
                history,
                config_values,
                started_at,
                metrics=metrics,
            )
            save_cvr_visualizations(ROOT, config.run_name or config.model, history, saved_metrics)
            row.update(saved_metrics)
            row["status"] = "ok"
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"trial={index} status=failed error={row['error']}")
        rows.append(row)
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

    best = select_best(rows)
    save_tuning_comparison(ROOT, rows, output_dir=experiment_dir)
    calibration_path = None
    if args.calibration and best:
        best_config = next(config for config in configs if config.run_name == best["run_name"])
        best_reader, best_stats = prepared[best_config.feature_set]
        best_model = build_model(
            best_config.model,
            num_sparse_bins=best_reader.num_sparse_bins,
            num_sparse_fields=best_reader.num_sparse_fields,
            num_dense=best_reader.num_dense,
            embedding_dim=best_config.embedding_dim,
        ).to(resolved_device)
        checkpoint = ROOT / "result" / str(best["run_name"]) / "checkpoints" / "model_final.pth"
        best_model.load_state_dict(torch.load(checkpoint, map_location=resolved_device))
        val_labels, val_logits, _ = predict_cvr(best_model, best_reader, best_stats, resolved_device, split="valid")
        test_labels, test_logits, _ = predict_cvr(best_model, best_reader, best_stats, resolved_device, split="test")
        calibration_path = save_calibration_report(
            experiment_dir / "calibration",
            val_labels,
            val_logits,
            test_labels,
            test_logits,
        )
        print(f"saved calibration report {calibration_path}")
    _write_summary(rows, best, args, experiment_dir, calibration_path=calibration_path)
    if best:
        print(
            f"best_run={best['run_name']} val_logloss={float(best['val_logloss']):.6f} "
            f"val_pr_auc={float(best['val_pr_auc']):.6f}"
        )
    print(f"saved {experiment_dir / 'cvr_tuning_summary.json'} trials={len(rows)}")
    return {
        "experiment_id": experiment_id,
        "trials": len(rows),
        "successful": sum(row.get("status") == "ok" for row in rows),
        "best_run": best,
        "calibration_report": str(calibration_path) if calibration_path else None,
    }


def main(argv: list[str] | None = None) -> dict[str, object]:
    """Parse parameters, validate them, and delegate the experiment pipeline."""
    return run_experiment(parse_args(argv))


if __name__ == "__main__":
    main()
