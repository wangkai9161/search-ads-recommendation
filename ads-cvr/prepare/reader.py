"""Streaming reader and feature preparation for Criteo sponsored-search logs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


COLUMNS = [
    "Sale",
    "SalesAmountInEuro",
    "time_delay_for_conversion",
    "click_timestamp",
    "nb_clicks_1week",
    "product_price",
    "product_age_group",
    "device_type",
    "audience_id",
    "product_gender",
    "product_brand",
    "product_category_1",
    "product_category_2",
    "product_category_3",
    "product_category_4",
    "product_category_5",
    "product_category_6",
    "product_category_7",
    "product_country",
    "product_id",
    "product_title",
    "partner_id",
    "user_id",
]

SPARSE_COLUMNS = [
    "product_age_group",
    "device_type",
    "audience_id",
    "product_gender",
    "product_brand",
    "product_category_1",
    "product_category_2",
    "product_category_3",
    "product_category_4",
    "product_category_5",
    "product_category_6",
    "product_category_7",
    "product_country",
    "product_id",
    "partner_id",
    "user_id",
]

# Revenue and delay are post-conversion fields, so they are deliberately not
# included as prediction features. They remain available as training targets.
DENSE_COLUMNS = ["click_timestamp", "nb_clicks_1week", "product_price"]


@dataclass(frozen=True)
class DenseStats:
    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class TimeSplit:
    """Timestamp boundaries for chronological train/validation/test splits."""

    train_end: float
    valid_end: float


@dataclass
class SponsoredSearchBatch:
    sparse: np.ndarray
    dense: np.ndarray
    labels: np.ndarray
    delay_seconds: np.ndarray
    observed_conversion: np.ndarray


def _hash_column(values: pd.Series, num_bins: int, namespace: str) -> np.ndarray:
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")
    values = values.astype("string")
    missing = values.isna() | values.eq("") | values.eq("-1")
    hashed = pd.util.hash_pandas_object(values.fillna("-1"), index=False).to_numpy(dtype=np.uint64)
    namespace_hash = pd.util.hash_pandas_object(
        pd.Series([namespace], dtype="string"), index=False
    ).to_numpy(dtype=np.uint64)[0]
    hashed ^= namespace_hash
    result = (hashed % (num_bins - 1) + 1).astype(np.int64)
    result[missing.to_numpy()] = 0
    return result


class SponsoredSearchReader:
    """Read the 6.4 GB file in bounded-memory chunks.

    The reader hashes anonymized categorical values into a fixed embedding
    vocabulary and z-score normalizes only pre-click numeric features.
    """

    def __init__(
        self,
        path: str | Path = "data/criteo-sponsored-search/data.tsv",
        num_sparse_bins: int = 100_000,
        chunksize: int = 65_536,
        max_rows: int | None = None,
        sparse_columns: tuple[str, ...] | list[str] | None = None,
        dense_columns: tuple[str, ...] | list[str] | None = None,
    ):
        self.path = Path(path)
        self.num_sparse_bins = num_sparse_bins
        self.chunksize = chunksize
        self.max_rows = max_rows
        self.sparse_columns = tuple(SPARSE_COLUMNS if sparse_columns is None else sparse_columns)
        self.dense_columns = tuple(DENSE_COLUMNS if dense_columns is None else dense_columns)
        unknown = (set(self.sparse_columns) | set(self.dense_columns)) - set(COLUMNS)
        if unknown:
            raise ValueError(f"unknown feature columns: {sorted(unknown)}")
        self.num_sparse_fields = len(self.sparse_columns)
        self.num_dense = len(self.dense_columns)
        self.time_split: TimeSplit | None = None

    def iter_raw(self) -> Iterator[pd.DataFrame]:
        if not self.path.is_file():
            raise FileNotFoundError(f"sponsored-search data not found: {self.path}")
        remaining = self.max_rows
        reader = pd.read_csv(
            self.path,
            sep="\t",
            header=None,
            names=COLUMNS,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            chunksize=self.chunksize,
        )
        for chunk in reader:
            if remaining is not None:
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining].copy()
                remaining -= len(chunk)
            if len(chunk):
                yield chunk

    def fit_time_split(
        self,
        train_fraction: float = 0.7,
        val_fraction: float = 0.15,
        sample_size: int = 1_000_000,
    ) -> TimeSplit:
        """Estimate chronological boundaries from click timestamps only.

        A deterministic reservoir keeps memory bounded for the full 16M-row file
        while providing stable quantile boundaries.
        """
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1")
        if not 0.0 < val_fraction < 1.0 or train_fraction + val_fraction >= 1.0:
            raise ValueError("train_fraction + val_fraction must be below 1")
        sample_size = max(10_000, int(sample_size))
        reservoir = np.empty(sample_size, dtype=np.float64)
        filled = 0
        seen = 0
        rng = np.random.default_rng(42)
        for chunk in self.iter_timestamps():
            values = pd.to_numeric(chunk, errors="coerce").to_numpy(dtype=np.float64)
            values = values[np.isfinite(values) & (values > 0)]
            if not len(values):
                continue
            seen += len(values)
            combined = np.concatenate((reservoir[:filled], values))
            if len(combined) <= sample_size:
                reservoir[:len(combined)] = combined
                filled = len(combined)
            else:
                selected = rng.choice(len(combined), size=sample_size, replace=False)
                reservoir[:] = combined[selected]
                filled = sample_size
        if filled < 10:
            raise ValueError("not enough valid click timestamps for a time split")
        sample = np.sort(reservoir[:filled])
        train_end = float(np.quantile(sample, train_fraction))
        valid_end = float(np.quantile(sample, train_fraction + val_fraction))
        if not train_end < valid_end:
            raise ValueError("timestamp quantiles did not produce distinct split boundaries")
        self.time_split = TimeSplit(train_end=train_end, valid_end=valid_end)
        return self.time_split

    def iter_timestamps(self) -> Iterator[pd.Series]:
        """Stream only the timestamp column for fast boundary estimation."""
        if not self.path.is_file():
            raise FileNotFoundError(f"sponsored-search data not found: {self.path}")
        remaining = self.max_rows
        reader = pd.read_csv(
            self.path,
            sep="\t",
            header=None,
            usecols=[3],
            names=["click_timestamp"],
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            chunksize=self.chunksize,
        )
        for chunk in reader:
            if remaining is not None:
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining].copy()
                remaining -= len(chunk)
            if len(chunk):
                yield chunk["click_timestamp"]

    def _split_ids(self, chunk: pd.DataFrame) -> np.ndarray:
        if self.time_split is None:
            raise ValueError("fit_time_split must be called before using time splits")
        timestamps = pd.to_numeric(chunk["click_timestamp"], errors="coerce").to_numpy(dtype=np.float64)
        split = np.zeros(len(chunk), dtype=np.int8)
        valid_timestamp = np.isfinite(timestamps) & (timestamps > 0)
        split[valid_timestamp & (timestamps > self.time_split.train_end)] = 1
        split[valid_timestamp & (timestamps > self.time_split.valid_end)] = 2
        return split

    def fit_dense_stats(self, max_rows: int | None = None, split: str | None = None) -> DenseStats:
        if self.num_dense == 0:
            return DenseStats(np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32))
        totals = np.zeros(self.num_dense, dtype=np.float64)
        squares = np.zeros(self.num_dense, dtype=np.float64)
        counts = np.zeros(self.num_dense, dtype=np.int64)
        scanned = 0
        for chunk in self.iter_raw():
            if split is not None:
                split_id = self._split_ids(chunk)
                wanted = _split_id(split)
                chunk = chunk.iloc[split_id == wanted]
                if not len(chunk):
                    continue
            if max_rows is not None and scanned >= max_rows:
                break
            if max_rows is not None and scanned + len(chunk) > max_rows:
                chunk = chunk.iloc[: max_rows - scanned]
            scanned += len(chunk)
            values, valid = self._numeric_values(chunk)
            for index in range(self.num_dense):
                current = values[:, index][valid[:, index]]
                if current.size:
                    totals[index] += current.sum(dtype=np.float64)
                    squares[index] += np.square(current, dtype=np.float64).sum(dtype=np.float64)
                    counts[index] += current.size
        if not counts.any():
            raise ValueError("no valid numeric features found in sponsored-search data")
        mean = totals / np.maximum(counts, 1)
        variance = np.maximum(squares / np.maximum(counts, 1) - mean * mean, 0.0)
        std = np.sqrt(variance)
        std[std < 1e-8] = 1.0
        return DenseStats(mean.astype(np.float32), std.astype(np.float32))

    def _numeric_values(self, chunk: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        values = np.zeros((len(chunk), self.num_dense), dtype=np.float64)
        valid = np.zeros_like(values, dtype=bool)
        for index, name in enumerate(self.dense_columns):
            series = pd.to_numeric(chunk[name], errors="coerce")
            values[:, index] = series.fillna(0.0).to_numpy(dtype=np.float64)
            valid[:, index] = series.notna().to_numpy()
            if name == "click_timestamp":
                valid[:, index] &= values[:, index] > 0
            else:
                valid[:, index] &= values[:, index] >= 0
        return values, valid

    def transform(self, chunk: pd.DataFrame, stats: DenseStats) -> SponsoredSearchBatch:
        sparse = (
            np.column_stack(
                [_hash_column(chunk[name], self.num_sparse_bins, name) for name in self.sparse_columns]
            )
            if self.sparse_columns else np.empty((len(chunk), 0), dtype=np.int64)
        )
        dense, valid = self._numeric_values(chunk)
        for index in range(self.num_dense):
            dense[~valid[:, index], index] = stats.mean[index]
        dense = ((dense - stats.mean) / stats.std).astype(np.float32)

        labels = pd.to_numeric(chunk["Sale"], errors="coerce").fillna(0).clip(0, 1).to_numpy(dtype=np.float32)
        delay = pd.to_numeric(chunk["time_delay_for_conversion"], errors="coerce").fillna(-1).to_numpy(dtype=np.float32)
        observed = ((labels > 0) & (delay >= 0)).astype(np.float32)
        return SponsoredSearchBatch(sparse, dense, labels, delay, observed)

    def iter_batches(self, stats: DenseStats | None = None, split: str | None = None) -> Iterator[SponsoredSearchBatch]:
        if stats is None:
            stats = self.fit_dense_stats()
        for chunk in self.iter_raw():
            if split is not None:
                split_id = self._split_ids(chunk)
                chunk = chunk.iloc[split_id == _split_id(split)]
                if not len(chunk):
                    continue
            yield self.transform(chunk, stats)


def read_sponsored_search(
    path: str | Path = "data/criteo-sponsored-search/data.tsv",
    *,
    num_sparse_bins: int = 100_000,
    chunksize: int = 65_536,
    max_rows: int | None = None,
    stats_rows: int | None = 1_000_000,
    split_mode: str = "time",
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    feature_set: str = "full",
    cache_dir: str | Path | None = None,
) -> tuple[SponsoredSearchReader, DenseStats]:
    """Build a reader and fit stats using only the configured training split."""
    if cache_dir is not None:
        if split_mode != "time":
            raise ValueError("cache_dir requires split_mode='time'")
        from .cache import CachedSponsoredSearchReader, build_or_load_cache

        source_reader = SponsoredSearchReader(
            path,
            num_sparse_bins,
            chunksize,
            max_rows,
            sparse_columns=SPARSE_COLUMNS,
            dense_columns=DENSE_COLUMNS,
        )
        build_or_load_cache(source_reader, cache_dir, train_fraction, val_fraction)
        reader = CachedSponsoredSearchReader(cache_dir, feature_set=feature_set, chunksize=chunksize)
        return reader, reader.fit_dense_stats()
    sparse_columns, dense_columns = feature_columns(feature_set)
    reader = SponsoredSearchReader(
        path,
        num_sparse_bins,
        chunksize,
        max_rows,
        sparse_columns=sparse_columns,
        dense_columns=dense_columns,
    )
    if split_mode == "time":
        reader.fit_time_split(train_fraction=train_fraction, val_fraction=val_fraction)
        stats = reader.fit_dense_stats(stats_rows, split="train")
    elif split_mode == "row":
        stats = reader.fit_dense_stats(stats_rows)
    else:
        raise ValueError("split_mode must be 'time' or 'row'")
    return reader, stats


FEATURE_SETS = {
    "full": (SPARSE_COLUMNS, DENSE_COLUMNS),
    "no_user_id": (tuple(name for name in SPARSE_COLUMNS if name != "user_id"), DENSE_COLUMNS),
    "no_product_id": (tuple(name for name in SPARSE_COLUMNS if name != "product_id"), DENSE_COLUMNS),
    "no_entity_ids": (
        tuple(name for name in SPARSE_COLUMNS if name not in {"user_id", "product_id"}),
        DENSE_COLUMNS,
    ),
    "coarse_context": (
        tuple(
            name
            for name in SPARSE_COLUMNS
            if name
            not in {
                "audience_id",
                "product_brand",
                "product_id",
                "product_title",
                "partner_id",
                "user_id",
            }
        ),
        DENSE_COLUMNS,
    ),
    "no_history": (SPARSE_COLUMNS, ("click_timestamp", "product_price")),
    "no_price": (SPARSE_COLUMNS, ("click_timestamp", "nb_clicks_1week")),
    "no_time": (SPARSE_COLUMNS, ("nb_clicks_1week", "product_price")),
    "sparse_only": (SPARSE_COLUMNS, ()),
    "numeric_only": ((), DENSE_COLUMNS),
}


def feature_columns(feature_set: str = "full") -> tuple[tuple[str, ...], tuple[str, ...]]:
    try:
        sparse, dense = FEATURE_SETS[feature_set.lower().strip()]
    except KeyError as exc:
        raise ValueError(f"unknown feature_set {feature_set!r}; choose from {', '.join(FEATURE_SETS)}") from exc
    return tuple(sparse), tuple(dense)


def _split_id(name: str) -> int:
    values = {"train": 0, "valid": 1, "test": 2}
    try:
        return values[name.lower()]
    except KeyError as exc:
        raise ValueError("split must be train, valid, or test") from exc


__all__ = [
    "COLUMNS",
    "SPARSE_COLUMNS",
    "DENSE_COLUMNS",
    "DenseStats",
    "TimeSplit",
    "SponsoredSearchBatch",
    "SponsoredSearchReader",
    "read_sponsored_search",
    "FEATURE_SETS",
    "feature_columns",
]
