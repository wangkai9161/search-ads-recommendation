"""Memory-mapped cache for repeated Criteo CVR experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .reader import (
    DENSE_COLUMNS,
    SPARSE_COLUMNS,
    DenseStats,
    SponsoredSearchBatch,
    SponsoredSearchReader,
    TimeSplit,
    feature_columns,
)


CACHE_VERSION = 1


def build_or_load_cache(
    source_reader: SponsoredSearchReader,
    cache_dir: str | Path,
    train_fraction: float,
    val_fraction: float,
) -> tuple[Path, dict[str, object]]:
    """Create a full-feature mmap cache once and return its metadata."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if _cache_matches(metadata, source_reader, train_fraction, val_fraction):
            return cache_dir, metadata

    if source_reader.sparse_columns != tuple(SPARSE_COLUMNS) or source_reader.dense_columns != tuple(DENSE_COLUMNS):
        raise ValueError("cache construction requires the full feature reader")
    if source_reader.time_split is None:
        source_reader.fit_time_split(train_fraction, val_fraction)
    stats = source_reader.fit_dense_stats(None, split="train")
    # Count physical lines without parsing the 23-column TSV a second time.
    with source_reader.path.open("rb") as handle:
        rows = sum(1 for _ in handle)
    if source_reader.max_rows is not None:
        rows = min(rows, source_reader.max_rows)
    if rows == 0:
        raise ValueError("cannot cache an empty sponsored-search file")

    sparse_path = cache_dir / "sparse.int32.npy"
    dense_path = cache_dir / "dense.float32.npy"
    labels_path = cache_dir / "labels.float32.npy"
    delay_path = cache_dir / "delay.float32.npy"
    split_path = cache_dir / "split.uint8.npy"
    sparse = np.lib.format.open_memmap(sparse_path, mode="w+", dtype=np.int32, shape=(rows, len(SPARSE_COLUMNS)))
    dense = np.lib.format.open_memmap(dense_path, mode="w+", dtype=np.float32, shape=(rows, len(DENSE_COLUMNS)))
    labels = np.lib.format.open_memmap(labels_path, mode="w+", dtype=np.float32, shape=(rows,))
    delay = np.lib.format.open_memmap(delay_path, mode="w+", dtype=np.float32, shape=(rows,))
    split = np.lib.format.open_memmap(split_path, mode="w+", dtype=np.uint8, shape=(rows,))

    offset = 0
    for chunk in source_reader.iter_raw():
        end = offset + len(chunk)
        batch = source_reader.transform(chunk, stats)
        sparse[offset:end] = batch.sparse.astype(np.int32, copy=False)
        dense[offset:end] = batch.dense
        labels[offset:end] = batch.labels
        delay[offset:end] = batch.delay_seconds
        split[offset:end] = source_reader._split_ids(chunk).astype(np.uint8, copy=False)
        offset = end
    for array in (sparse, dense, labels, delay, split):
        array.flush()

    metadata = {
        "cache_version": CACHE_VERSION,
        "source_path": str(source_reader.path.resolve()),
        "source_size": source_reader.path.stat().st_size,
        "max_rows": source_reader.max_rows,
        "rows": rows,
        "num_sparse_bins": source_reader.num_sparse_bins,
        "train_fraction": train_fraction,
        "val_fraction": val_fraction,
        "sparse_columns": list(SPARSE_COLUMNS),
        "dense_columns": list(DENSE_COLUMNS),
        "train_end": source_reader.time_split.train_end,
        "valid_end": source_reader.time_split.valid_end,
        "mean": stats.mean.tolist(),
        "std": stats.std.tolist(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return cache_dir, metadata


class CachedSponsoredSearchReader:
    """Reader-compatible view over a memory-mapped full-feature cache."""

    def __init__(self, cache_dir: str | Path, feature_set: str = "full", chunksize: int = 65_536):
        cache_dir = Path(cache_dir)
        metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
        sparse_names, dense_names = feature_columns(feature_set)
        sparse_indices = [metadata["sparse_columns"].index(name) for name in sparse_names]
        dense_indices = [metadata["dense_columns"].index(name) for name in dense_names]
        self.cache_dir = cache_dir
        self.metadata = metadata
        self.path = Path(metadata["source_path"])
        self.max_rows = metadata["max_rows"]
        self.num_sparse_bins = int(metadata["num_sparse_bins"])
        self.sparse_columns = tuple(sparse_names)
        self.dense_columns = tuple(dense_names)
        self._sparse_indices = sparse_indices
        self._dense_indices = dense_indices
        self.chunksize = int(chunksize)
        self.num_sparse_fields = len(sparse_names)
        self.num_dense = len(dense_names)
        self.time_split = TimeSplit(float(metadata["train_end"]), float(metadata["valid_end"]))
        self._sparse = np.load(cache_dir / "sparse.int32.npy", mmap_mode="r")
        self._dense = np.load(cache_dir / "dense.float32.npy", mmap_mode="r")
        self._labels = np.load(cache_dir / "labels.float32.npy", mmap_mode="r")
        self._delay = np.load(cache_dir / "delay.float32.npy", mmap_mode="r")
        self._split = np.load(cache_dir / "split.uint8.npy", mmap_mode="r")

    def iter_batches(self, stats: DenseStats | None = None, split: str | None = None):
        split_id = {None: None, "train": 0, "valid": 1, "test": 2}.get(split)
        if split is not None and split_id is None:
            raise ValueError("split must be train, valid, or test")
        if stats is None:
            stats = self.fit_dense_stats()
        for start in range(0, len(self._labels), self.chunksize):
            end = min(start + self.chunksize, len(self._labels))
            indices = np.arange(start, end)
            if split_id is not None:
                indices = indices[self._split[start:end] == split_id]
            if not len(indices):
                continue
            # Dense values are normalized once during cache construction using
            # train-only statistics; feature subsets only select columns.
            dense = np.asarray(self._dense[indices][:, self._dense_indices], dtype=np.float32)
            sparse = np.asarray(self._sparse[indices][:, self._sparse_indices], dtype=np.int64)
            labels = np.asarray(self._labels[indices], dtype=np.float32)
            delay = np.asarray(self._delay[indices], dtype=np.float32)
            observed = ((labels > 0) & (delay >= 0)).astype(np.float32)
            yield SponsoredSearchBatch(sparse, dense, labels, delay, observed)

    def fit_dense_stats(self, max_rows: int | None = None, split: str | None = "train") -> DenseStats:
        mean = np.asarray(self.metadata["mean"], dtype=np.float32)[self._dense_indices]
        std = np.asarray(self.metadata["std"], dtype=np.float32)[self._dense_indices]
        return DenseStats(mean, std)


def _cache_matches(metadata: dict[str, object], reader: SponsoredSearchReader, train_fraction: float, val_fraction: float) -> bool:
    return (
        metadata.get("cache_version") == CACHE_VERSION
        and metadata.get("source_path") == str(reader.path.resolve())
        and metadata.get("source_size") == reader.path.stat().st_size
        and metadata.get("max_rows") == reader.max_rows
        and metadata.get("num_sparse_bins") == reader.num_sparse_bins
        and metadata.get("train_fraction") == train_fraction
        and metadata.get("val_fraction") == val_fraction
    )


__all__ = ["build_or_load_cache", "CachedSponsoredSearchReader"]
