# Data Preparation

The current project uses the Criteo Sponsored Search Conversion Log Dataset:

```text
data/criteo-sponsored-search/data.tsv
```

[`reader.py`](reader.py) is the single preparation entry point. It reads the
23-column, tab-separated file in bounded-memory chunks, estimates chronological
train/validation/test boundaries from `click_timestamp`, hashes anonymized
categorical fields into a fixed vocabulary, and z-score normalizes the
pre-click numeric fields using training rows only. Conversion delay remains a
target rather than an input to avoid post-conversion leakage.

```python
from prepare.reader import SponsoredSearchReader

reader = SponsoredSearchReader(max_rows=200_000)
reader.fit_time_split(train_fraction=0.7, val_fraction=0.15)
stats = reader.fit_dense_stats(max_rows=100_000, split="train")
for batch in reader.iter_batches(stats, split="train"):
    # batch.sparse: [batch, 16], batch.dense: [batch, 3]
    # batch.labels: Sale, batch.delay_seconds: conversion delay
    pass
```

The project entry point uses this `time` split by default. `row` remains only
as a legacy compatibility mode. Feature sets available for ablation are
`full`, `no_history`, `no_price`, `no_time`, `sparse_only`, and `numeric_only`.

重复进行全量 trial 时，可将 `cache_dir` 传给 `read_sponsored_search()`。首次运行
会创建 memory-mapped 数组；后续 feature set 共享同一时间切分和训练集统计量。
