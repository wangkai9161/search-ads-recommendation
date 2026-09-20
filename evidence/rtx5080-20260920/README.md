# RTX 5080 Reproducibility Snapshot

This directory contains the reviewed, lightweight evidence from the 2026-09-20 remote runs. Raw datasets, mmap caches, logs, and checkpoints remain on the experiment server.

## Reviewed conclusions

- Criteo: `product_price` has a near-deterministic relationship with `Sale` in the released file and is excluded from primary results. Clean DeepFM reaches test AUC `0.777870 +/- 0.001254` across three seeds.
- MovieLens: leakage-free validation selection and prefix training make GRU4Rec best at Recall@20 `0.090979 +/- 0.002772`.
- LastFM: BPR with one negative is best at Recall@20 `0.081076 +/- 0.003935`; Tail Recall@20 remains zero.

## Files

- `EXPERIMENT_REPORT_CN.md`: detailed Chinese audit report and interpretation.
- `cvr-robust-report.md`, `cvr-robust-summary.json`: leakage-audited Criteo results.
- `movielens-10epoch-report.md`, `movielens-10epoch-summary.json`: MovieLens multi-seed results.
- `lastfm/report.md`, `lastfm/summary.json`: LastFM multi-seed ablation.
- `environment.json`: runtime and dataset checksums.

JSON files are the machine-readable source for the Markdown tables. These offline results do not establish online lift or production performance.
