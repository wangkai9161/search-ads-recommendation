# RTX 5080 Reproducibility Snapshot

This directory contains the reviewed, lightweight evidence from the 2026-09-20
remote run. Raw public datasets, mmap caches, logs, and model checkpoints remain
on the experiment server.

## Environment

- Python 3.10.12
- PyTorch 2.11.0 with CUDA 12.8
- NVIDIA GeForce RTX 5080
- Dataset byte sizes and SHA-256 values are recorded in `environment.json`.

## Conclusions

- LastFM: BCE with 5 negatives and tail weighting had the best aggregate
  Recall@20 (`0.085987`), but rare-artist Tail Recall@20 remained zero.
- MovieLens: SASRec had the best 10-epoch full-catalog Recall@20 (`0.074197`);
  the controlled Two-Tower sweep did not exceed it.
- Criteo Sponsored Search: DeepFM was selected by validation LogLoss and reached
  test LogLoss/PR-AUC/AUC of `0.035910 / 0.977485 / 0.995453` on 15,995,634 rows.

The JSON files are the machine-readable source for the Markdown tables. These
offline results do not establish online lift or production performance.
