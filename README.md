# Search, Ads, and Recommendation Lab

Reproducible offline experiments spanning recommendation retrieval, sequential ranking, and post-click conversion prediction.

## Projects

| Directory | Task | Models and evaluation |
| --- | --- | --- |
| [`ads-cvr`](ads-cvr/) | Post-click conversion prediction on Criteo Sponsored Search | LR, FM, Wide & Deep, DeepFM; LogLoss, ROC-AUC, PR-AUC, Brier score, ECE |
| [`retrieval-generative`](retrieval-generative/) | Retrieval and generative recommendation | DSSM, negative sampling, multi-interest routing, discrete item codes, decoder-only recommendation; Recall/NDCG/coverage |
| [`sequential-ranking`](sequential-ranking/) | Retrieval, sequence modeling, and ranking on MovieLens-1M | Two-Tower, GRU4Rec, SASRec, DIN, DeepFM, popularity baseline |

## Design principles

- Keep CTR, CVR, retrieval, and ranking objectives distinct.
- Use chronological or leave-one-out splits where the task requires them.
- Report workload, training budget, random seeds, and evaluation scope with every result.
- Treat public datasets and offline metrics as learning evidence, not as claims of production performance.

Large datasets, checkpoints, caches, and repeated trial artifacts are intentionally excluded. Small metrics, configurations, and figures remain next to the code that produced them.

## Verified RTX 5080 evidence

The reproducibility snapshot under [`evidence/rtx5080-20260920`](evidence/rtx5080-20260920/)
records Python/PyTorch/CUDA versions, dataset hashes, fixed protocols, and
machine-readable results. The source datasets and checkpoints remain on the
experiment server and are not redistributed.

| Task | Verified result |
| --- | --- |
| LastFM two-tower ablation | 92,824 interactions; BPR with one negative reached Recall@20 `0.081076 +/- 0.003935`; Tail Recall@20 remained `0` |
| MovieLens unified retrieval | Leakage-free validation selection over three seeds; GRU4Rec Recall@20 `0.090979 +/- 0.002772` |
| Criteo full-data CVR | 15,995,634 rows; after excluding the audited `product_price` artifact, DeepFM test LogLoss/PR-AUC/AUC `0.274910 / 0.285382 / 0.779100` |

These are offline public-dataset results, not production or online A/B claims.
The detailed protocol audit and cause analysis are in
[`evidence/rtx5080-20260920/EXPERIMENT_REPORT_CN.md`](evidence/rtx5080-20260920/EXPERIMENT_REPORT_CN.md).

## Quick checks

Each subproject has an independent environment and README. A repository-wide syntax check can be run with:

```bash
python -m compileall ads-cvr retrieval-generative sequential-ranking
```
