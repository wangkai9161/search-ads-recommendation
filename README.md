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
| LastFM two-tower ablation | 92,826 interactions; best Recall@20 `0.085987` with BCE, 5 negatives, and tail weighting; rare-artist Tail Recall@20 remained `0` |
| MovieLens unified retrieval | 10 epochs per learned model; SASRec was best at Recall@20 `0.074197`, ahead of Two-Tower `0.070541` |
| Criteo full-data CVR | 15,995,634 rows; DeepFM test LogLoss/PR-AUC/AUC `0.035910 / 0.977485 / 0.995453` |

These are offline public-dataset results, not production or online A/B claims.

## Quick checks

Each subproject has an independent environment and README. A repository-wide syntax check can be run with:

```bash
python -m compileall ads-cvr retrieval-generative sequential-ranking
```
