# LastFM Two-Tower Ablation

HetRec LastFM implicit feedback with a fixed seeded per-user holdout. Because the dataset has no timestamps, the split is not chronological.

Interactions: 92,826; users: 1,884; artists: 17,626; epochs per trial: 10.

| Trial | Best epoch | Recall@20 | NDCG@20 | Coverage@20 | Tail Recall@20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bce-neg0 | 9 | 0.018047 | 0.006980 | 0.121014 | 0.000000 |
| bce-neg1 | 1 | 0.065817 | 0.021526 | 0.018495 | 0.000000 |
| bce-neg3 | 2 | 0.074841 | 0.025292 | 0.047487 | 0.000000 |
| bce-neg5 | 1 | 0.069533 | 0.024193 | 0.025984 | 0.000000 |
| bpr-neg1 | 1 | 0.071125 | 0.029186 | 0.003120 | 0.000000 |
| bpr-neg3 | 1 | 0.068471 | 0.030231 | 0.002950 | 0.000000 |
| bpr-neg5 | 1 | 0.078556 | 0.030690 | 0.002326 | 0.000000 |
| bce-neg5-tail | 1 | 0.085987 | 0.031289 | 0.003574 | 0.000000 |
| bpr-neg5-tail | 1 | 0.071656 | 0.031437 | 0.003631 | 0.000000 |
