# CVR Hyper-parameter Search

Target: `Sale` after click. Objective: minimize `val_logloss`, then maximize `val_pr_auc` and `val_auc`.

| Trial | Model | Features | Embedding | Learning rate | Seed | Status | Val LogLoss | Val PR-AUC | Val AUC | Test LogLoss | Test PR-AUC | Test AUC |
| ---: | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | deepfm | full | 16 | 0.001 | 42 | ok | 0.321006 | 0.479342 | 0.804541 | 0.295269 | 0.464898 | 0.817913 |
| 2 | deepfm | no_history | 16 | 0.001 | 42 | ok | 0.286723 | 0.824294 | 0.925433 | 0.260368 | 0.834296 | 0.935901 |
| 3 | deepfm | no_price | 16 | 0.001 | 42 | ok | 0.330694 | 0.118274 | 0.559577 | 0.302257 | 0.107635 | 0.579753 |
| 4 | deepfm | no_time | 16 | 0.001 | 42 | ok | 0.294268 | 0.843396 | 0.936792 | 0.274691 | 0.851280 | 0.943384 |
| 5 | deepfm | sparse_only | 16 | 0.001 | 42 | ok | 0.382831 | 0.108198 | 0.517063 | 0.366700 | 0.097221 | 0.529688 |
| 6 | deepfm | numeric_only | 16 | 0.001 | 42 | ok | 0.290356 | 0.875135 | 0.921716 | 0.260141 | 0.857789 | 0.927726 |

## Selected Trial

- Run: `sponsored-search-tune-deepfm-e16-lr0p001-s42-fno_history`
- Model: `deepfm`
- Validation LogLoss: `0.286723`
- Validation PR-AUC: `0.824294`
- Validation ROC-AUC: `0.925433`
