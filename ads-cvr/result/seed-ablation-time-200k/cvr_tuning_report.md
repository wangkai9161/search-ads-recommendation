# CVR Hyper-parameter Search

Target: `Sale` after click. Objective: minimize `val_logloss`, then maximize `val_pr_auc` and `val_auc`.

| Trial | Model | Features | Embedding | Learning rate | Seed | Status | Val LogLoss | Val PR-AUC | Val AUC | Test LogLoss | Test PR-AUC | Test AUC |
| ---: | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | deepfm | full | 16 | 0.001 | 42 | ok | 0.321006 | 0.479342 | 0.804541 | 0.295269 | 0.464898 | 0.817913 |
| 2 | deepfm | no_history | 16 | 0.001 | 42 | ok | 0.286723 | 0.824294 | 0.925433 | 0.260368 | 0.834296 | 0.935901 |
| 3 | deepfm | no_time | 16 | 0.001 | 42 | ok | 0.294268 | 0.843396 | 0.936792 | 0.274691 | 0.851280 | 0.943384 |
| 4 | deepfm | numeric_only | 16 | 0.001 | 42 | ok | 0.290356 | 0.875135 | 0.921716 | 0.260141 | 0.857789 | 0.927726 |
| 5 | deepfm | full | 16 | 0.001 | 123 | ok | 0.284363 | 0.715908 | 0.886990 | 0.258353 | 0.744361 | 0.905768 |
| 6 | deepfm | no_history | 16 | 0.001 | 123 | ok | 0.362533 | 0.616334 | 0.833557 | 0.342195 | 0.644719 | 0.861857 |
| 7 | deepfm | no_time | 16 | 0.001 | 123 | ok | 0.359455 | 0.615763 | 0.835825 | 0.344850 | 0.629045 | 0.855885 |
| 8 | deepfm | numeric_only | 16 | 0.001 | 123 | ok | 0.276302 | 0.882300 | 0.933733 | 0.249189 | 0.875295 | 0.941357 |
| 9 | deepfm | full | 16 | 0.001 | 2024 | ok | 0.405337 | 0.578563 | 0.844448 | 0.398266 | 0.558744 | 0.855839 |
| 10 | deepfm | no_history | 16 | 0.001 | 2024 | ok | 0.411654 | 0.654586 | 0.852101 | 0.407031 | 0.670032 | 0.870501 |
| 11 | deepfm | no_time | 16 | 0.001 | 2024 | ok | 0.321068 | 0.624195 | 0.841369 | 0.304745 | 0.636906 | 0.858524 |
| 12 | deepfm | numeric_only | 16 | 0.001 | 2024 | ok | 0.277593 | 0.808885 | 0.909525 | 0.248680 | 0.795174 | 0.916214 |

## Selected Trial

- Run: `sponsored-search-tune-deepfm-e16-lr0p001-s123-fnumeric_only`
- Model: `deepfm`
- Validation LogLoss: `0.276302`
- Validation PR-AUC: `0.882300`
- Validation ROC-AUC: `0.933733`
