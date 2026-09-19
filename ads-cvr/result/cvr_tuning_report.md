# CVR Hyper-parameter Search

Target: `Sale` after click. Objective: minimize `val_logloss`, then maximize `val_pr_auc` and `val_auc`.

| Trial | Model | Features | Embedding | Learning rate | Seed | Status | Val LogLoss | Val PR-AUC | Val AUC | Test LogLoss | Test PR-AUC | Test AUC |
| ---: | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | deepfm | full | 16 | 0.001 | 42 | ok | 0.038595 | 0.980417 | 0.995914 | 0.036665 | 0.980275 | 0.996125 |
| 2 | deepfm | no_time | 16 | 0.001 | 42 | ok | 0.037136 | 0.980273 | 0.995928 | 0.035076 | 0.980348 | 0.996189 |
| 3 | deepfm | numeric_only | 16 | 0.001 | 42 | ok | 0.063368 | 0.953542 | 0.979137 | 0.061701 | 0.952694 | 0.979530 |

## Selected Trial

- Run: `sponsored-search-tune-deepfm-e16-lr0p001-s42-fno_time`
- Model: `deepfm`
- Validation LogLoss: `0.037136`
- Validation PR-AUC: `0.980273`
- Validation ROC-AUC: `0.995928`
