# CVR Hyper-parameter Search

Target: `Sale` after click. Objective: minimize `val_logloss`, then maximize `val_pr_auc` and `val_auc`.

| Trial | Model | Features | Embedding | Learning rate | Seed | Status | Val LogLoss | Val PR-AUC | Val AUC | Test LogLoss | Test PR-AUC | Test AUC |
| ---: | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | lr | full | 16 | 0.001 | 42 | ok | 0.521977 | 0.231436 | 0.623236 | 0.516700 | 0.226440 | 0.615841 |
| 2 | fm | full | 16 | 0.001 | 42 | ok | 0.525025 | 0.303900 | 0.682929 | 0.519055 | 0.317273 | 0.705369 |
| 3 | wide_deep | full | 16 | 0.001 | 42 | ok | 0.591293 | 0.129843 | 0.572271 | 0.540201 | 0.110116 | 0.573786 |
| 4 | deepfm | full | 16 | 0.001 | 42 | ok | 0.321006 | 0.479342 | 0.804541 | 0.295269 | 0.464898 | 0.817913 |

## Selected Trial

- Run: `sponsored-search-tune-deepfm-e16-lr0p001-s42`
- Model: `deepfm`
- Validation LogLoss: `0.321006`
- Validation PR-AUC: `0.479342`
- Validation ROC-AUC: `0.804541`
