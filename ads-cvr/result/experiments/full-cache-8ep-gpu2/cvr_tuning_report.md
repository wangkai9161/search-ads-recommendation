# CVR Hyper-parameter Search

Target: `Sale` after click. Objective: minimize `val_logloss`, then maximize `val_pr_auc` and `val_auc`.

| Trial | Model | Features | Embedding | Learning rate | Seed | Status | Val LogLoss | Val PR-AUC | Val AUC | Test LogLoss | Test PR-AUC | Test AUC |
| ---: | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | deepfm | full | 16 | 0.001 | 42 | ok | 0.037845 | 0.977277 | 0.995116 | 0.035917 | 0.977513 | 0.995463 |
| 2 | deepfm | no_time | 16 | 0.001 | 42 | ok | 0.037035 | 0.980340 | 0.995942 | 0.034990 | 0.980408 | 0.996200 |
| 3 | deepfm | numeric_only | 16 | 0.001 | 42 | ok | 0.047084 | 0.954779 | 0.980430 | 0.047038 | 0.953623 | 0.980400 |

## Selected Trial

- Run: `sponsored-search-tune-deepfm-e16-lr0p001-s42-fno_time-rfull-cache-8ep-gpu2`
- Model: `deepfm`
- Validation LogLoss: `0.037035`
- Validation PR-AUC: `0.980340`
- Validation ROC-AUC: `0.995942`
