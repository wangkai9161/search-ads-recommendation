# Criteo Full-Data Four-Model Comparison

Rows: 15,995,634. Target: post-click `Sale`. Split: chronological 70/15/15. Training budget: at most 10 epochs with validation LogLoss early stopping.

| Model | Best/last epoch | Test LogLoss | Test PR-AUC | Test AUC | Test ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| lr | 10/10 | 0.196421 | 0.689681 | 0.920590 | 0.037857 |
| fm | 10/10 | 0.206346 | 0.584846 | 0.891373 | 0.007406 |
| wide_deep | 8/10 | 0.039846 | 0.973750 | 0.994344 | 0.004650 |
| deepfm | 3/6 | 0.035910 | 0.977485 | 0.995453 | 0.001592 |

Selected by validation objective: **deepfm**.
