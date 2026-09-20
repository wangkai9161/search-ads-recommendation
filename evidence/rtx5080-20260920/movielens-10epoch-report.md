# MovieLens-1M Leakage-Free Multi-Seed Comparison

The penultimate positive selects the epoch and the final positive is evaluated once. Reported values are population mean and standard deviation across seeds.

| Model | Runs | Test Recall@20 | Test NDCG@20 | Val Recall@20 |
| --- | ---: | ---: | ---: | ---: |
| twotower | 3 | 0.064591 +/- 0.002042 | 0.023828 +/- 0.000993 | 0.078503 +/- 0.001412 |
| poprec | 3 | 0.067241 +/- 0.000000 | 0.024795 +/- 0.000000 | 0.078006 +/- 0.000000 |
| gru4rec | 3 | 0.090979 +/- 0.002772 | 0.033728 +/- 0.001643 | 0.096610 +/- 0.001895 |
| sasrec | 3 | 0.064646 +/- 0.006707 | 0.024493 +/- 0.002478 | 0.068566 +/- 0.003524 |
