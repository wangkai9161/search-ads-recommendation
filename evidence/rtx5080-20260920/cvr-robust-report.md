# Criteo CVR Leakage-Audited Report

Rows: 15,995,634. Target: post-click `Sale`. Split: chronological 70/15/15. Post-conversion revenue and delay fields are excluded. `product_price` is also excluded from primary results after the audit below found a near-deterministic label artifact.

## Clean four-model baseline (seed 42)

| Model | Test LogLoss | Test PR-AUC | Test AUC | Test ECE |
| --- | ---: | ---: | ---: | ---: |
| lr | 0.279342 | 0.268036 | 0.765749 | 0.005451 |
| fm | 0.278196 | 0.276872 | 0.770010 | 0.008619 |
| wide_deep | 0.315669 | 0.243251 | 0.740691 | 0.044198 |
| deepfm | 0.274910 | 0.285382 | 0.779100 | 0.011312 |

## Clean DeepFM feature robustness (three seeds)

| Features | Runs | Test LogLoss | Test PR-AUC | Test AUC |
| --- | ---: | ---: | ---: | ---: |
| clean | 3 | 0.275297 +/- 0.000761 | 0.285394 +/- 0.000128 | 0.777870 +/- 0.001254 |
| clean_no_user_id | 3 | 0.274713 +/- 0.000570 | 0.288641 +/- 0.001203 | 0.777521 +/- 0.000649 |
| clean_no_product_id | 3 | 0.275063 +/- 0.000780 | 0.285660 +/- 0.001777 | 0.776302 +/- 0.001271 |
| clean_no_entity_ids | 3 | 0.274787 +/- 0.000716 | 0.290042 +/- 0.002098 | 0.776285 +/- 0.001836 |
| clean_coarse_context | 3 | 0.295249 +/- 0.000103 | 0.197438 +/- 0.000204 | 0.712745 +/- 0.000447 |

## Excluded product-price artifact

On the chronological test split, 90.85% of rows have `product_price=0`. P(Sale | price=0) is 0.7488%, while P(Sale | price!=0) is 100.0000%. The non-sale rows with a non-zero price account for 0.0000% of non-sales. The field is retained only for diagnosis, not for the primary comparison.

| Diagnostic feature set | Runs | Test LogLoss | Test PR-AUC | Test AUC |
| --- | ---: | ---: | ---: | ---: |
| time_only | 3 | 0.324569 +/- 0.001943 | 0.099291 +/- 0.000000 | 0.503971 +/- 0.000000 |
| click_history_only | 3 | 0.321470 +/- 0.000233 | 0.106271 +/- 0.000010 | 0.518173 +/- 0.000028 |
| price_only | 3 | 0.042640 +/- 0.000581 | 0.948377 +/- 0.000073 | 0.965585 +/- 0.000259 |
