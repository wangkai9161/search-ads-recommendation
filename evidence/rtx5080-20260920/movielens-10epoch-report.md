# MovieLens-1M Unified Retrieval Comparison

All learned models use 10 training epochs. Evaluation holds out each user's final positive interaction, ranks the full catalog, and filters prior interactions.

| Model | Training epochs | Best epoch | Recall@20 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: |
| twotower | 10 | 9 | 0.070541 | 0.024981 |
| poprec | 0 | 0 | 0.067230 | 0.024792 |
| gru4rec | 10 | 8 | 0.072375 | 0.029453 |
| sasrec | 10 | 5 | 0.074197 | 0.031387 |
