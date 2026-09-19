# Training

## Call Chain

The project-level `main.py` owns orchestration only:

1. Parse and validate CVR parameters (`task=cvr`, `target=Sale`).
2. Call `prepare.reader.read_sponsored_search()` for each feature set; time
   boundaries are estimated first and normalization statistics use train rows.
3. Build each trial and call `train.train_model()` with the prepared data.
4. Call `result.evaluate.evaluate_cvr()` and
   `result.storage.save_training_result()` for metrics and checkpoints.
5. Call `result.visualize.save_cvr_visualizations()`; the function writes
   generated plots into `output/`.

The training implementation does not parse the tuning grid. This keeps model
training reusable from notebooks while the root entry point controls the
experiment parameters.

`main.py` is the only training entry point for the Criteo Sponsored Search
file:

```text
data/criteo-sponsored-search/data.tsv
```

It streams the data, hashes 16 categorical fields, normalizes three pre-click
numeric fields, and trains one of the four tabular CVR models:

```text
lr, fm, wide_deep, deepfm
```

Example (one DeepFM configuration):

```bash
python main.py \
  --models deepfm \
  --embedding-dims 16 \
  --learning-rates 0.001 \
  --epochs 8 \
  --max-rows 200000
```

`--max-rows 0` means the full `data.tsv` (about 16 million click rows), while
`--stats-rows 0` fits normalization statistics on the full file. Use a smaller
limit first to validate a new model or configuration.

For a full run from the repository root:

```bash
python main.py \
  --models deepfm --embedding-dims 16 --learning-rates 0.001 \
  --seeds 42 --epochs 8 --max-rows 0 --stats-rows 0 \
  --batch-size 65536 --device auto
```

使用主函数进行多组参数调优并生成汇总：

```bash
python main.py \
  --models deepfm \
  --embedding-dims 8,16 \
  --learning-rates 0.0005,0.001 \
  --epochs 1 \
  --max-rows 200000
```

Training stops after three consecutive validation LogLoss rounds without a
meaningful improvement and restores the best validation checkpoint. Each
trial records train/validation/test LogLoss, ROC-AUC, PR-AUC, Brier, ECE and
the positive rate. Every invocation receives a unique UTC experiment ID by
default; use `--experiment-id` to provide a reproducible namespace. Trial
directories include that ID, so repeated runs do not overwrite checkpoints.

For the selected trial, add `--calibration` to fit Temperature Scaling on the
validation logits and write reliability, calibration-bin, decile-lift and
before/after metric files under `result/experiments/<experiment-id>/calibration/`.

`main.py` is the project-level hyper-parameter entry point; the implementation
is in `train/tuning.py`. It runs trials
sequentially, gives every trial a unique `result/sponsored-search-tune-*/`
directory, and selects the lowest validation CVR LogLoss (then PR-AUC and
ROC-AUC). Use `--dry-run` to inspect the expanded grid. The search summary is
written to the following files:

The default search space is declared at the top of the project-level
`main.py`:
`DEFAULT_MODELS=("deepfm",)`, `DEFAULT_EMBEDDING_DIMS=(8, 16)`,
`DEFAULT_LEARNING_RATES=(0.0005, 0.001)`, and `DEFAULT_SEEDS=(42,)`.
Command-line flags override these defaults.

```text
result/cvr_tuning_summary.json
result/cvr_tuning_summary.csv
result/cvr_tuning_report.md
output/cvr_tuning_comparison.png
```

For a full-data search, replace `--max-rows 200000` with `--max-rows 0` after
checking a small run first.

The function API is also available for notebooks and later runners:

```python
from train.train_model import TrainConfig, train_model

model, history = train_model(TrainConfig(model="fm", max_rows=100_000))
```

The target is `Sale`, so this is post-click conversion prediction,
`P(Sale=1 | click context)`. Each run writes:

```text
result/sponsored-search-<model>/metrics.json
result/sponsored-search-<model>/train_history.csv
result/sponsored-search-<model>/config.json
result/sponsored-search-<model>/checkpoints/model_final.pth
```

The project-level `main.py` additionally calls `result.visualize` and writes
plots to `output/sponsored-search-<model>/`. Direct calls to `train_model()`
only persist numeric results and checkpoints through `result.storage`.

Validation metrics include CVR LogLoss, ROC-AUC, PR-AUC, Brier score, ECE, and
the positive conversion rate. DIN, DIEN, DSSM, ESMM, and Transformer have model
implementations in `model/`, but need task-specific sequence or
impression-level batches. The current Sponsored Search file contains click
rows rather than full impression logs, so ESMM should not be trained with
fabricated CTR labels; CTR is only an auxiliary task when impression-level
data is available.
