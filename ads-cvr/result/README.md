# Result

Training metrics, evaluation tables, and visualization data are stored here.

Each training run gets its own `result/<run>/` directory. Keep numerical
results and data exports here; rendered charts and other visual model outputs
belong in `output/<run>/`.

Examples:

- JSON metrics
- CSV histories
- exported charts data
- model checkpoints under `checkpoints/`

The current CVR pipeline uses:

- `evaluate.py` for `Sale` validation metrics (`LogLoss`, ROC-AUC, PR-AUC,
  Brier, and ECE);
- `storage.py` for checkpoints, histories, configs, and metrics JSON.
- `visualize.py` for plot generation; rendered files are written to `output/`.

Time-split experiment snapshots are kept separately so later smoke runs do not
overwrite the conclusions:

- `baseline-time-200k/`: four-model baseline on one seed;
- `ablation-time-200k/`: six DeepFM feature sets;
- `seed-ablation-time-200k/`: four feature sets repeated with three seeds.
- `full-time-2epoch/`: full 15.99M-row chronological comparison of `full`,
  `no_time`, and `numeric_only`.

Each new `main.py` invocation also writes a unique namespace under
`result/experiments/<experiment-id>/`, containing the tuning summary and
comparison plot. Calibration runs add reliability and lift artifacts there.

`storage.py` is the single persistence implementation. Training code no
longer owns a second artifact writer.
