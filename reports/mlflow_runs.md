
# MLflow Experiment Runs (exported)

Exported from the local MLflow tracking store — 11 run(s). Regenerate with `python src/export_mlflow_runs.py`.

| Run | Status | RMSLE | RMSE (s) | MAE (s) | R² | Git commit | Run ID |
|-----|--------|------:|---------:|--------:|---:|-----------|--------|
| xgboost | FINISHED | 0.3046 | 276.0878 | 169.4574 | 0.8178 | `ce53f11b8c66` | `b1fe8c216c9e` |
| xgboost | FINISHED | 0.3046 | 276.0878 | 169.4574 | 0.8178 | `ce53f11b8c66` | `fd8adca100e7` |
| hist_gradient_boosting | FINISHED | 0.3117 | 284.4045 | 174.9729 | 0.8067 | `ce53f11b8c66` | `387c869d2934` |
| hist_gradient_boosting | FINISHED | 0.3117 | 284.4045 | 174.9729 | 0.8067 | `ce53f11b8c66` | `f4fcea98fe28` |
| random_forest | FINISHED | 0.3238 | 291.5225 | 180.2733 | 0.7969 | `ce53f11b8c66` | `016700cee9c5` |
| random_forest | FINISHED | 0.3238 | 291.5225 | 180.2733 | 0.7969 | `ce53f11b8c66` | `2cedd4259f04` |
| gradient_boosting | FINISHED | 0.3330 | 307.3167 | 190.4153 | 0.7743 | `ce53f11b8c66` | `70a626d9f54c` |
| gradient_boosting | FINISHED | 0.3330 | 307.3167 | 190.4153 | 0.7743 | `ce53f11b8c66` | `ca24b0874de9` |
| linear_regression | FAILED | 0.4984 | 557.1845 | 312.6489 | 0.2581 | `ce53f11b8c66` | `59e2e2910306` |
| linear_regression | FINISHED | 0.4984 | 557.1845 | 312.6489 | 0.2581 | `ce53f11b8c66` | `661c3c7384ce` |
| linear_regression | FINISHED | 0.4984 | 557.1845 | 312.6489 | 0.2581 | `ce53f11b8c66` | `ccf0d981b797` |

Every run is tagged with the git commit it was produced from, so any run reproduces from `params.yaml` at that commit. This file is committed as durable experiment-tracking evidence; the raw `mlruns/` store stays git-ignored.
