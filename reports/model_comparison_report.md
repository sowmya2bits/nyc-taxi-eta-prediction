# Week 2 — Model Comparison Report (M3)

5 experiments were trained on the DVC-versioned processed dataset (118,538 train / 29,635 validation rows, 80/20 seeded split) and tracked with MLflow. Models are ranked by validation RMSE (seconds); RMSLE (log-space RMSE) is the NYC Taxi Trip Duration competition metric.

| Model | RMSLE | RMSE (s) | MAE (s) | R2 |
|-------|------:|---------:|--------:|---:|
| xgboost | 0.3046 | 276.1 | 169.5 | 0.8178 |
| hist_gradient_boosting | 0.3117 | 284.4 | 175.0 | 0.8067 |
| random_forest | 0.3238 | 291.5 | 180.3 | 0.7969 |
| gradient_boosting | 0.3330 | 307.3 | 190.4 | 0.7743 |
| linear_regression | 0.4984 | 557.2 | 312.6 | 0.2581 |

## Selected model: **xgboost**

`xgboost` is selected as the best model — it achieves the lowest validation RMSE (276.1s) and RMSLE (0.3046). Tree-based ensembles capture the non-linear interactions between distance, time-of-day and location features that the linear baseline cannot. The full pipeline (preprocessing + estimator) is serialized to `models/final_model.pkl` and every run is reproducible from `params.yaml` via the MLflow logs.
