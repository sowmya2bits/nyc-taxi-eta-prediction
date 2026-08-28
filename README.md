# NYC Taxi ETA Prediction

End-to-end ML engineering mini-project (Flavor A — Ride/Delivery ETA prediction)
built on the [NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration)
dataset. The pipeline ingests raw trip data, validates and cleans it, engineers
spatial/temporal features, versions the dataset with DVC, and trains and compares
models with MLflow experiment tracking.

## Architecture

```
                  ┌──────────────┐    ┌──────────────┐    ┌───────────────────────┐
  train.csv  ───► │  ingestion   │──► │  validation  │──► │  feature_engineering  │
  (Kaggle)        │  (unzip/read)│    │ (pandera +   │    │ (spatial + temporal + │
                  └──────────────┘    │  GPS/speed)  │    │  cyclical features)   │
                                      └──────────────┘    └───────────┬───────────┘
                                                                       ▼
                                            data/processed/train_processed.parquet
                                                       (versioned with DVC)
                                                                       │
                                                                       ▼
                          ┌────────────────────────────────────────────────────────┐
                          │  train.py  —  sklearn Pipeline (preprocess + estimator) │
                          │    • Linear Regression (baseline)                       │
                          │    • Random Forest / Gradient Boosting / HistGB         │
                          │    • XGBoost Regressor (usual winner)                   │
                          │  every run tracked in MLflow (params + metrics + model) │
                          └───────────────────────────┬────────────────────────────┘
                                                       ▼
                          models/final_model.pkl  +  reports/model_comparison_report.md
```

## Project Structure

```
nyc-taxi-eta-prediction/
├── data/
│   └── processed/train_processed.parquet   ← DVC-tracked (pointer in git)
├── src/
│   ├── ingestion.py            Week 1 — extract & load raw trips
│   ├── validation.py           Week 1 — pandera schema + GPS/speed filters
│   ├── feature_engineering.py  Week 1 — spatial/temporal/cyclical features
│   └── train.py                Week 2 — model training + MLflow tracking
├── reports/
│   ├── model_comparison_report.md   Week 2 — generated comparison report
│   └── model_comparison.csv         Week 2 — generated metrics table
├── models/                     Week 2 — final_model.pkl + metadata (git-ignored)
├── params.yaml                 Central experiment configuration
├── WEEK2.md                    Week 2 runbook
└── requirements.txt
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Fetch the versioned processed dataset (or regenerate it from raw):
dvc pull                        # if a DVC remote is configured
# --- or ---
python src/feature_engineering.py   # rebuilds data/processed/ from data/raw/train.csv
```

## Week 1 — Data Engineering (M2)

Ingestion, validation, and feature engineering produce a clean, feature-rich
dataset versioned with DVC.

```bash
python src/ingestion.py
python src/validation.py
python src/feature_engineering.py
```

- **Validation** (`pandera`): enforces schema, NYC GPS bounds, and drops
  physically impossible trips (implied speed 0.5–80 mph, distance 0.05–50 mi).
- **Features**: haversine/manhattan distance, bearing, JFK/LGA airport
  proximity, hour/day/month, weekend & rush-hour flags, cyclical hour encoding,
  and a `log1p` target for the RMSLE objective.

## Week 2 — Experimentation & Reproducibility (M3)

Train and compare two models; every run is tracked in MLflow with its
parameters, metrics, and the fitted pipeline.

```bash
python src/train.py            # runs both experiments + writes the comparison report
mlflow ui --backend-store-uri mlruns   # inspect runs at http://127.0.0.1:5000
```

**Experiments**

| Model | Role | Preprocessing |
|-------|------|---------------|
| Linear Regression | interpretable baseline | median impute + standardize + one-hot |
| Random Forest | bagged-tree nonlinear baseline | median impute + one-hot |
| Gradient Boosting | sequential shallow-tree boosting | median impute + one-hot |
| Hist Gradient Boosting | fast histogram-based boosting | median impute + one-hot |
| XGBoost Regressor | gradient boosting (usual winner) | median impute + one-hot |

Every model is wrapped in a scikit-learn `Pipeline` so the exact preprocessing
is bound to the estimator, eliminating training–serving skew. Models are
compared on validation **RMSE (seconds)**, **RMSLE** (the competition metric),
**MAE**, and **R²**. The best model (lowest RMSE) is serialized to
`models/final_model.pkl` with `models/model_metadata.json`, and the results are
written to `reports/model_comparison_report.md`.

All hyperparameters live in `params.yaml`, so any tracked run can be reproduced
from source. On a 150k-row sample, XGBoost led (RMSE ≈ 276s, R² ≈ 0.82) with the
tree ensembles close behind and the linear baseline well back (R² ≈ 0.26) —
rerun on the full dataset for final numbers.

## Roadmap

- **Week 3 (M4):** package the best model and serve ETA predictions via a REST API.
- **Week 4 (M5):** log predictions vs. actuals, simulate drift, and design a retraining trigger.
