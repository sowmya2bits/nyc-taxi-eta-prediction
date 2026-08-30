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

## Week 3 — FastAPI

Run locally:

```powershell
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs`.

Endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Service status |
| GET | `/health` | Health/model status |
| GET | `/model-info` | Model information |
| POST | `/predict` | Predict trip duration |

Example request:

```json
{
  "pickup_datetime": "2016-06-01T08:30:00",
  "pickup_longitude": -73.985,
  "pickup_latitude": 40.758,
  "dropoff_longitude": -73.985,
  "dropoff_latitude": 40.748,
  "passenger_count": 2,
  "vendor_id": 1,
  "store_and_fwd_flag": "N"
}
```

**Important:** `pickup_datetime` must be a valid ISO-style timestamp. Do not send `"string"`.

## Docker Deployment

```powershell
docker build -t nyc-taxi-eta:latest -f src/Dockerfile .
docker rm -f nyc-taxi-eta-container
docker run -d --name nyc-taxi-eta-container -p 8000:8000 nyc-taxi-eta:latest
docker ps
docker logs nyc-taxi-eta-container
```

Then open `http://localhost:8000/docs`.

## Week 4 — Monitoring and Drift

Monitor:

1. Request/prediction volume and failures
2. Prediction latency
3. Important input distributions
4. Predicted-duration distribution
5. MAE/RMSE/RMSLE when actual labels become available
6. Simulated input drift
7. Retraining trigger

A practical trigger is to start a new validation/training cycle when meaningful drift persists above the configured threshold or labelled production error exceeds the accepted baseline. Validate a candidate model before promotion.

## Testing

```powershell
pytest -q
```

Final demo should show `/`, `/health`, `/model-info`, one valid `/predict` call and one invalid request.

## Reproducibility

- Git — source control
- DVC — dataset versioning
- MLflow — experiment tracking
- `params.yaml` — experiment configuration
- scikit-learn Pipeline — consistent preprocessing and estimator
- Docker — repeatable service packaging

## Final 5–7 Minute Demo

1. Repository and commit history
2. Architecture
3. Data validation/features
4. MLflow experiments
5. Model comparison
6. Docker build/run
7. FastAPI Swagger UI
8. Valid prediction
9. Invalid input validation
10. Monitoring/drift
11. Retraining strategy
12. Reproducibility and conclusion

## Submission Artifacts

- Repository link
- DVC evidence
- MLflow evidence
- `reports/model_comparison.csv`
- `reports/model_comparison_report.md`
- final model metadata
- Dockerfile
- API screenshots/test evidence
- monitoring/drift evidence
- project report
- 5–7 minute demo recording
- Google Drive/submission-folder link, if required

## References

- NYC Taxi Trip Duration: https://www.kaggle.com/c/nyc-taxi-trip-duration
- FastAPI
- MLflow
- DVC
- Scikit-learn
- XGBoost
- Crowe et al., *Machine Learning Production Systems*, O'Reilly, 2024
- Burkov, *Machine Learning Engineering*, 2020
- McMahon, *Machine Learning Engineering with Python*, 2nd Edition, Packt, 2023