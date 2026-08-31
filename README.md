# NYC Taxi ETA Prediction

An end-to-end **Machine Learning Engineering** mini-project (Flavor A — Ride/Delivery
ETA prediction) built on the
[NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration) dataset.

The project takes **1.46 million raw trips** through the full ML system lifecycle:
ingest → validate & clean → engineer features → version the dataset (DVC) →
train & compare five models (MLflow) → package the winner → serve it over a REST
API (FastAPI + Docker) → monitor it for data drift with a documented retraining
trigger.

| Week | Module | Focus | Key deliverable |
|------|--------|-------|-----------------|
| 1 | M2 | Data engineering & versioning | Clean, feature-rich dataset (DVC-tracked) |
| 2 | M3 | Experimentation & reproducibility | 5 tracked models + comparison report |
| 3 | M4 | Packaging & deployment | FastAPI service + Docker image |
| 4 | M5 | Monitoring, drift & retraining | PSI drift report + retraining strategy |
| 5 | — | Deployment & submission packaging | Containerized service, demo, docs |

📁 **[Demo & Report — Google Drive](https://drive.google.com/drive/folders/1sxA208pYxySr8vjQzEFEsYhWzbopABdq)**

---

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup](#setup)
- [Week 1 — Data Engineering (M2)](#week-1--data-engineering-m2)
- [Week 2 — Experimentation & Reproducibility (M3)](#week-2--experimentation--reproducibility-m3)
- [Week 3 — Packaging & Deployment (M4)](#week-3--packaging--deployment-m4)
- [Week 4 — Monitoring, Drift & Retraining (M5)](#week-4--monitoring-drift--retraining-m5)
- [Week 5 — Deployment & Submission Packaging](#week-5--deployment--submission-packaging)
- [Testing](#testing)
- [Design Decisions](#design-decisions)
- [Reproducibility](#reproducibility)
- [Submission Checklist Mapping](#submission-checklist-mapping)
- [References](#references)

---

## Architecture

The full system — from raw CSV to a monitored, deployed service with a drift
feedback loop back into retraining:

```
                            WEEK 1 · Data Engineering (M2)
  ┌────────────┐   ┌────────────┐   ┌──────────────┐   ┌───────────────────────┐
  │ train.csv  │──►│ ingestion  │──►│  validation  │──►│  feature_engineering  │
  │ (Kaggle)   │   │ (unzip/read)│  │ (pandera +   │   │ (spatial + temporal + │
  └────────────┘   └────────────┘   │  GPS/speed)  │   │  cyclical features)   │
                                     └──────────────┘   └───────────┬───────────┘
                                                                     ▼
                                       data/processed/train_processed.parquet
                                                 (versioned with DVC · tag v1.0-data)
                                                                     │
              WEEK 2 · Experimentation (M3)                          ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  train.py — sklearn Pipeline (preprocess + estimator), 5 models            │
  │    Linear Regression · Random Forest · Gradient Boosting · HistGB · XGBoost│
  │    every run tracked in MLflow (params + metrics + fitted pipeline)        │
  └───────────────────────────────────┬────────────────────────────────────────┘
                                       ▼
             models/final_model.pkl  +  reports/model_comparison_report.md
                                       │
              WEEK 3 · Deployment (M4) ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  api.py — FastAPI service (Docker-packaged)                                │
  │    POST /predict · GET /health · GET /model-info · GET /                   │
  │    Pydantic validation · reuses engineer_features() → zero train/serve skew│
  └───────────────────────────────────┬────────────────────────────────────────┘
                                       ▼  predictions
              WEEK 4 · Monitoring (M5) │
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  monitoring.py — data quality · prediction quality · PSI drift detection   │
  │    reports/monitoring_report.md                                            │
  │    retraining trigger: PSI > 0.25 ──────────────────────┐                  │
  └──────────────────────────────────────────────────────────┼─────────────────┘
                                                              │ retrain
                                       └──────────────────────┘
                                       (loop back to train.py)
```

---

## Project Structure

```
nyc-taxi-eta-prediction/
├── data/
│   └── processed/train_processed.parquet   ← DVC-tracked (pointer in git)
├── src/
│   ├── ingestion.py            Week 1 — extract & load raw trips
│   ├── validation.py           Week 1 — pandera schema + GPS/speed filters
│   ├── feature_engineering.py  Week 1 — spatial/temporal/cyclical features
│   ├── train.py                Week 2 — 5-model training + MLflow tracking
│   ├── export_mlflow_runs.py   Week 2 — export MLflow runs → committable CSV/MD
│   ├── api.py                  Week 3 — FastAPI prediction service
│   ├── monitoring.py           Week 4 — drift detection + monitoring report
│   └── Dockerfile              Week 3/5 — container image for the service
├── reports/
│   ├── model_comparison_report.md   Week 2 — model ranking + winner rationale
│   ├── model_comparison.csv         Week 2 — metrics table
│   ├── mlflow_runs.md               Week 2 — exported MLflow run table
│   ├── mlflow_runs.csv              Week 2 — exported MLflow run metrics
│   └── monitoring_report.md         Week 4 — data-quality + PSI drift report
├── models/                     final_model.pkl + model_metadata.json (committed)
├── tests/
│   ├── test_api.py             API endpoint + validation tests
│   └── test_monitoring.py      monitoring/PSI unit tests
├── params.yaml                 Central experiment & feature configuration
└── requirements.txt
```

---

## Tech Stack

All open-source: **Python 3.12**, **pandas / NumPy**, **pandera** (schema
validation), **scikit-learn** + **XGBoost** (modeling), **MLflow** (experiment
tracking), **DVC** (dataset versioning), **FastAPI + Uvicorn + Pydantic**
(serving), **Docker** (packaging), **pytest** (testing).

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Obtain the processed dataset — either pull from DVC (if a remote is configured)
dvc pull
# --- or regenerate it end-to-end from the raw Kaggle CSV (recommended / reliable) ---
python src/feature_engineering.py   # rebuilds data/processed/ from data/raw/train.csv
```

> **Prerequisite:** place the Kaggle NYC Taxi Trip Duration `train.csv` under
> `data/raw/`.

---

## Week 1 — Data Engineering (M2)

Ingestion, validation, and feature engineering turn raw, messy trip records into
a clean, feature-rich dataset — then version it with DVC.

```bash
python src/ingestion.py            # extract & load raw trips
python src/validation.py           # schema + physics-based cleaning
python src/feature_engineering.py  # chains ingest → validate → features
```

- **Validation** (`src/validation.py`) — a `pandera` schema enforces id
  uniqueness, vendor/passenger ranges, and an NYC GPS bounding box, then drops
  physically impossible trips (implied speed outside 0.5–80 mph, distance outside
  0.05–50 mi). Unparseable timestamps are coerced and dropped. This filters
  GPS-teleport glitches: **~1.46M raw rows → ~148k clean rows**.
- **Feature engineering** (`src/feature_engineering.py`) — three families:
  - **Spatial:** haversine (great-circle) *and* Manhattan distance (the street
    grid), plus compass bearing.
  - **Airport proximity:** distance from pickup/dropoff to **JFK** and **LaGuardia**.
  - **Temporal:** hour, day-of-week, month, weekend & rush-hour flags, and
    **cyclical hour encoding** (`hour_sin` / `hour_cos`) so 23:00 and 00:00 sit
    adjacent. A `log1p` target supports the RMSLE objective.
- **Versioning:** the processed dataset is tracked with **DVC**
  (`train_processed.parquet.dvc`, pointer committed to git) and marked with the
  annotated tag **`v1.0-data`**.

---

## Week 2 — Experimentation & Reproducibility (M3)

Five models are trained and compared; every run is tracked in MLflow with its
parameters, metrics, and the fitted pipeline.

```bash
python src/train.py                     # trains all 5 models + writes the comparison report
mlflow ui --backend-store-uri mlruns    # inspect runs at http://127.0.0.1:5000
python src/export_mlflow_runs.py        # export runs → reports/mlflow_runs.{md,csv}
```

The raw MLflow store (`mlruns/`) is git-ignored — it holds absolute paths and
binary artifacts. `export_mlflow_runs.py` reads that store directly (no `mlflow`
dependency) and writes durable, reviewable evidence of every tracked run to
`reports/mlflow_runs.md` and `reports/mlflow_runs.csv`, so anyone cloning the repo
sees the experiment history (run name, status, metrics, git commit) without
starting MLflow.

**Experiments**

| Model | Role | Preprocessing |
|-------|------|---------------|
| Linear Regression | interpretable baseline | median impute + standardize + one-hot |
| Random Forest | bagged-tree nonlinear baseline | median impute + one-hot |
| Gradient Boosting | sequential shallow-tree boosting | median impute + one-hot |
| Hist Gradient Boosting | fast histogram-based boosting | median impute + one-hot |
| XGBoost Regressor | gradient boosting (**winner**) | median impute + one-hot |

Each estimator is wrapped in a scikit-learn `Pipeline`, so preprocessing is bound
to the model and identical at train and serve time. Models are ranked on
validation **RMSE (seconds)**, with **RMSLE** (the competition metric), **MAE**,
and **R²** (118,538 train / 29,635 validation rows, seeded 80/20 split):

| Model | RMSLE | RMSE (s) | MAE (s) | R² |
|-------|------:|---------:|--------:|----:|
| **xgboost (winner)** | **0.305** | **276.1** | **169.5** | **0.818** |
| hist_gradient_boosting | 0.312 | 284.4 | 175.0 | 0.807 |
| random_forest | 0.324 | 291.5 | 180.3 | 0.797 |
| gradient_boosting | 0.333 | 307.3 | 190.4 | 0.774 |
| linear_regression | 0.498 | 557.2 | 312.6 | 0.258 |

The linear baseline barely beats guessing (R² 0.26); **XGBoost reaches R² 0.82
(~4.6 min error)** because trees capture the interaction between distance, time of
day, and location that a straight line cannot. All hyperparameters live in
`params.yaml`, seeds are fixed, and each run is tagged with its git commit — so
any run reproduces. The winner is serialized to `models/final_model.pkl` with
`models/model_metadata.json`, and results are written to
`reports/model_comparison_report.md`.

---

## Week 3 — Packaging & Deployment (M4)

The winning model is wrapped in a **FastAPI** service and packaged with Docker.

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000   # docs at http://localhost:8000/docs
```

**Endpoints**

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/` | Service status |
| GET | `/health` | Health / model-loaded status |
| GET | `/model-info` | Model type, features, target |
| POST | `/predict` | Predict trip duration (ETA) |

**Sample request**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_datetime": "2016-06-01T08:30:00",
    "pickup_longitude": -73.9855, "pickup_latitude": 40.7580,
    "dropoff_longitude": -73.9772, "dropoff_latitude": 40.7527,
    "passenger_count": 1, "vendor_id": 1, "store_and_fwd_flag": "N"
  }'
```

**Sample response**

```json
{ "predicted_duration_seconds": 612.4, "predicted_duration_minutes": 10.2 }
```

Key engineering properties:

- The API calls the **exact same `engineer_features()`** used in training —
  **zero train/serve skew**.
- **Pydantic** validates every field: an out-of-range `passenger_count` returns a
  clean **422**, and an invalid `store_and_fwd_flag` a **400** — never a raw stack
  trace. `pickup_datetime` must be a valid ISO timestamp (do not send `"string"`).

**Docker**

```bash
docker build -t nyc-taxi-eta:latest -f src/Dockerfile .
docker run -d --name nyc-taxi-eta-container -p 8000:8000 nyc-taxi-eta:latest
docker logs nyc-taxi-eta-container      # then open http://localhost:8000/docs
```

---

## Week 4 — Monitoring, Drift & Retraining (M5)

Models decay as traffic and seasonal patterns shift, so the deployed model is
monitored for drift.

```bash
python src/monitoring.py   # writes reports/monitoring_report.md
```

`monitoring.py` computes data-quality checks, prediction-quality checks (flagging
ETAs outside 0–300 min), and **Population Stability Index (PSI)** between a
reference window and current traffic. Example output
(`reports/monitoring_report.md`):

| Feature | PSI | Drift |
|---------|----:|-------|
| trip_distance | 0.162 | **MODERATE** |
| passenger_count | 0.0004 | LOW |

**Drift thresholds & retraining trigger**

| PSI | Classification | Action |
|-----|----------------|--------|
| < 0.10 | LOW | none |
| 0.10 – 0.25 | MODERATE | warning / watch |
| > 0.25 | HIGH | **fire retraining trigger** |

When any feature's PSI exceeds **0.25** (or labelled production error exceeds the
accepted baseline), the retraining strategy is to rerun `python src/train.py` on
fresh data and **validate the candidate before promotion** — closing the loop back
to Week 2.

---

## Week 5 — Deployment & Submission Packaging

Week 5 integrates the pieces into a shippable artifact: the FastAPI service is
containerized via `src/Dockerfile`, the model + metadata are committed so the
service loads without retraining, the monitoring report is generated, and the
documentation and demo are finalized. This is the state captured on the
`feature/week5-deployment` branch.

```bash
# One-command service from a clean checkout:
docker build -t nyc-taxi-eta:latest -f src/Dockerfile .
docker run -d -p 8000:8000 nyc-taxi-eta:latest
```

---

## Testing

```bash
pytest -q
```

- `tests/test_api.py` — endpoint responses, valid prediction, and input-validation
  errors (422 / 400).
- `tests/test_monitoring.py` — data-quality, MAE/RMSE, PSI computation, and drift
  classification boundaries.

---

## Design Decisions

Every design choice is justified, not just implemented:

- **`log1p` target / RMSLE objective** — trip durations are right-skewed; log-space
  training penalizes proportional error and matches the competition metric.
- **Manhattan distance alongside haversine** — NYC is a street grid, so block
  distance often models real travel better than great-circle distance.
- **XGBoost over the linear baseline** — trees capture distance × time × location
  interactions (R² 0.82 vs 0.26); see [Week 2](#week-2--experimentation--reproducibility-m3).
- **PSI for drift** — simple, threshold-interpretable, and standard for tabular
  feature-distribution monitoring; thresholds 0.10 / 0.25 are the conventional
  low/moderate/high bands.
- **Shared `engineer_features()` at train and serve** — the single most important
  correctness choice; it eliminates train/serve skew by construction.

---

## Reproducibility

| Tool | Role |
|------|------|
| Git | source control, run-to-commit tagging |
| DVC | dataset versioning (`v1.0-data` tag) |
| MLflow | experiment tracking (params + metrics + model) |
| `params.yaml` | single source of truth for config & hyperparameters |
| scikit-learn `Pipeline` | preprocessing bound to estimator |
| Docker | repeatable service packaging |

Fixed seeds (`random_state: 42`) make the split and models deterministic, and each
MLflow run is tagged with its git commit.

---

## Submission Checklist Mapping

| # | Deliverable | Where |
|---|-------------|-------|
| 1 | Versioned dataset + pipeline code | this repo · `src/` · `*.dvc` · tag `v1.0-data` |
| 2 | Experiment logs + comparison report | `mlflow ui` · `reports/mlflow_runs.md` · `reports/mlflow_runs.csv` · `reports/model_comparison_report.md` · `reports/model_comparison.csv` |
| 3 | Deployed model + API + sample calls | `src/api.py` · `src/Dockerfile` · [Week 3](#week-3--packaging--deployment-m4) curl examples |
| 4 | Monitoring log + drift report + retraining design | `src/monitoring.py` · `reports/monitoring_report.md` · [Week 4](#week-4--monitoring-drift--retraining-m5) |
| 5 | README + architecture diagram + demo | this file · [Architecture](#architecture) · [Demo & Report](#demo--report) |

### Demo & Report

The recorded demo and the final project report are available here:

📁 **[Demo & Report — Google Drive](https://drive.google.com/drive/folders/1sxA208pYxySr8vjQzEFEsYhWzbopABdq)**

---

## References

- **Dataset:** [NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration) (Kaggle)
- **Tools:** FastAPI · Uvicorn · Pydantic · MLflow · DVC · scikit-learn · XGBoost · pandera · Docker · pytest
- T1: Crowe et al., *Machine Learning Production Systems*, O'Reilly, 2024
- T2: Burkov, *Machine Learning Engineering*, 2020
- R1: McMahon, *Machine Learning Engineering with Python* (2nd ed.), Packt, 2023
```

