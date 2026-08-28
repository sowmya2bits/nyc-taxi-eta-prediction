# Week 2 — Experimentation & Model Comparison (Module M3)

**Brief task:** *Train and compare models (e.g., linear regression vs. gradient
boosting); track experiments and hyperparameters.*
**Milestone:** *At least two tracked experiments completed; best model
identified with justification.*
**Deliverable (checklist #2):** *Experiment tracking logs and a short model
comparison report (e.g., MLflow).*

---

## 1. Clone repo & check out code

```powershell
git clone <repository-url>
cd nyc-taxi-eta-prediction
git checkout feature_engineering_model_training
```

## 2. Set up virtual environment & dependencies

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows PowerShell
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

(Use Python 3.12 — 3.14 has spotty ML wheels.)

## 3. Get the processed dataset

The processed features are versioned with DVC. Either pull them or regenerate
from the raw Kaggle `train.csv`:

```powershell
dvc pull                              # if a DVC remote is configured
# --- or rebuild locally ---
# place train.zip / train.csv in data/raw/, then:
python src/feature_engineering.py     # ingestion -> validation -> features
```

This writes `data/processed/train_processed.parquet` (raw files are git-ignored,
so this step is local-only).

## 4. Train, compare, and track the experiments

```powershell
python src/train.py
```

Trains **five** models in MLflow experiment `nyc-taxi-eta`, logging params +
`rmsle`/`rmse_sec`/`mae_sec`/`r2` + the fitted pipeline per run:

| Model | Role |
|---|---|
| Linear Regression | interpretable baseline |
| Random Forest | bagged-tree nonlinear baseline |
| Gradient Boosting | sequential shallow-tree boosting |
| Hist Gradient Boosting | histogram-based boosting (fast on large data) |
| XGBoost | gradient boosting, usual winner |

All hyperparameters live in `params.yaml`. Each model is wrapped in a
scikit-learn `Pipeline` (impute + scale/encode) so the exact preprocessing is
bound to the estimator — eliminating the training–serving skew from Week 1.

Then it:

- saves the lowest-RMSE model → `models/final_model.pkl`
- writes `models/model_metadata.json`
- writes **`reports/model_comparison_report.md`** (Deliverable #2) + `.csv`

Inspect / screenshot the runs:

```powershell
mlflow ui        # http://127.0.0.1:5000
```

## 5. What to hand in for Week 2

| Item | Where |
|---|---|
| ≥2 tracked experiments (params, metrics, artifacts) | `mlruns/` + MLflow UI screenshot |
| Short model comparison report | `reports/model_comparison_report.md` |
| Best model + justification | report §"Selected model" |
| Reproducibility | pinned `requirements.txt`, `params.yaml`, seeded split, DVC-versioned parquet |

## 6. Commit (incremental weekly history is graded)

```powershell
git add src/train.py params.yaml requirements.txt WEEK2.md README.md .gitignore \
        reports/model_comparison_report.md
git commit -m "Week 2 (M3): train & compare 5 models with MLflow tracking + comparison report"
git tag week2-experiments
```

> Don't commit `data/`, `models/*.pkl`, or `mlruns/` (all git-ignored). The
> processed dataset is tracked with DVC (`dvc add data/processed/train_processed.parquet`).
