"""Week 2 (M3): train and compare ETA models with MLflow experiment tracking.

Trains five models on the versioned processed dataset and logs every run
(parameters, metrics, and the fitted pipeline) to MLflow:

  1. Linear Regression         - fast, interpretable baseline.
  2. Random Forest             - bagged trees, strong nonlinear baseline.
  3. Gradient Boosting         - sequential boosting of shallow trees.
  4. Hist Gradient Boosting    - histogram-based boosting, fast on large data.
  5. XGBoost Regressor         - gradient boosting, usually the winner.

Each is wrapped in a scikit-learn Pipeline so the exact same preprocessing
(imputation + scaling/encoding) is bound to the estimator, preventing the
training-serving skew observed in the Week 1 fragile service. The best model
(lowest validation RMSE) is serialized to models/final_model.pkl and a
comparison report is written to reports/.
"""

import os
import json

import numpy as np
import pandas as pd
import yaml
import joblib
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

PARAMS_PATH = "params.yaml"
MODEL_DIR = "models"
REPORT_DIR = "reports"


def load_params(path: str = PARAMS_PATH) -> dict:
    """Loads the central experiment configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def load_dataset(cfg: dict):
    """Loads the DVC-tracked processed dataset and splits X / y."""
    data_cfg = cfg["data"]
    path = data_cfg["processed_path"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed dataset '{path}' not found. Run `dvc pull` or "
            f"`python src/feature_engineering.py` to regenerate it first."
        )

    print(f"[Train] Loading processed dataset from '{path}'...")
    df = pd.read_parquet(path)

    feature_cols = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    target = data_cfg["target"]

    X = df[feature_cols]
    y = df[target]  # log1p(trip_duration)
    print(f"[Train] Loaded {len(df):,} rows | {len(feature_cols)} features | target='{target}'.")
    return X, y


def build_preprocessor(cfg: dict, scale_numeric: bool) -> ColumnTransformer:
    """Builds preprocessing shared by both models.

    Numeric columns are median-imputed (and standardized for the linear model);
    categorical columns are imputed and one-hot encoded with unseen categories
    handled gracefully at serving time.
    """
    numeric = cfg["features"]["numeric"]
    categorical = cfg["features"]["categorical"]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), numeric),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ]
    )


def evaluate(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    """Computes metrics on both log and original (seconds) scales.

    RMSLE equals the RMSE in log space because the target is log1p(duration) —
    this is the competition metric for NYC Taxi Trip Duration.
    """
    rmsle = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))

    y_true = np.expm1(y_true_log)
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {"rmsle": rmsle, "rmse_sec": rmse, "mae_sec": mae, "r2": r2}


def train_and_log(name: str, pipeline: Pipeline, params: dict,
                  X_train, X_val, y_train, y_val) -> dict:
    """Fits one pipeline, logs the run to MLflow, and returns its metrics."""
    print(f"\n[Train] === Experiment: {name} ===")
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model_type", name)
        mlflow.log_params(params)

        pipeline.fit(X_train, y_train)
        metrics = evaluate(y_val.to_numpy(), pipeline.predict(X_val))

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        print(f"[Train] {name}: RMSLE={metrics['rmsle']:.4f} | "
              f"RMSE={metrics['rmse_sec']:.1f}s | MAE={metrics['mae_sec']:.1f}s | "
              f"R2={metrics['r2']:.4f}")

    return {"model_name": name, "pipeline": pipeline, "metrics": metrics}


def build_experiments(cfg: dict) -> dict:
    """Assembles the model pipelines defined in params.yaml.

    Only the linear model needs standardized inputs; the tree/ensemble models
    are scale-invariant, so numeric columns are passed through for them.
    """
    m = cfg["models"]
    specs = {
        "linear_regression": (LinearRegression(**m["linear_regression"]),
                              m["linear_regression"], True),
        "random_forest": (RandomForestRegressor(**m["random_forest"]),
                          m["random_forest"], False),
        "gradient_boosting": (GradientBoostingRegressor(**m["gradient_boosting"]),
                             m["gradient_boosting"], False),
        "hist_gradient_boosting": (HistGradientBoostingRegressor(**m["hist_gradient_boosting"]),
                                  m["hist_gradient_boosting"], False),
        "xgboost": (XGBRegressor(**m["xgboost"]), m["xgboost"], False),
    }

    return {
        name: {
            "params": params,
            "pipeline": Pipeline([
                ("prep", build_preprocessor(cfg, scale_numeric=scale)),
                ("model", estimator),
            ]),
        }
        for name, (estimator, params, scale) in specs.items()
    }


def write_reports(results: list, best: dict, n_train: int, n_val: int) -> None:
    """Writes the model-comparison report (CSV + Markdown) required by M3."""
    os.makedirs(REPORT_DIR, exist_ok=True)

    table = pd.DataFrame([{"model": r["model_name"], **r["metrics"]} for r in results])
    table = table.sort_values("rmse_sec").reset_index(drop=True)
    table.to_csv(os.path.join(REPORT_DIR, "model_comparison.csv"), index=False)
    print("\n[Train] Model comparison (sorted by validation RMSE):")
    print(table.to_string(index=False))

    lines = [
        "# Week 2 — Model Comparison Report (M3)",
        "",
        f"{len(results)} experiments were trained on the DVC-versioned processed "
        f"dataset ({n_train:,} train / {n_val:,} validation rows, 80/20 seeded split) "
        "and tracked with MLflow. Models are ranked by validation RMSE (seconds); "
        "RMSLE (log-space RMSE) is the NYC Taxi Trip Duration competition metric.",
        "",
        "| Model | RMSLE | RMSE (s) | MAE (s) | R2 |",
        "|-------|------:|---------:|--------:|---:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {row['model']} | {row['rmsle']:.4f} | {row['rmse_sec']:.1f} "
            f"| {row['mae_sec']:.1f} | {row['r2']:.4f} |"
        )
    lines += [
        "",
        f"## Selected model: **{best['model_name']}**",
        "",
        f"`{best['model_name']}` is selected as the best model — it achieves the "
        f"lowest validation RMSE ({best['metrics']['rmse_sec']:.1f}s) and RMSLE "
        f"({best['metrics']['rmsle']:.4f}). Tree-based ensembles capture the "
        "non-linear interactions between distance, time-of-day and location "
        "features that the linear baseline cannot. The full pipeline (preprocessing "
        "+ estimator) is serialized to `models/final_model.pkl` and every "
        "run is reproducible from `params.yaml` via the MLflow logs.",
        "",
    ]
    with open(os.path.join(REPORT_DIR, "model_comparison_report.md"), "w") as f:
        f.write("\n".join(lines))
    print(f"[Train] Comparison report written to '{REPORT_DIR}/model_comparison_report.md'.")


def main() -> None:
    cfg = load_params()

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    X, y = load_dataset(cfg)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
    )
    print(f"[Train] Split into {len(X_train):,} train / {len(X_val):,} validation rows.")

    experiments = build_experiments(cfg)
    results = [
        train_and_log(name, spec["pipeline"], spec["params"],
                      X_train, X_val, y_train, y_val)
        for name, spec in experiments.items()
    ]

    best = min(results, key=lambda r: r["metrics"]["rmse_sec"])
    print(f"\n[Train] Best model: '{best['model_name']}' "
          f"(RMSE={best['metrics']['rmse_sec']:.1f}s).")

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "final_model.pkl")
    joblib.dump(best["pipeline"], model_path)

    metadata = {
        "model_name": best["model_name"],
        "metrics": best["metrics"],
        "features": cfg["features"],
        "target": cfg["data"]["target"],
    }
    with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    write_reports(results, best, len(X_train), len(X_val))
    print(f"[Train] Saved best model and metadata to '{MODEL_DIR}/'.")


if __name__ == "__main__":
    main()
