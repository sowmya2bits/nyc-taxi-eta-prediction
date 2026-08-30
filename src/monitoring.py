"""
Week 4 - Model Monitoring and Validation

Provides:
- Data quality checks
- Prediction monitoring
- PSI-based drift detection
- Model performance metrics
- Monitoring report generation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def check_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """
    Check basic data quality:
    - row/column count
    - missing values
    - duplicate rows
    - numeric infinite values
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    missing = df.isna().sum()
    duplicate_rows = int(df.duplicated().sum())

    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        infinite_values = 0
    else:
        infinite_values = int(
            np.isinf(numeric_df.to_numpy()).sum()
        )

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": int(missing.sum()),
        "missing_by_column": {
            str(k): int(v)
            for k, v in missing.items()
            if v > 0
        },
        "duplicate_rows": duplicate_rows,
        "infinite_values": infinite_values,
        "quality_passed": (
            int(missing.sum()) == 0
            and duplicate_rows == 0
            and infinite_values == 0
        ),
    }


def prediction_summary(predictions) -> dict[str, float]:
    """
    Calculate summary statistics for model predictions.
    """

    values = np.asarray(predictions, dtype=float)

    if values.size == 0:
        raise ValueError("predictions cannot be empty")

    if not np.isfinite(values).all():
        raise ValueError("predictions contain NaN or infinite values")

    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p95": float(np.percentile(values, 95)),
    }


def check_prediction_quality(
    predictions,
    minimum: float = 0.0,
    maximum: float = 300.0,
) -> dict[str, Any]:
    """
    Validate predictions against reasonable ETA boundaries.

    Default range:
        0 to 300 minutes
    """

    values = np.asarray(predictions, dtype=float)

    if values.size == 0:
        raise ValueError("predictions cannot be empty")

    invalid = (~np.isfinite(values)) | (values < minimum) | (values > maximum)

    return {
        "count": int(values.size),
        "invalid_count": int(invalid.sum()),
        "invalid_rate": float(invalid.mean()),
        "quality_passed": not bool(invalid.any()),
        "minimum_allowed": float(minimum),
        "maximum_allowed": float(maximum),
    }


def calculate_mae(actuals, predictions) -> float:
    """Calculate Mean Absolute Error."""

    actual = np.asarray(actuals, dtype=float)
    predicted = np.asarray(predictions, dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError("actuals and predictions must have the same shape")

    if actual.size == 0:
        raise ValueError("actuals and predictions cannot be empty")

    return float(np.mean(np.abs(actual - predicted)))


def calculate_rmse(actuals, predictions) -> float:
    """Calculate Root Mean Squared Error."""

    actual = np.asarray(actuals, dtype=float)
    predicted = np.asarray(predictions, dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError("actuals and predictions must have the same shape")

    if actual.size == 0:
        raise ValueError("actuals and predictions cannot be empty")

    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def _safe_histogram(values, bins):
    """
    Create normalized histogram percentages.
    """

    hist, _ = np.histogram(values, bins=bins)

    total = hist.sum()

    if total == 0:
        return np.zeros(len(hist))

    return hist / total


def calculate_psi(
    reference,
    current,
    bins: int = 10,
) -> float:
    """
    Calculate Population Stability Index (PSI).

    Interpretation:
        PSI < 0.10       -> little/no drift
        0.10 - 0.25      -> moderate drift
        > 0.25           -> significant drift
    """

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if reference.size == 0 or current.size == 0:
        raise ValueError("reference and current data cannot be empty")

    if np.all(reference == reference[0]):
        low = reference[0] - 0.5
        high = reference[0] + 0.5
        edges = np.linspace(low, high, bins + 1)
    else:
        edges = np.quantile(
            reference,
            np.linspace(0, 1, bins + 1),
        )
        edges = np.unique(edges)

        if len(edges) < 2:
            low = float(np.min(reference)) - 0.5
            high = float(np.max(reference)) + 0.5
            edges = np.linspace(low, high, bins + 1)

    reference_pct = _safe_histogram(reference, edges)
    current_pct = _safe_histogram(current, edges)

    epsilon = 1e-6

    reference_pct = np.clip(reference_pct, epsilon, None)
    current_pct = np.clip(current_pct, epsilon, None)

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return float(psi)


def classify_drift(psi: float) -> str:
    """Classify PSI drift level."""

    if psi < 0.10:
        return "LOW"
    if psi < 0.25:
        return "MODERATE"
    return "HIGH"


def detect_numeric_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    bins: int = 10,
) -> dict[str, dict[str, Any]]:
    """
    Calculate PSI for numeric columns present in both datasets.
    """

    if not isinstance(reference_df, pd.DataFrame):
        raise TypeError("reference_df must be a pandas DataFrame")

    if not isinstance(current_df, pd.DataFrame):
        raise TypeError("current_df must be a pandas DataFrame")

    common_columns = [
        column
        for column in reference_df.columns
        if column in current_df.columns
        and pd.api.types.is_numeric_dtype(reference_df[column])
        and pd.api.types.is_numeric_dtype(current_df[column])
    ]

    results = {}

    for column in common_columns:
        try:
            psi = calculate_psi(
                reference_df[column].dropna(),
                current_df[column].dropna(),
                bins=bins,
            )

            results[column] = {
                "psi": round(psi, 6),
                "drift": classify_drift(psi),
            }

        except ValueError:
            results[column] = {
                "psi": None,
                "drift": "UNAVAILABLE",
            }

    return results


def generate_monitoring_report(
    output_path: str | Path,
    data_quality: dict[str, Any],
    prediction_stats: dict[str, Any] | None = None,
    prediction_quality: dict[str, Any] | None = None,
    drift_results: dict[str, dict[str, Any]] | None = None,
    performance: dict[str, float] | None = None,
) -> Path:
    """
    Generate Markdown monitoring report.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Week 4 - Model Monitoring Report",
        "",
        "## 1. Monitoring Objective",
        "",
        "Monitor data quality, model predictions, data drift, "
        "and model performance for the NYC Taxi ETA prediction system.",
        "",
        "## 2. Data Quality",
        "",
        f"- Rows monitored: {data_quality['rows']}",
        f"- Columns monitored: {data_quality['columns']}",
        f"- Missing values: {data_quality['missing_values']}",
        f"- Duplicate rows: {data_quality['duplicate_rows']}",
        f"- Infinite values: {data_quality['infinite_values']}",
        f"- Quality status: "
        f"{'PASS' if data_quality['quality_passed'] else 'FAIL'}",
        "",
    ]

    if prediction_stats:
        lines.extend(
            [
                "## 3. Prediction Monitoring",
                "",
                f"- Prediction count: {prediction_stats['count']}",
                f"- Mean ETA: {prediction_stats['mean']:.3f}",
                f"- Median ETA: {prediction_stats['median']:.3f}",
                f"- Standard deviation: {prediction_stats['std']:.3f}",
                f"- Minimum ETA: {prediction_stats['min']:.3f}",
                f"- Maximum ETA: {prediction_stats['max']:.3f}",
                f"- P95 ETA: {prediction_stats['p95']:.3f}",
                "",
            ]
        )

    if prediction_quality:
        lines.extend(
            [
                "## 4. Prediction Quality",
                "",
                f"- Invalid predictions: "
                f"{prediction_quality['invalid_count']}",
                f"- Invalid prediction rate: "
                f"{prediction_quality['invalid_rate']:.2%}",
                f"- Allowed range: "
                f"{prediction_quality['minimum_allowed']} - "
                f"{prediction_quality['maximum_allowed']} minutes",
                f"- Quality status: "
                f"{'PASS' if prediction_quality['quality_passed'] else 'FAIL'}",
                "",
            ]
        )

    if drift_results:
        lines.extend(
            [
                "## 5. Data Drift",
                "",
                "| Feature | PSI | Drift |",
                "|---|---:|---|",
            ]
        )

        for feature, result in drift_results.items():
            psi = result["psi"]

            if psi is None:
                psi_text = "N/A"
            else:
                psi_text = f"{psi:.4f}"

            lines.append(
                f"| {feature} | {psi_text} | {result['drift']} |"
            )

        lines.append("")

    if performance:
        lines.extend(
            [
                "## 6. Model Performance",
                "",
                f"- MAE: {performance['mae']:.4f}",
                f"- RMSE: {performance['rmse']:.4f}",
                "",
            ]
        )

    lines.extend(
        [
            "## 7. Monitoring Thresholds",
            "",
            "- PSI < 0.10: LOW drift",
            "- PSI 0.10-0.25: MODERATE drift",
            "- PSI > 0.25: HIGH drift",
            "- ETA outside 0-300 minutes: invalid prediction",
            "",
            "## 8. Conclusion",
            "",
            "The Week 4 monitoring framework provides automated "
            "data-quality validation, prediction monitoring, "
            "drift detection, and performance evaluation.",
            "",
        ]
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return output_path


if __name__ == "__main__":
    # Small local demonstration so the monitoring module
    # can be executed independently.

    rng = np.random.default_rng(42)

    reference = pd.DataFrame(
        {
            "trip_distance": rng.normal(5, 1, 1000),
            "passenger_count": rng.integers(1, 4, 1000),
        }
    )

    current = pd.DataFrame(
        {
            "trip_distance": rng.normal(5.5, 1.2, 500),
            "passenger_count": rng.integers(1, 4, 500),
        }
    )

    predictions = rng.normal(25, 5, 100)

    quality = check_data_quality(current)
    prediction_stats = prediction_summary(predictions)
    prediction_quality = check_prediction_quality(predictions)
    drift = detect_numeric_drift(reference, current)

    report = generate_monitoring_report(
        "reports/monitoring_report.md",
        data_quality=quality,
        prediction_stats=prediction_stats,
        prediction_quality=prediction_quality,
        drift_results=drift,
    )

    print("Monitoring completed.")
    print(f"Report generated: {report}")
