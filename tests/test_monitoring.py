import numpy as np
import pandas as pd

from src.monitoring import (
    check_data_quality,
    prediction_summary,
    check_prediction_quality,
    calculate_mae,
    calculate_rmse,
    calculate_psi,
    classify_drift,
    detect_numeric_drift,
    generate_monitoring_report,
)


def test_data_quality_passes_for_clean_data():
    df = pd.DataFrame(
        {
            "trip_distance": [1.0, 2.0, 3.0],
            "passenger_count": [1, 2, 1],
        }
    )

    result = check_data_quality(df)

    assert result["rows"] == 3
    assert result["columns"] == 2
    assert result["missing_values"] == 0
    assert result["duplicate_rows"] == 0
    assert result["quality_passed"] is True


def test_data_quality_detects_missing_values():
    df = pd.DataFrame(
        {
            "trip_distance": [1.0, None, 3.0],
            "passenger_count": [1, 2, 1],
        }
    )

    result = check_data_quality(df)

    assert result["missing_values"] == 1
    assert result["quality_passed"] is False


def test_prediction_summary():
    predictions = [10, 20, 30]

    result = prediction_summary(predictions)

    assert result["count"] == 3
    assert result["mean"] == 20
    assert result["min"] == 10
    assert result["max"] == 30


def test_prediction_quality():
    predictions = [10, 20, 30, 400]

    result = check_prediction_quality(predictions)

    assert result["invalid_count"] == 1
    assert result["quality_passed"] is False


def test_mae():
    actual = [10, 20, 30]
    predicted = [12, 18, 33]

    result = calculate_mae(actual, predicted)

    assert round(result, 6) == round((2 + 2 + 3) / 3, 6)


def test_rmse():
    actual = [10, 20, 30]
    predicted = [12, 18, 33]

    result = calculate_rmse(actual, predicted)

    expected = np.sqrt((4 + 4 + 9) / 3)

    assert round(result, 6) == round(expected, 6)


def test_psi_detects_no_major_drift():
    reference = np.random.default_rng(42).normal(10, 2, 1000)
    current = np.random.default_rng(43).normal(10, 2, 1000)

    psi = calculate_psi(reference, current)

    assert psi < 0.10


def test_drift_classification():
    assert classify_drift(0.05) == "LOW"
    assert classify_drift(0.15) == "MODERATE"
    assert classify_drift(0.30) == "HIGH"


def test_numeric_drift():
    rng = np.random.default_rng(42)

    reference = pd.DataFrame(
        {
            "trip_distance": rng.normal(5, 1, 500),
            "passenger_count": rng.integers(1, 4, 500),
        }
    )

    current = pd.DataFrame(
        {
            "trip_distance": rng.normal(6, 1, 500),
            "passenger_count": rng.integers(1, 4, 500),
        }
    )

    result = detect_numeric_drift(reference, current)

    assert "trip_distance" in result
    assert "passenger_count" in result
    assert "psi" in result["trip_distance"]
    assert "drift" in result["trip_distance"]


def test_monitoring_report_is_created(tmp_path):
    output = tmp_path / "monitoring_report.md"

    quality = {
        "rows": 10,
        "columns": 2,
        "missing_values": 0,
        "duplicate_rows": 0,
        "infinite_values": 0,
        "quality_passed": True,
    }

    prediction_stats = prediction_summary([10, 20, 30])

    prediction_quality = check_prediction_quality(
        [10, 20, 30]
    )

    generate_monitoring_report(
        output,
        quality,
        prediction_stats,
        prediction_quality,
    )

    assert output.exists()
    assert output.stat().st_size > 0

    content = output.read_text(encoding="utf-8")

    assert "Week 4 - Model Monitoring Report" in content
    assert "Data Quality" in content
    assert "Prediction Monitoring" in content
