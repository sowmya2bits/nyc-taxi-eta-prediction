# Week 4 - Model Monitoring Report

## 1. Monitoring Objective

Monitor data quality, model predictions, data drift, and model performance for the NYC Taxi ETA prediction system.

## 2. Data Quality

- Rows monitored: 500
- Columns monitored: 2
- Missing values: 0
- Duplicate rows: 0
- Infinite values: 0
- Quality status: PASS

## 3. Prediction Monitoring

- Prediction count: 100
- Mean ETA: 25.491
- Median ETA: 26.346
- Standard deviation: 5.746
- Minimum ETA: 9.681
- Maximum ETA: 38.837
- P95 ETA: 34.443

## 4. Prediction Quality

- Invalid predictions: 0
- Invalid prediction rate: 0.00%
- Allowed range: 0.0 - 300.0 minutes
- Quality status: PASS

## 5. Data Drift

| Feature | PSI | Drift |
|---|---:|---|
| trip_distance | 0.1619 | MODERATE |
| passenger_count | 0.0004 | LOW |

## 7. Monitoring Thresholds

- PSI < 0.10: LOW drift
- PSI 0.10-0.25: MODERATE drift
- PSI > 0.25: HIGH drift
- ETA outside 0-300 minutes: invalid prediction

## 8. Conclusion

The Week 4 monitoring framework provides automated data-quality validation, prediction monitoring, drift detection, and performance evaluation.
