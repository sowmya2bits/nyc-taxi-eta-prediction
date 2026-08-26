import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import Series

LAT_MIN, LAT_MAX = 40.50, 40.92
LON_MIN, LON_MAX = -74.25, -73.70

def haversine_distance(lat1, lon1, lat2, lon2):
    r = 3958.8  # Radius in miles
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return r * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

class NYCValidationSchema(pa.DataFrameModel):
    id: Series[str] = pa.Field(unique=True, nullable=False)
    vendor_id: Series[int] = pa.Field(isin=[1, 2])
    passenger_count: Series[int] = pa.Field(ge=1, le=8)
    pickup_longitude: Series[float] = pa.Field(ge=LON_MIN, le=LON_MAX)
    pickup_latitude: Series[float] = pa.Field(ge=LAT_MIN, le=LAT_MAX)
    dropoff_longitude: Series[float] = pa.Field(ge=LON_MIN, le=LON_MAX)
    dropoff_latitude: Series[float] = pa.Field(ge=LAT_MIN, le=LAT_MAX)
    trip_duration: Series[int] = pa.Field(gt=10, le=7200)

    class Config:
        coerce = True
        drop_invalid_rows = True

def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    initial_rows = len(df)
    print(f"[Validation] Starting validation on {initial_rows:,} rows...")

    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
    df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'], errors='coerce')
    df = df.dropna(subset=['pickup_datetime', 'dropoff_datetime'])

    # 1. Schema & GPS Bounds Check
    validated_df = NYCValidationSchema.validate(df, lazy=True)

    # 2. Distance Calculation
    calc_dist = haversine_distance(
        validated_df['pickup_latitude'], validated_df['pickup_longitude'],
        validated_df['dropoff_latitude'], validated_df['dropoff_longitude']
    )

    # 3. Implied Speed Calculation (mph)
    duration_hours = validated_df['trip_duration'] / 3600.0
    speed_mph = calc_dist / duration_hours

    # 4. Multi-condition Filter (Distance: 0.05-50 miles | Speed: 0.5-80 mph)
    mask = (calc_dist >= 0.05) & (calc_dist <= 50.0) & (speed_mph >= 0.5) & (speed_mph <= 80.0)
    clean_df = validated_df[mask].copy()

    removed_rows = initial_rows - len(clean_df)
    print(f"[Validation] Validation complete. Retained {len(clean_df):,} rows ({removed_rows:,} invalid/outlier rows dropped).")
    
    return clean_df

if __name__ == "__main__":
    from ingestion import ingest_data
    raw_df = ingest_data()
    clean_df = validate_and_clean_data(raw_df)