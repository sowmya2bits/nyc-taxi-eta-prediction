import os
import numpy as np
import pandas as pd

# NYC Airport Landmarks (Latitude, Longitude)
JFK_COORDS = (40.6413, -73.7781)
LGA_COORDS = (40.7769, -73.8740)

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates spherical distance in miles."""
    r = 3958.8
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    return r * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def manhattan_distance(lat1, lon1, lat2, lon2):
    """Calculates grid distance along city street blocks."""
    lat_dist = haversine_distance(lat1, lon1, lat2, lon1)
    lon_dist = haversine_distance(lat1, lon1, lat1, lon2)
    return lat_dist + lon_dist

def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculates compass heading angle in degrees (0 - 360)."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlambda = np.radians(lon2 - lon1)
    y = np.sin(dlambda) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    return np.degrees(np.arctan2(y, x)) % 360

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generates spatial, temporal, cyclical, and target-log features."""
    print("[Features] Engineering spatial and temporal features...")
    df = df.copy()

    # 1. Spatial Features
    df['haversine_dist'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    df['manhattan_dist'] = manhattan_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )
    df['direction_bearing'] = calculate_bearing(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )

    # Airport Proximity (JFK & LaGuardia)
    df['jfk_pickup_dist'] = haversine_distance(df['pickup_latitude'], df['pickup_longitude'], JFK_COORDS[0], JFK_COORDS[1])
    df['jfk_dropoff_dist'] = haversine_distance(df['dropoff_latitude'], df['dropoff_longitude'], JFK_COORDS[0], JFK_COORDS[1])
    df['lga_pickup_dist'] = haversine_distance(df['pickup_latitude'], df['pickup_longitude'], LGA_COORDS[0], LGA_COORDS[1])
    df['lga_dropoff_dist'] = haversine_distance(df['dropoff_latitude'], df['dropoff_longitude'], LGA_COORDS[0], LGA_COORDS[1])

    # 2. Temporal Features
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['pickup_dayofweek'] = df['pickup_datetime'].dt.dayofweek
    df['pickup_month'] = df['pickup_datetime'].dt.month
    df['is_weekend'] = df['pickup_dayofweek'].isin([5, 6]).astype(int)
    df['is_rush_hour'] = df['pickup_hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    # Cyclical Encoding (Hour of day)
    df['hour_sin'] = np.sin(2 * np.pi * df['pickup_hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['pickup_hour'] / 24.0)

    # 3. Target Transformation (Optimizes for RMSLE metric)
    df['log_trip_duration'] = np.log1p(df['trip_duration'])

    print(f"[Features] Successfully engineered dataset with {df.shape[1]} total columns.")
    return df

if __name__ == "__main__":
    from ingestion import ingest_data
    from validation import validate_and_clean_data

    raw_df = ingest_data()
    clean_df = validate_and_clean_data(raw_df)
    featured_df = engineer_features(clean_df)

    # Save to data/processed in Parquet format
    os.makedirs("data/processed", exist_ok=True)
    output_path = "data/processed/train_processed.parquet"
    featured_df.to_parquet(output_path, index=False)
    print(f"[Features] Saved processed feature dataset to '{output_path}'.")