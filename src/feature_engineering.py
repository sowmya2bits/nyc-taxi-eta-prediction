import os
import numpy as np
import pandas as pd


# ============================================================
# NYC Airport Landmarks (Latitude, Longitude)
# ============================================================

JFK_COORDS = (40.6413, -73.7781)
LGA_COORDS = (40.7769, -73.8740)


# ============================================================
# Distance Features
# ============================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates spherical distance in miles."""
    r = 3958.8

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = (
        np.sin(dphi / 2) ** 2
        + np.cos(phi1)
        * np.cos(phi2)
        * np.sin(dlambda / 2) ** 2
    )

    return r * 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )


def manhattan_distance(lat1, lon1, lat2, lon2):
    """Calculates grid distance along city street blocks."""

    lat_dist = haversine_distance(
        lat1,
        lon1,
        lat2,
        lon1
    )

    lon_dist = haversine_distance(
        lat1,
        lon1,
        lat1,
        lon2
    )

    return lat_dist + lon_dist


def calculate_bearing(lat1, lon1, lat2, lon2):
    """Calculates compass heading angle in degrees (0 - 360)."""

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlambda = np.radians(lon2 - lon1)

    y = np.sin(dlambda) * np.cos(phi2)

    x = (
        np.cos(phi1) * np.sin(phi2)
        - np.sin(phi1)
        * np.cos(phi2)
        * np.cos(dlambda)
    )

    return np.degrees(np.arctan2(y, x)) % 360


# ============================================================
# Feature Engineering
# ============================================================

# OLD VERSION:
# def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

# NEW VERSION:
# include_target=True  -> used during training
# include_target=False -> used during API prediction
def engineer_features(
    df: pd.DataFrame,
    include_target: bool = True
) -> pd.DataFrame:

    """
    Generates spatial, temporal, cyclical,
    and optional target-log features.
    """

    print("[Features] Engineering spatial and temporal features...")

    df = df.copy()


    # ========================================================
    # NEW FIX:
    # Convert pickup_datetime to pandas datetime BEFORE
    # using .dt.hour, .dt.dayofweek, .dt.month, etc.
    #
    # This is required because API requests will send
    # pickup_datetime as a string.
    # ========================================================

    if 'pickup_datetime' in df.columns:

        df['pickup_datetime'] = pd.to_datetime(
            df['pickup_datetime'],
            errors='raise'
        )


    # ========================================================
    # 1. Spatial Features
    # ========================================================

    df['haversine_dist'] = haversine_distance(
        df['pickup_latitude'],
        df['pickup_longitude'],
        df['dropoff_latitude'],
        df['dropoff_longitude']
    )


    df['manhattan_dist'] = manhattan_distance(
        df['pickup_latitude'],
        df['pickup_longitude'],
        df['dropoff_latitude'],
        df['dropoff_longitude']
    )


    df['direction_bearing'] = calculate_bearing(
        df['pickup_latitude'],
        df['pickup_longitude'],
        df['dropoff_latitude'],
        df['dropoff_longitude']
    )


    # ========================================================
    # Airport Proximity Features
    # JFK & LaGuardia
    # ========================================================

    df['jfk_pickup_dist'] = haversine_distance(
        df['pickup_latitude'],
        df['pickup_longitude'],
        JFK_COORDS[0],
        JFK_COORDS[1]
    )


    df['jfk_dropoff_dist'] = haversine_distance(
        df['dropoff_latitude'],
        df['dropoff_longitude'],
        JFK_COORDS[0],
        JFK_COORDS[1]
    )


    df['lga_pickup_dist'] = haversine_distance(
        df['pickup_latitude'],
        df['pickup_longitude'],
        LGA_COORDS[0],
        LGA_COORDS[1]
    )


    df['lga_dropoff_dist'] = haversine_distance(
        df['dropoff_latitude'],
        df['dropoff_longitude'],
        LGA_COORDS[0],
        LGA_COORDS[1]
    )


    # ========================================================
    # 2. Temporal Features
    # ========================================================

    df['pickup_hour'] = (
        df['pickup_datetime'].dt.hour
    )


    df['pickup_dayofweek'] = (
        df['pickup_datetime'].dt.dayofweek
    )


    df['pickup_month'] = (
        df['pickup_datetime'].dt.month
    )


    df['is_weekend'] = (
        df['pickup_dayofweek']
        .isin([5, 6])
        .astype(int)
    )


    df['is_rush_hour'] = (
        df['pickup_hour']
        .isin([7, 8, 9, 16, 17, 18, 19])
        .astype(int)
    )


    # ========================================================
    # Cyclical Encoding
    # ========================================================

    df['hour_sin'] = np.sin(
        2 * np.pi * df['pickup_hour'] / 24.0
    )


    df['hour_cos'] = np.cos(
        2 * np.pi * df['pickup_hour'] / 24.0
    )


    # ========================================================
    # 3. Target Transformation
    #
    # OLD VERSION:
    # df['log_trip_duration'] = np.log1p(df['trip_duration'])
    #
    # PROBLEM:
    # During API prediction there is NO trip_duration,
    # because trip_duration is the value we are predicting.
    # ========================================================

    # OLD TARGET LINE - COMMENTED OUT:
    # df['log_trip_duration'] = np.log1p(df['trip_duration'])


    # ========================================================
    # NEW TARGET LOGIC:
    #
    # Training:
    #     include_target=True
    #     trip_duration exists
    #     log_trip_duration is created
    #
    # API prediction:
    #     include_target=False
    #     trip_duration does NOT exist
    #     log_trip_duration is NOT created
    # ========================================================

    if include_target and 'trip_duration' in df.columns:

        df['log_trip_duration'] = np.log1p(
            df['trip_duration']
        )


    # ========================================================
    # Final Information
    # ========================================================

    print(
        f"[Features] Successfully engineered dataset "
        f"with {df.shape[1]} total columns."
    )

    return df


# ============================================================
# Main - Training / Dataset Generation
# ============================================================

if __name__ == "__main__":

    from ingestion import ingest_data
    from validation import validate_and_clean_data


    # --------------------------------------------------------
    # Load raw data
    # --------------------------------------------------------

    raw_df = ingest_data()


    # --------------------------------------------------------
    # Validate and clean data
    # --------------------------------------------------------

    clean_df = validate_and_clean_data(
        raw_df
    )


    # --------------------------------------------------------
    # Feature engineering
    #
    # include_target=True because training data contains
    # trip_duration and we need log_trip_duration.
    # --------------------------------------------------------

    # OLD VERSION:
    # featured_df = engineer_features(clean_df)

    # NEW VERSION:
    featured_df = engineer_features(
        clean_df,
        include_target=True
    )


    # --------------------------------------------------------
    # Save processed dataset to Parquet
    # --------------------------------------------------------

    os.makedirs(
        "data/processed",
        exist_ok=True
    )


    output_path = (
        "data/processed/train_processed.parquet"
    )


    featured_df.to_parquet(
        output_path,
        index=False
    )


    print(
        f"[Features] Saved processed feature dataset "
        f"to '{output_path}'."
    )