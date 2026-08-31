import joblib
import numpy as np
import pandas as pd

from datetime import datetime

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.feature_engineering import engineer_features


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/final_model.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load(MODEL_PATH)

    print(
        f"[API] Model loaded successfully from: {MODEL_PATH}"
    )

except Exception as exc:
    raise RuntimeError(
        f"[API] Failed to load model: {exc}"
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="NYC Taxi ETA Prediction API",
    description=(
        "REST API for predicting NYC taxi trip duration "
        "using an XGBoost model."
    ),
    version="1.0.0"
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class PredictionRequest(BaseModel):

    # --------------------------------------------------------
    # Date / Time
    # --------------------------------------------------------

    pickup_datetime: datetime = Field(
        ...,
        description=(
            "Pickup date and time in YYYY-MM-DDTHH:MM:SS format."
        ),
        examples=["2016-06-01T08:30:00"]
    )

    # --------------------------------------------------------
    # Pickup Location
    # --------------------------------------------------------

    pickup_longitude: float = Field(
        ...,
        description="Pickup longitude",
        examples=[-73.9857]
    )

    pickup_latitude: float = Field(
        ...,
        description="Pickup latitude",
        examples=[40.7484]
    )

    # --------------------------------------------------------
    # Dropoff Location
    # --------------------------------------------------------

    dropoff_longitude: float = Field(
        ...,
        description="Dropoff longitude",
        examples=[-73.9851]
    )

    dropoff_latitude: float = Field(
        ...,
        description="Dropoff latitude",
        examples=[40.7580]
    )

    # --------------------------------------------------------
    # Passenger Information
    # --------------------------------------------------------

    passenger_count: int = Field(
        default=1,
        ge=1,
        le=6,
        description="Number of passengers",
        examples=[2]
    )

    # --------------------------------------------------------
    # Taxi Information
    # --------------------------------------------------------

    vendor_id: int = Field(
        default=1,
        description="Taxi vendor ID",
        examples=[1]
    )

    store_and_fwd_flag: str = Field(
        default="N",
        description="Store and forward flag: Y or N",
        examples=["N"]
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "NYC Taxi ETA Prediction API",
        "status": "running",
        "model": "XGBoost",
        "version": "1.0.0"
    }


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# ============================================================
# MODEL INFO ENDPOINT
# ============================================================

@app.get("/model-info")
def model_info():

    feature_names = getattr(
        model,
        "feature_names_in_",
        None
    )

    if feature_names is None:

        return {
            "model_type": "XGBoost",
            "expected_features": [],
            "number_of_features": None,
            "target": "log_trip_duration"
        }

    return {
        "model_type": "XGBoost",
        "expected_features": list(feature_names),
        "number_of_features": len(feature_names),
        "target": "log_trip_duration"
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    try:

        # ====================================================
        # 1. DATETIME
        # ====================================================

        # Pydantic has already validated pickup_datetime
        # and converted it into a Python datetime object.

        pickup_datetime = request.pickup_datetime


        # ====================================================
        # 2. VALIDATE STORE/FWD FLAG
        # ====================================================

        store_flag = (
            request.store_and_fwd_flag
            .strip()
            .upper()
        )

        if store_flag not in ["Y", "N"]:

            raise HTTPException(
                status_code=400,
                detail=(
                    "store_and_fwd_flag must be "
                    "'Y' or 'N'."
                )
            )


        # ====================================================
        # 3. CREATE INPUT DATAFRAME
        # ====================================================

        input_data = {

            "pickup_datetime": [
                pickup_datetime
            ],

            "pickup_longitude": [
                request.pickup_longitude
            ],

            "pickup_latitude": [
                request.pickup_latitude
            ],

            "dropoff_longitude": [
                request.dropoff_longitude
            ],

            "dropoff_latitude": [
                request.dropoff_latitude
            ],

            "passenger_count": [
                request.passenger_count
            ],

            "vendor_id": [
                request.vendor_id
            ],

            "store_and_fwd_flag": [
                store_flag
            ]
        }


        input_df = pd.DataFrame(
            input_data
        )


        # ====================================================
        # 4. FEATURE ENGINEERING
        #
        # trip_duration is NOT supplied because it is the
        # target that we are trying to predict.
        # ====================================================

        features_df = engineer_features(
            input_df,
            include_target=False
        )


        # ====================================================
        # 5. REMOVE NON-MODEL COLUMNS
        # ====================================================

        columns_to_remove = [

            "pickup_datetime",

            "trip_duration",

            "log_trip_duration"

        ]

        X = features_df.drop(
            columns=[
                column
                for column in columns_to_remove
                if column in features_df.columns
            ],
            errors="ignore"
        )


        # ====================================================
        # 6. CHECK MODEL FEATURES
        # ====================================================

        expected_features = getattr(
            model,
            "feature_names_in_",
            None
        )

        if expected_features is not None:

            expected_features = list(
                expected_features
            )

            missing_features = [
                feature
                for feature in expected_features
                if feature not in X.columns
            ]

            if missing_features:

                raise ValueError(
                    "Missing model features: "
                    f"{missing_features}"
                )

            # Keep exactly the same feature order
            # used when the model was trained.

            X = X[
                expected_features
            ]


        # ====================================================
        # 7. DEBUG INFORMATION
        # ====================================================

        print(
            "\n[API] Input features:"
        )

        print(
            X.to_dict(
                orient="records"
            )[0]
        )

        print(
            "\n[API] Feature count:",
            X.shape[1]
        )


        # ====================================================
        # 8. MODEL PREDICTION
        # ====================================================

        model_prediction = model.predict(X)

        raw_prediction = float(
            model_prediction[0]
        )

        print(
            "\n[API] Raw model prediction:",
            raw_prediction
        )


        # ====================================================
        # 9. CONVERT LOG PREDICTION TO SECONDS
        #
        # Training target:
        #
        # log_trip_duration =
        # log(1 + trip_duration)
        #
        # Therefore:
        #
        # trip_duration =
        # expm1(prediction)
        # ====================================================

        predicted_seconds = float(
            np.expm1(
                raw_prediction
            )
        )


        # ====================================================
        # 10. SANITY CHECK
        # ====================================================

        if not np.isfinite(
            predicted_seconds
        ):

            raise ValueError(
                "Model returned an invalid prediction."
            )

        if predicted_seconds < 0:

            raise ValueError(
                "Model returned a negative "
                "trip duration."
            )


        predicted_minutes = (
            predicted_seconds / 60.0
        )


        # ====================================================
        # 11. API RESPONSE
        # ====================================================

        return {

            "status": "success",

            "prediction": {

                "trip_duration_seconds": round(
                    predicted_seconds,
                    2
                ),

                "trip_duration_minutes": round(
                    predicted_minutes,
                    2
                )

            },

            "model": {

                "name": "XGBoost",

                "target": "log_trip_duration"

            },

            "input": {

                "pickup_datetime":
                    request.pickup_datetime.isoformat(),

                "passenger_count":
                    request.passenger_count,

                "vendor_id":
                    request.vendor_id,

                "store_and_fwd_flag":
                    store_flag

            }

        }


    # ========================================================
    # HTTP ERRORS
    # ========================================================

    except HTTPException:
        raise


    # ========================================================
    # OTHER ERRORS
    # ========================================================

    except Exception as exc:

        print(
            "\n[API ERROR]",
            str(exc)
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )