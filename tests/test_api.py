from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


# ============================================================
# Test 1: Root endpoint
# ============================================================

def test_root():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "NYC Taxi ETA Prediction API"

    assert data["status"] == "running"


# ============================================================
# Test 2: Health endpoint
# ============================================================

def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

    assert data["model_loaded"] is True


# ============================================================
# Test 3: Model information
# ============================================================

def test_model_info():

    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model_type"] == "XGBoost"

    assert data["target"] == "log_trip_duration"

    assert data["number_of_features"] == 21

    assert len(data["expected_features"]) == 21


# ============================================================
# Test 4: Prediction endpoint
# ============================================================

def test_prediction():

    payload = {

        "pickup_datetime":
            "2016-06-01T08:30:00",

        "pickup_longitude":
            -73.9855,

        "pickup_latitude":
            40.7580,

        "dropoff_longitude":
            -73.9772,

        "dropoff_latitude":
            40.7527,

        "passenger_count":
            1,

        "vendor_id":
            1,

        "store_and_fwd_flag":
            "N"
    }


    response = client.post(
        "/predict",
        json=payload
    )


    assert response.status_code == 200


    data = response.json()


    assert data["status"] == "success"


    assert "prediction" in data


    assert (
        "trip_duration_seconds"
        in data["prediction"]
    )


    assert (
        "trip_duration_minutes"
        in data["prediction"]
    )


# ============================================================
# Test 5: Different passenger count
# ============================================================

def test_prediction_multiple_passengers():

    payload = {

        "pickup_datetime":
            "2016-06-01T18:30:00",

        "pickup_longitude":
            -73.9855,

        "pickup_latitude":
            40.7580,

        "dropoff_longitude":
            -73.8700,

        "dropoff_latitude":
            40.7750,

        "passenger_count":
            4,

        "vendor_id":
            2,

        "store_and_fwd_flag":
            "N"
    }


    response = client.post(
        "/predict",
        json=payload
    )


    assert response.status_code == 200


    data = response.json()


    seconds = (
        data["prediction"]
        ["trip_duration_seconds"]
    )


    assert seconds >= 0


# ============================================================
# Test 6: Invalid passenger count
# ============================================================

def test_invalid_passenger_count():

    payload = {

        "pickup_datetime":
            "2016-06-01T08:30:00",

        "pickup_longitude":
            -73.9855,

        "pickup_latitude":
            40.7580,

        "dropoff_longitude":
            -73.9772,

        "dropoff_latitude":
            40.7527,

        "passenger_count":
            0,

        "vendor_id":
            1,

        "store_and_fwd_flag":
            "N"
    }


    response = client.post(
        "/predict",
        json=payload
    )


    # Pydantic validation should reject this.

    assert response.status_code == 422


# ============================================================
# Test 7: Invalid store/fwd flag
# ============================================================

def test_invalid_store_and_forward_flag():

    payload = {

        "pickup_datetime":
            "2016-06-01T08:30:00",

        "pickup_longitude":
            -73.9855,

        "pickup_latitude":
            40.7580,

        "dropoff_longitude":
            -73.9772,

        "dropoff_latitude":
            40.7527,

        "passenger_count":
            1,

        "vendor_id":
            1,

        "store_and_fwd_flag":
            "ABC"
    }


    response = client.post(
        "/predict",
        json=payload
    )


    assert response.status_code == 400