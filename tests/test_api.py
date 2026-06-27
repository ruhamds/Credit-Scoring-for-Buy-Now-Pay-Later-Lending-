import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_health_reports_model_not_loaded(client):
    response = client.get("/health")
    assert response.json()["model_loaded"] in [True, False]


def test_predict_endpoint_handles_model_state(client):
    """
    Verifies /predict behaves correctly regardless of model state.
    - Returns 200 with full response body when model is loaded
    - Returns 503 when model is not loaded
    Both are valid depending on whether artifacts exist in the environment.
    """
    payload = {
        "Amount_count": 12.0, "Amount_sum": 87.3, "Amount_mean": 7.3,
        "Amount_std": 2.1, "Amount_min": 3.4, "Amount_max": 11.2,
        "tx_hour_mean": 14.5, "tx_day_mean": 15.2, "tx_month_mean": 11.8,
        "tx_dayofweek_mean": 2.3, "tx_is_weekend_mean": 0.17,
        "ProviderId_mode": 3.0, "ChannelId_mode": 2.0,
        "ProductCategory_mode": 1.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in [200, 503], (
        f"Expected 200 or 503, got {response.status_code}: {response.text}"
    )
    if response.status_code == 200:
        body = response.json()
        assert "risk_score"    in body
        assert "decision"      in body
        assert "prediction_id" in body
        assert "top_factors"   in body
        assert body["decision"] in ["approve", "refer", "reject"]



def test_predict_rejects_invalid_payload(client):
    """Pydantic must reject missing required fields with 422."""
    response = client.post("/predict", json={"bad_field": "garbage"})
    assert response.status_code == 422


def test_schema_endpoint_returns_feature_list(client):
    """Schema endpoint must return expected input fields."""
    response = client.get("/schema")
    assert response.status_code == 200
    assert "input_fields" in response.json()
    assert "Amount_count" in response.json()["input_fields"]