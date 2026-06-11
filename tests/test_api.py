from fastapi import status


def test_root_endpoint_returns_404(client):
    response = client.get("/")
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_predict_endpoint_requires_model(client):
    response = client.post("/predict", json={"customer_id": 1, "credit_amount": 1000, "term_months": 12, "annual_income": 50000})
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
