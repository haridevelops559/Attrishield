"""
API Route Integration Tests.
"""

def test_health_endpoints(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

    res_ml = client.get("/health/ml")
    assert res_ml.status_code == 200
    assert res_ml.json()["status"] == "healthy"

    res_fs = client.get("/health/feature-store")
    assert res_fs.status_code == 200
    assert res_fs.json()["status"] == "healthy"


def test_auth_login(client):
    res = client.post("/api/v1/auth/login/json", json={"email": "admin@attrishield.com", "password": "AdminPass123!"})
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["user_role"] == "HR_ADMIN"


def test_individual_predict_route(client, auth_headers, sample_employee_raw):
    res = client.post("/api/v1/inference/predict", json=sample_employee_raw, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert "attrition_probability" in data
    assert "risk_recommendation" in data
    assert data["selected_threshold"] == 0.15


def test_feature_store_definitions(client, auth_headers):
    res = client.get("/api/v1/features/definitions", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 7
    feature_names = [f["feature_name"] for f in data]
    assert "IncomePerJobLevel" in feature_names


def test_model_info(client, auth_headers):
    res = client.get("/api/v1/model/info", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["selected_threshold"] == 0.15
    assert data["model_name"] == "XGBoost"
