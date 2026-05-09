import pytest
import os
import sys

# I change into the ml directory first
# so main.py can find models/titanic_model.pkl
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# I add ml/ to path so Python finds main.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


# ── TEST 1 ───────────────────────────────────────
# I test my health check returns 200
def test_health_check_status():
    response = client.get("/")
    assert response.status_code == 200
    print("✓ Health check returns 200")


# ── TEST 2 ───────────────────────────────────────
# I test my health check returns healthy status
def test_health_check_content():
    response = client.get("/")
    data = response.json()
    assert data["status"] == "healthy"
    print("✓ Health check returns healthy status")


# ── TEST 3 ───────────────────────────────────────
# I test a valid prediction — wealthy woman
# in 1st class should SURVIVE
def test_predict_valid_passenger():
    response = client.post("/predict", json={
        "pclass":   1,
        "sex":      "female",
        "age":      25,
        "sibsp":    0,
        "parch":    0,
        "fare":     100.0,
        "embarked": "C"
    })
    assert response.status_code == 200
    data = response.json()
    assert "survived"   in data
    assert "prediction" in data
    assert "confidence" in data
    assert data["prediction"] in ["SURVIVED", "DIED"]
    assert 0 <= data["probability"] <= 1
    print(f"✓ Valid prediction: {data['prediction']}")


# ── TEST 4 ───────────────────────────────────────
# I test that pclass=5 is rejected with 422
def test_predict_invalid_pclass():
    response = client.post("/predict", json={
        "pclass":   5,
        "sex":      "female",
        "age":      25,
        "sibsp":    0,
        "parch":    0,
        "fare":     100.0,
        "embarked": "C"
    })
    assert response.status_code == 422
    print("✓ Invalid pclass=5 correctly rejected (422)")


# ── TEST 5 ───────────────────────────────────────
# I test that sex=alien is rejected with 422
def test_predict_invalid_sex():
    response = client.post("/predict", json={
        "pclass":   1,
        "sex":      "alien",
        "age":      25,
        "sibsp":    0,
        "parch":    0,
        "fare":     100.0,
        "embarked": "C"
    })
    assert response.status_code == 422
    print("✓ Invalid sex=alien correctly rejected (422)")


# ── TEST 6 ───────────────────────────────────────
# I test a poor man in 3rd class — should DIE
def test_predict_poor_man():
    response = client.post("/predict", json={
        "pclass":   3,
        "sex":      "male",
        "age":      30,
        "sibsp":    0,
        "parch":    0,
        "fare":     8.0,
        "embarked": "S"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "DIED"
    print("✓ Poor 3rd class man correctly predicted DIED")


# ── TEST 7 ───────────────────────────────────────
# I test my model-info endpoint
def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type"   in data
    assert "n_estimators" in data
    assert "features"     in data
    assert data["n_estimators"] == 100
    print(f"✓ Model info returns correctly")
