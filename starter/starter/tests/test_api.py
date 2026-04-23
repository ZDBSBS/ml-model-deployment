import importlib.util
from pathlib import Path
from fastapi.testclient import TestClient

# Dynamically load FastAPI app from starter/main.py
main_path = Path(__file__).resolve().parents[2] / "main.py"
spec = importlib.util.spec_from_file_location("main", main_path)
main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main)

app = main.app
client = TestClient(app)


def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_post_model_prediction_leq_50k():
    payload = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }

    response = client.post("/model", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">50K"]


def test_post_model_prediction_gt_50k():
    payload = {
        "age": 52,
        "workclass": "Self-emp-not-inc",
        "fnlgt": 209642,
        "education": "HS-grad",
        "education_num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States",
    }

    response = client.post("/model", json=payload)
    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">50K"]