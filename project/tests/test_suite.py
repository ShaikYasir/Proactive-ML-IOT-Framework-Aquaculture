# tests/test_data_processing.py
import pytest
import pandas as pd
from pathlib import Path

from data_processing import load_dataset, preprocess, split_data

@pytest.fixture
def sample_csv(tmp_path):
    data = {
        "feature1": [1, 2, 3, 4],
        "feature2": ["A", "B", "A", "B"],
        "disease_risk": ["Low", "Moderate", "High", "Low"],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

def test_load_dataset(sample_csv):
    df = load_dataset(str(sample_csv))
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 4

def test_preprocess(sample_csv):
    df = load_dataset(str(sample_csv))
    cfg = {"training": {}}
    X, y, encoder, scaler = preprocess(df, cfg, training=True)
    assert X.shape[0] == 4
    assert len(y) == 4
    # Categorical column should be encoded to numeric
    assert X["feature2"].dtype.kind in "iuf"

def test_split_data(sample_csv):
    df = load_dataset(str(sample_csv))
    cfg = {"training": {"test_size": 0.5, "random_state": 42}}
    X, y, _, _ = preprocess(df, cfg, training=True)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.5, random_state=42)
    assert len(X_train) == len(y_train) == 2
    assert len(X_test) == len(y_test) == 2

# tests/test_model_training.py
import pytest
import pandas as pd
from pathlib import Path

from model_training import train_models

@pytest.fixture
def sample_csv(tmp_path):
    data = {
        "feature1": [1, 2, 3, 4, 5, 6],
        "feature2": ["A", "B", "A", "B", "A", "B"],
        "disease_risk": ["Low", "Low", "Moderate", "High", "High", "Low"],
    }
    df = pd.DataFrame(data)
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

def test_train_models(sample_csv, tmp_path, monkeypatch):
    # Redirect model saving to a temporary directory
    monkeypatch.setattr('pathlib.Path.mkdir', lambda self, parents=False, exist_ok=False: None)
    best_model, results = train_models(str(sample_csv), config="../config.yaml")
    assert best_model in {"logistic", "random_forest", "lightgbm"}
    assert isinstance(results, dict)
    for name, metrics in results.items():
        assert "recall" in metrics
        assert "precision" in metrics

# tests/test_api.py
import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)

def test_predict_endpoint():
    payload = {
        "pond_data": {
            "feature1": 2.5,
            "feature2": "A"
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "disease_risk" in data
    assert "confidence" in data
    assert "key_risk_factors" in data
    assert "recommended_actions" in data
