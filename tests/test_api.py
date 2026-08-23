from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

import src.api.main as api_module
from src.api.main import app


client = TestClient(app)


class FakeModel:
    def __init__(self, probability: float):
        self.probability = probability

    def predict_proba(self, row):
        return np.array(
            [[1.0 - self.probability, self.probability]]
        )


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "ok"
    assert "model_available" in payload
    assert "threshold_available" in payload


def test_invalid_machine_type_is_rejected():
    response = client.post(
        "/predict",
        json={
            "type": "X",
            "air_temperature": 298.1,
            "process_temperature": 308.6,
            "rotational_speed": 1551,
            "torque": 42.8,
            "tool_wear": 0,
        },
    )

    assert response.status_code == 422


def test_invalid_rotational_speed_is_rejected():
    response = client.post(
        "/predict",
        json={
            "type": "L",
            "air_temperature": 298.1,
            "process_temperature": 308.6,
            "rotational_speed": 0,
            "torque": 42.8,
            "tool_wear": 0,
        },
    )

    assert response.status_code == 422


def test_successful_prediction_above_threshold(
    monkeypatch,
):
    monkeypatch.setattr(
        api_module,
        "load_model",
        lambda: FakeModel(0.80),
    )

    monkeypatch.setattr(
        api_module,
        "load_threshold",
        lambda: 0.10,
    )

    monkeypatch.setattr(
        api_module,
        "explain_prediction",
        lambda row: [
            {
                "feature": "Torque",
                "shap_value": 0.42,
            }
        ],
    )

    response = client.post(
        "/predict",
        json={
            "type": "L",
            "air_temperature": 298.1,
            "process_temperature": 308.6,
            "rotational_speed": 1551,
            "torque": 42.8,
            "tool_wear": 0,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["failure_probability"] == 0.8
    assert payload["predicted_failure"] is True
    assert payload["threshold"] == 0.1

    assert len(payload["explanation"]) == 1
    assert payload["explanation"][0]["feature"] == "Torque"


def test_prediction_below_threshold(
    monkeypatch,
):
    monkeypatch.setattr(
        api_module,
        "load_model",
        lambda: FakeModel(0.05),
    )

    monkeypatch.setattr(
        api_module,
        "load_threshold",
        lambda: 0.10,
    )

    monkeypatch.setattr(
        api_module,
        "explain_prediction",
        lambda row: [],
    )

    response = client.post(
        "/predict",
        json={
            "type": "H",
            "air_temperature": 300.0,
            "process_temperature": 310.0,
            "rotational_speed": 1400,
            "torque": 30.0,
            "tool_wear": 100,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["failure_probability"] == 0.05
    assert payload["predicted_failure"] is False
    assert payload["threshold"] == 0.1


def test_drift_monitor_endpoint():
    response = client.get("/monitor/drift")

    assert response.status_code == 200

    payload = response.json()

    assert "drift_detected" in payload
    assert "drifted_features" in payload
    assert "features" in payload

    assert isinstance(
        payload["drifted_features"],
        list,
    )