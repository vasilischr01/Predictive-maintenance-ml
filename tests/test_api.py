from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

import src.api.main as api_module
from src.api.main import app

client = TestClient(app)


class FakeModel:
    def __init__(
        self,
        probability: float,
    ):
        self.probability = probability

    def predict_proba(
        self,
        row,
    ):
        return np.array(
            [
                [
                    1.0 - self.probability,
                    self.probability,
                ]
            ]
        )


class BrokenModel:
    def predict_proba(
        self,
        row,
    ):
        raise RuntimeError(
            "private prediction internals"
        )


def valid_payload() -> dict:
    return {
        "type": "L",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551,
        "torque": 42.8,
        "tool_wear": 0,
    }


def test_health_endpoint():
    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "ok"
    assert (
        "model_available"
        in payload
    )
    assert (
        "threshold_available"
        in payload
    )


def test_invalid_machine_type_is_rejected():
    payload = valid_payload()
    payload["type"] = "X"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_invalid_rotational_speed_is_rejected():
    payload = valid_payload()
    payload["rotational_speed"] = 0

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_successful_prediction_above_threshold(
    monkeypatch,
):
    monkeypatch.setattr(
        api_module,
        "load_model",
        lambda: FakeModel(
            0.80
        ),
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
        json=valid_payload(),
    )

    assert response.status_code == 200

    payload = response.json()

    assert (
        payload[
            "failure_probability"
        ]
        == 0.8
    )

    assert (
        payload[
            "predicted_failure"
        ]
        is True
    )

    assert (
        payload["threshold"]
        == 0.1
    )

    assert (
        len(
            payload[
                "explanation"
            ]
        )
        == 1
    )

    assert (
        payload[
            "explanation"
        ][0]["feature"]
        == "Torque"
    )


def test_prediction_below_threshold(
    monkeypatch,
):
    monkeypatch.setattr(
        api_module,
        "load_model",
        lambda: FakeModel(
            0.05
        ),
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

    payload = valid_payload()
    payload["type"] = "H"
    payload[
        "air_temperature"
    ] = 300.0
    payload[
        "process_temperature"
    ] = 310.0
    payload[
        "rotational_speed"
    ] = 1400
    payload["torque"] = 30.0
    payload["tool_wear"] = 100

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    result = response.json()

    assert (
        result[
            "failure_probability"
        ]
        == 0.05
    )

    assert (
        result[
            "predicted_failure"
        ]
        is False
    )

    assert (
        result["threshold"]
        == 0.1
    )


def test_drift_monitor_endpoint():
    response = client.get(
        "/monitor/drift"
    )

    assert response.status_code == 200

    payload = response.json()

    assert (
        "drift_detected"
        in payload
    )

    assert (
        "drifted_features"
        in payload
    )

    assert (
        "features"
        in payload
    )

    assert isinstance(
        payload[
            "drifted_features"
        ],
        list,
    )


def test_security_headers_are_present():
    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    assert (
        response.headers[
            "X-Content-Type-Options"
        ]
        == "nosniff"
    )

    assert (
        response.headers[
            "X-Frame-Options"
        ]
        == "DENY"
    )

    assert (
        response.headers[
            "Referrer-Policy"
        ]
        == "no-referrer"
    )

    assert (
        response.headers[
            "Cache-Control"
        ]
        == "no-store"
    )

    assert (
        "camera=()"
        in response.headers[
            "Permissions-Policy"
        ]
    )


def test_oversized_request_is_rejected():
    oversized_payload = {
        "type": "L",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551,
        "torque": 42.8,
        "tool_wear": 0,
        "padding": (
            "x"
            * (
                api_module
                .MAX_REQUEST_BYTES
                + 1024
            )
        ),
    }

    response = client.post(
        "/predict",
        json=oversized_payload,
    )

    assert response.status_code == 413

    assert response.json() == {
        "detail": (
            "Request body too large"
        ),
    }


def test_rate_limit_is_enforced():
    api_module._request_history.clear()

    try:
        last_response = None

        for _ in range(
            api_module
            .RATE_LIMIT_REQUESTS
            + 1
        ):
            last_response = (
                client.get(
                    "/health"
                )
            )

        assert (
            last_response
            is not None
        )

        assert (
            last_response
            .status_code
            == 429
        )

        assert (
            last_response.json()
            == {
                "detail": (
                    "Too many requests"
                ),
            }
        )

        assert (
            last_response.headers[
                "Retry-After"
            ]
            == str(
                api_module
                .RATE_LIMIT_WINDOW_SECONDS
            )
        )

    finally:
        api_module._request_history.clear()


def test_missing_model_artifacts_are_sanitized(
    monkeypatch,
):
    secret_message = (
        "C:/private/model/path/"
        "model.joblib"
    )

    def fail_model():
        raise FileNotFoundError(
            secret_message
        )

    monkeypatch.setattr(
        api_module,
        "load_model",
        fail_model,
    )

    response = client.post(
        "/predict",
        json=valid_payload(),
    )

    assert response.status_code == 503

    assert response.json() == {
        "detail": (
            "Model artifacts are "
            "unavailable."
        ),
    }

    assert (
        secret_message
        not in response.text
    )


def test_prediction_failure_is_sanitized(
    monkeypatch,
):
    monkeypatch.setattr(
        api_module,
        "load_model",
        lambda: BrokenModel(),
    )

    monkeypatch.setattr(
        api_module,
        "load_threshold",
        lambda: 0.10,
    )

    response = client.post(
        "/predict",
        json=valid_payload(),
    )

    assert response.status_code == 500

    assert response.json() == {
        "detail": (
            "Prediction failed."
        ),
    }

    assert (
        "private prediction internals"
        not in response.text
    )


def test_shap_failure_is_sanitized(
    monkeypatch,
):
    secret_message = (
        "private SHAP stack trace"
    )

    monkeypatch.setattr(
        api_module,
        "load_model",
        lambda: FakeModel(
            0.80
        ),
    )

    monkeypatch.setattr(
        api_module,
        "load_threshold",
        lambda: 0.10,
    )

    def fail_explanation(
        row,
    ):
        raise RuntimeError(
            secret_message
        )

    monkeypatch.setattr(
        api_module,
        "explain_prediction",
        fail_explanation,
    )

    response = client.post(
        "/predict",
        json=valid_payload(),
    )

    assert response.status_code == 500

    assert response.json() == {
        "detail": (
            "Prediction explanation "
            "failed."
        ),
    }

    assert (
        secret_message
        not in response.text
    )


def test_drift_missing_artifact_is_sanitized(
    monkeypatch,
):
    secret_message = (
        "C:/private/reference.csv"
    )

    def fail_drift():
        raise FileNotFoundError(
            secret_message
        )

    monkeypatch.setattr(
        api_module,
        "run_drift_analysis",
        fail_drift,
    )

    response = client.get(
        "/monitor/drift"
    )

    assert response.status_code == 503

    assert response.json() == {
        "detail": (
            "Drift analysis artifacts "
            "are unavailable."
        ),
    }

    assert (
        secret_message
        not in response.text
    )


def test_drift_internal_error_is_sanitized(
    monkeypatch,
):
    secret_message = (
        "private drift traceback"
    )

    def fail_drift():
        raise RuntimeError(
            secret_message
        )

    monkeypatch.setattr(
        api_module,
        "run_drift_analysis",
        fail_drift,
    )

    response = client.get(
        "/monitor/drift"
    )

    assert response.status_code == 500

    assert response.json() == {
        "detail": (
            "Drift analysis failed."
        ),
    }

    assert (
        secret_message
        not in response.text
    )