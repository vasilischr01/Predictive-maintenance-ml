from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


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

def test_drift_monitor_endpoint():
    response = client.get("/monitor/drift")

    assert response.status_code == 200

    payload = response.json()

    assert "drift_detected" in payload
    assert "drifted_features" in payload
    assert "features" in payload
    assert isinstance(payload["drifted_features"], list)
