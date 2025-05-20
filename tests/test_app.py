"""HTTP contract tests for the FastAPI adapter (requires the ``live`` extra)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app, get_service  # noqa: E402

client = TestClient(app)


def setup_function() -> None:
    get_service.cache_clear()


def test_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "onnx-rust-wasm-triton-gateway"


def test_health_and_model() -> None:
    assert client.get("/health").json()["status"] == "ok"
    model = client.get("/model").json()
    assert model["model"] == "onnx_wasm_model"
    assert client.get("/ready").json()["status"] == "ready"


def test_predict_success() -> None:
    response = client.post(
        "/predict", json={"input_data": [0.1, 0.2, 0.3, 0.4], "input_shape": [4]}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert len(body["output"]) == 4


def test_predict_validation_error() -> None:
    response = client.post("/predict", json={"input_shape": [2]})
    assert response.status_code == 400
    assert response.json()["success"] is False


def test_predict_payload_too_large() -> None:
    response = client.post(
        "/predict",
        json={"input_data": [0.0] * 20, "input_shape": [20]},
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 200  # under the default 1M limit


def test_metrics_exposition() -> None:
    client.post("/predict", json={"input_data": [1.0], "input_shape": [1]})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "inference_requests_total" in response.text
    assert response.headers["content-type"].startswith("text/plain")
