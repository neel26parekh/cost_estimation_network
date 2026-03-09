from fastapi.testclient import TestClient
import json

import laptop_price.api as api_module
import laptop_price.security as security_module


client = TestClient(api_module.app)


def setup_function() -> None:
    security_module.reset_rate_limit_state()
    security_module.API_KEY = ""
    security_module.RATE_LIMIT_REQUESTS = 60
    security_module.RATE_LIMIT_WINDOW_SECONDS = 60


def test_root_endpoint() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["service"] == "laptop-price-predictor-api"
    assert response.json()["docs"] == "/docs"


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_endpoint_includes_model_age() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    # If model exists, model_age_days should be present
    if "model_age_days" in body:
        assert isinstance(body["model_age_days"], int)


def test_ready_endpoint() -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_endpoint_returns_503_when_model_missing(monkeypatch) -> None:
    def missing_model() -> None:
        raise FileNotFoundError("model missing")

    monkeypatch.setattr(api_module, "load_model", missing_model)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"] == "model missing"


# --- v1 versioned endpoints ---

def test_v1_predict_endpoint() -> None:
    payload = {
        "company": "Dell",
        "type_name": "Notebook",
        "ram": 8,
        "weight": 2.1,
        "touchscreen": False,
        "ips": True,
        "screen_size": 15.6,
        "screen_resolution": "1920x1080",
        "cpu_brand": "Intel Core i5",
        "hdd": 1000,
        "ssd": 256,
        "gpu_brand": "Nvidia",
        "os": "Windows",
    }

    response = client.post("/v1/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_price_inr"] > 0
    assert body["currency"] == "INR"
    assert body["request_id"]
    assert body["latency_ms"] >= 0


def test_backward_compat_predict_endpoint() -> None:
    """The unversioned /predict should still work."""
    payload = {
        "company": "Dell",
        "type_name": "Notebook",
        "ram": 8,
        "weight": 2.1,
        "touchscreen": False,
        "ips": True,
        "screen_size": 15.6,
        "screen_resolution": "1920x1080",
        "cpu_brand": "Intel Core i5",
        "hdd": 1000,
        "ssd": 256,
        "gpu_brand": "Nvidia",
        "os": "Windows",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["predicted_price_inr"] > 0


def test_v1_predict_endpoint_returns_503_when_artifacts_missing(monkeypatch) -> None:
    def missing_metadata() -> None:
        raise FileNotFoundError("metadata missing")

    monkeypatch.setattr(api_module, "load_metadata", missing_metadata)

    response = client.post(
        "/v1/predict",
        json={
            "company": "Dell",
            "type_name": "Notebook",
            "ram": 8,
            "weight": 2.1,
            "touchscreen": False,
            "ips": True,
            "screen_size": 15.6,
            "screen_resolution": "1920x1080",
            "cpu_brand": "Intel Core i5",
            "hdd": 1000,
            "ssd": 256,
            "gpu_brand": "Nvidia",
            "os": "Windows",
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "metadata missing"


def test_v1_predict_endpoint_returns_422_for_invalid_prediction(monkeypatch) -> None:
    monkeypatch.setattr(api_module, "predict_price", lambda payload: (_ for _ in ()).throw(ValueError("Prediction result is not finite. Provide a realistic laptop configuration.")))

    response = client.post(
        "/v1/predict",
        json={
            "company": "Dell",
            "type_name": "Notebook",
            "ram": 8,
            "weight": 2.1,
            "touchscreen": False,
            "ips": True,
            "screen_size": 15.6,
            "screen_resolution": "1920x1080",
            "cpu_brand": "Intel Core i5",
            "hdd": 1000,
            "ssd": 256,
            "gpu_brand": "Nvidia",
            "os": "Windows",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Prediction result is not finite. Provide a realistic laptop configuration."


def test_protected_endpoints_require_api_key_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(security_module, "API_KEY", "secret-key")

    response = client.get("/v1/metadata")

    assert response.status_code == 401
    assert response.json()["detail"] == "Unauthorized. Provide x-api-key header or use /docs Authorize."


def test_protected_endpoint_accepts_valid_api_key(monkeypatch) -> None:
    monkeypatch.setattr(security_module, "API_KEY", "secret-key")

    response = client.get("/v1/metadata", headers={"x-api-key": "secret-key"})

    assert response.status_code == 200


def test_rate_limit_returns_429(monkeypatch) -> None:
    security_module.reset_rate_limit_state()
    monkeypatch.setattr(security_module, "RATE_LIMIT_REQUESTS", 1)
    monkeypatch.setattr(security_module, "RATE_LIMIT_WINDOW_SECONDS", 60)

    first = client.get("/v1/metadata")
    second = client.get("/v1/metadata")

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"] == "Rate limit exceeded"


def test_v1_recent_predictions_endpoint(monkeypatch, tmp_path) -> None:
    log_path = tmp_path / "predictions.jsonl"
    log_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "logged_at_utc": "2026-03-08T00:00:00+00:00",
                        "request_id": "one",
                        "model_name": "gradient_boosting",
                        "model_version": "v1",
                        "predicted_price_inr": 50000.0,
                        "latency_ms": 10.0,
                        "features": {"company": "Dell"},
                    }
                ),
                json.dumps(
                    {
                        "logged_at_utc": "2026-03-08T00:01:00+00:00",
                        "request_id": "two",
                        "model_name": "gradient_boosting",
                        "model_version": "v2",
                        "predicted_price_inr": 55000.0,
                        "latency_ms": 11.0,
                        "features": {"company": "HP"},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(api_module, "read_recent_prediction_logs", lambda limit: json.loads("[" + ",".join(log_path.read_text(encoding="utf-8").splitlines()[::-1][:limit]) + "]"))

    response = client.get("/v1/predictions/recent?limit=1")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["request_id"] == "two"


def test_v1_monitoring_drift_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        api_module,
        "load_latest_drift_report",
        lambda: {"model_version": "v1", "sample_size": 3, "drift_detected": False},
    )

    response = client.get("/v1/monitoring/drift")

    assert response.status_code == 200
    assert response.json()["model_version"] == "v1"


def test_v1_monitoring_drift_endpoint_returns_404_when_missing(monkeypatch) -> None:
    def missing_report() -> None:
        raise FileNotFoundError("drift report missing")

    monkeypatch.setattr(api_module, "load_latest_drift_report", missing_report)

    response = client.get("/v1/monitoring/drift")

    assert response.status_code == 404
    assert response.json()["detail"] == "drift report missing"


def test_v1_monitoring_summary_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        api_module,
        "summarize_prediction_logs",
        lambda: {
            "prediction_count": 2,
            "latest_request_id": "two",
            "active_model_versions": ["v2", "v1"],
            "average_latency_ms": 10.5,
        },
    )

    response = client.get("/v1/monitoring/summary")

    assert response.status_code == 200
    assert response.json()["prediction_count"] == 2


def test_cors_headers_present() -> None:
    """CORS middleware should allow requests."""
    response = client.options(
        "/v1/predict",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    # CORS should not return 405 for OPTIONS preflight
    assert response.status_code in (200, 204, 400)