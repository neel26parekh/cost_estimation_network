"""Tests for the prediction monitoring and SQLite persistence layer."""

import json

import laptop_price.monitoring as monitoring_module


def test_sqlite_write_and_read_roundtrip(tmp_path, monkeypatch) -> None:
    """Write a prediction log to SQLite and read it back."""
    db_path = tmp_path / "test_predictions.db"
    log_path = tmp_path / "test_predictions.jsonl"

    monkeypatch.setattr(monitoring_module, "PREDICTION_DB_PATH", db_path)
    monkeypatch.setattr(monitoring_module, "PREDICTION_LOG_PATH", log_path)

    monitoring_module.append_prediction_log({
        "request_id": "test-req-001",
        "model_name": "gradient_boosting",
        "model_version": "v1",
        "predicted_price_inr": 45000.0,
        "latency_ms": 8.5,
        "features": {"company": "HP", "ram": 16},
    })

    monitoring_module.append_prediction_log({
        "request_id": "test-req-002",
        "model_name": "gradient_boosting",
        "model_version": "v1",
        "predicted_price_inr": 62000.0,
        "latency_ms": 7.2,
        "features": {"company": "Dell", "ram": 32},
    })

    recent = monitoring_module.read_recent_prediction_logs(limit=10)
    assert len(recent) == 2
    # Most recent first
    assert recent[0]["request_id"] == "test-req-002"
    assert recent[1]["request_id"] == "test-req-001"


def test_jsonl_fallback_when_no_db(tmp_path, monkeypatch) -> None:
    """When there is no SQLite DB, fall back to JSONL file."""
    log_path = tmp_path / "test_predictions.jsonl"
    nonexistent_db = tmp_path / "nonexistent.db"

    monkeypatch.setattr(monitoring_module, "PREDICTION_DB_PATH", nonexistent_db)
    monkeypatch.setattr(monitoring_module, "PREDICTION_LOG_PATH", log_path)

    # Write directly to JSONL (simulating a scenario where the DB doesn't exist yet for reads)
    log_path.write_text(
        json.dumps({
            "logged_at_utc": "2026-01-01T00:00:00+00:00",
            "request_id": "jsonl-001",
            "model_name": "ridge",
            "model_version": "v1",
            "predicted_price_inr": 30000.0,
            "latency_ms": 3.0,
            "features": {"company": "Lenovo"},
        }) + "\n",
        encoding="utf-8",
    )

    recent = monitoring_module.read_recent_prediction_logs(limit=5)
    assert len(recent) == 1
    assert recent[0]["request_id"] == "jsonl-001"


def test_summarize_prediction_logs(tmp_path, monkeypatch) -> None:
    """Summarize returns correct counts and average latency."""
    db_path = tmp_path / "summary_test.db"
    log_path = tmp_path / "summary_test.jsonl"

    monkeypatch.setattr(monitoring_module, "PREDICTION_DB_PATH", db_path)
    monkeypatch.setattr(monitoring_module, "PREDICTION_LOG_PATH", log_path)

    for i in range(3):
        monitoring_module.append_prediction_log({
            "request_id": f"sum-{i}",
            "model_name": "gradient_boosting",
            "model_version": "v1",
            "predicted_price_inr": 50000.0 + i * 1000,
            "latency_ms": 10.0 + i,
            "features": {"company": "Dell"},
        })

    summary = monitoring_module.summarize_prediction_logs()
    assert summary["prediction_count"] == 3
    assert summary["latest_request_id"] == "sum-2"
    assert summary["average_latency_ms"] == round((10.0 + 11.0 + 12.0) / 3, 2)
