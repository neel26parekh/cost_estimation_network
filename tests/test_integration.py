"""Integration tests: full train → predict → log → drift roundtrip."""

from pathlib import Path

import laptop_price.monitoring as monitoring_module
import laptop_price.predict as predictor
from laptop_price.schemas import PredictionRequest
from laptop_price.train import train_and_save


def test_full_train_predict_log_roundtrip(tmp_path: Path, monkeypatch) -> None:
    """Train a model, predict, log to DB, and read back."""
    model_dir = tmp_path / "model"
    metrics_path = tmp_path / "latest_metrics.json"
    log_path = tmp_path / "predictions.jsonl"
    db_path = tmp_path / "predictions.db"

    metadata = train_and_save(model_dir=model_dir, metrics_path=metrics_path, enable_tuning=False)

    predictor.load_model.cache_clear()
    predictor.load_metadata.cache_clear()
    monkeypatch.setattr(predictor, "MODEL_PATH", model_dir / "model.joblib")
    monkeypatch.setattr(predictor, "METADATA_PATH", model_dir / "metadata.json")

    request = PredictionRequest(
        company="Dell",
        type_name="Notebook",
        ram=8,
        weight=2.1,
        touchscreen=False,
        ips=True,
        screen_size=15.6,
        screen_resolution="1920x1080",
        cpu_brand="Intel Core i5",
        hdd=1000,
        ssd=256,
        gpu_brand="Nvidia",
        os="Windows",
    )

    predicted_price = predictor.predict_price(request)
    assert predicted_price > 0

    # Log to custom paths
    monkeypatch.setattr(monitoring_module, "PREDICTION_LOG_PATH", log_path)
    monkeypatch.setattr(monitoring_module, "PREDICTION_DB_PATH", db_path)

    monitoring_module.append_prediction_log({
        "request_id": "integration-test-001",
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "predicted_price_inr": round(predicted_price, 2),
        "latency_ms": 5.0,
        "features": request.model_dump(),
    })

    # Read back from DB
    recent = monitoring_module.read_recent_prediction_logs(limit=5)
    assert len(recent) == 1
    assert recent[0]["request_id"] == "integration-test-001"
    assert recent[0]["predicted_price_inr"] == round(predicted_price, 2)

    # Verify JSONL also written
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1

    # Metadata has cv_scores (new feature)
    assert "cv_scores" in metadata
    assert metadata["model_name"] in metadata["cv_scores"]


def test_training_metadata_includes_feature_importances(tmp_path: Path) -> None:
    """Training with tree models should produce feature importances."""
    model_dir = tmp_path / "model"
    metrics_path = tmp_path / "metrics.json"
    metadata = train_and_save(model_dir=model_dir, metrics_path=metrics_path, enable_tuning=False)

    # Tree models (RF, GBR) should have feature importances
    if metadata["model_name"] in ("random_forest", "gradient_boosting"):
        assert "feature_importances" in metadata
        assert isinstance(metadata["feature_importances"], dict)
        assert len(metadata["feature_importances"]) > 0
