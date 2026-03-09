from pathlib import Path
import json

import laptop_price.predict as predictor
from laptop_price.schemas import PredictionRequest
import laptop_price.train as train_module
from laptop_price.train import activate_model_version, candidate_beats_production, load_registry_index, train_and_save


def test_training_and_prediction_round_trip(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    metrics_path = tmp_path / "latest_metrics.json"
    metadata = train_and_save(model_dir=model_dir, metrics_path=metrics_path)

    predictor.load_model.cache_clear()
    predictor.load_metadata.cache_clear()
    predictor.MODEL_PATH = model_dir / "model.joblib"
    predictor.METADATA_PATH = model_dir / "metadata.json"

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

    prediction = predictor.predict_price(request)

    assert prediction > 0
    assert metadata["model_name"]
    assert metrics_path.exists()


def test_training_registers_versions_and_can_activate_them(tmp_path: Path, monkeypatch) -> None:
    registry_dir = tmp_path / "registry"
    production_dir = tmp_path / "production"
    metrics_path = tmp_path / "metrics.json"
    index_path = registry_dir / "index.json"

    monkeypatch.setattr(train_module, "MODEL_REGISTRY_DIR", registry_dir)
    monkeypatch.setattr(train_module, "MODEL_DIR", production_dir)
    monkeypatch.setattr(train_module, "REGISTRY_INDEX_PATH", index_path)
    monkeypatch.setattr(train_module, "LATEST_METRICS_PATH", metrics_path)

    first_metadata = train_and_save(model_dir=production_dir, metrics_path=metrics_path)
    first_index = load_registry_index()

    assert first_index["active_version"] == first_metadata["model_version"]
    assert len(first_index["versions"]) == 1
    assert (registry_dir / first_metadata["model_version"] / "model.joblib").exists()

    second_metadata = train_and_save(model_dir=production_dir, metrics_path=metrics_path)
    second_index = load_registry_index()

    assert second_index["active_version"] == second_metadata["model_version"]
    assert len(second_index["versions"]) == 2

    activated_metadata = activate_model_version(first_metadata["model_version"])
    active_metadata = json.loads((production_dir / "metadata.json").read_text(encoding="utf-8"))
    final_index = load_registry_index()

    assert activated_metadata["model_version"] == first_metadata["model_version"]
    assert active_metadata["model_version"] == first_metadata["model_version"]
    assert final_index["active_version"] == first_metadata["model_version"]


def test_candidate_beats_production_uses_r2_then_rmse() -> None:
    production_metadata = {"metrics": {"r2": 0.80, "rmse": 100.0}}

    assert candidate_beats_production({"metrics": {"r2": 0.81, "rmse": 200.0}}, production_metadata) is True
    assert candidate_beats_production({"metrics": {"r2": 0.79, "rmse": 50.0}}, production_metadata) is False
    assert candidate_beats_production({"metrics": {"r2": 0.80, "rmse": 90.0}}, production_metadata) is True
    assert candidate_beats_production({"metrics": {"r2": 0.80, "rmse": 110.0}}, production_metadata) is False
