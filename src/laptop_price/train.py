from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import (
    CATEGORICAL_COLUMNS,
    LATEST_METRICS_PATH,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    MODEL_REGISTRY_DIR,
    NUMERIC_COLUMNS,
    RANDOM_STATE,
    REGISTRY_INDEX_PATH,
    ensure_directories,
    resolve_raw_data_path,
)
from .features import build_modeling_dataframe, build_training_matrices, extract_ui_options, load_raw_dataset
from .logger import get_logger
from .performance_history import append_training_run
from .validation import validate_training_data

logger = get_logger(__name__)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    if not path.exists():
        return default.copy() if default is not None else {}
    return json.loads(path.read_text(encoding="utf-8"))


def clear_prediction_caches() -> None:
    from . import predict as predictor

    predictor.load_model.cache_clear()
    predictor.load_metadata.cache_clear()


def load_registry_index() -> dict[str, Any]:
    return read_json(REGISTRY_INDEX_PATH, default={"active_version": None, "versions": []})


def build_registry_entry(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_version": metadata["model_version"],
        "model_name": metadata["model_name"],
        "trained_at_utc": metadata["trained_at_utc"],
        "metrics": metadata["metrics"],
        "artifact_dir": str((MODEL_REGISTRY_DIR / metadata["model_version"]).resolve()),
    }


def update_registry_index(entry: dict[str, Any], make_active: bool = True) -> dict[str, Any]:
    index = load_registry_index()
    versions = [item for item in index["versions"] if item["model_version"] != entry["model_version"]]
    versions.append(entry)
    versions.sort(key=lambda item: item["model_version"], reverse=True)
    index["versions"] = versions
    if make_active:
        index["active_version"] = entry["model_version"]
    write_json(REGISTRY_INDEX_PATH, index)
    return index


def set_active_registry_version(model_version: str | None) -> dict[str, Any]:
    index = load_registry_index()
    index["active_version"] = model_version
    write_json(REGISTRY_INDEX_PATH, index)
    return index


def register_model_version(metadata: dict[str, Any], source_model_dir: Path, make_active: bool = False) -> Path:
    version_dir = MODEL_REGISTRY_DIR / metadata["model_version"]
    version_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_model_dir / MODEL_PATH.name, version_dir / MODEL_PATH.name)
    shutil.copy2(source_model_dir / METADATA_PATH.name, version_dir / METADATA_PATH.name)

    update_registry_index(build_registry_entry(metadata), make_active=make_active)
    return version_dir


def list_registered_versions() -> list[dict[str, Any]]:
    return load_registry_index()["versions"]


def activate_model_version(model_version: str) -> dict[str, Any]:
    version_dir = MODEL_REGISTRY_DIR / model_version
    model_artifact = version_dir / MODEL_PATH.name
    metadata_artifact = version_dir / METADATA_PATH.name

    if not model_artifact.exists() or not metadata_artifact.exists():
        raise FileNotFoundError(f"Model version {model_version} was not found in the local registry.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_artifact, MODEL_DIR / MODEL_PATH.name)
    shutil.copy2(metadata_artifact, MODEL_DIR / METADATA_PATH.name)

    metadata = read_json(metadata_artifact)
    update_registry_index(build_registry_entry(metadata), make_active=True)
    clear_prediction_caches()
    logger.info("Activated model version %s", model_version)
    return metadata


def candidate_beats_production(candidate_metadata: dict[str, Any], production_metadata: dict[str, Any] | None) -> bool:
    if production_metadata is None:
        return True

    candidate_metrics = candidate_metadata["metrics"]
    production_metrics = production_metadata["metrics"]

    candidate_r2 = candidate_metrics["r2"]
    production_r2 = production_metrics["r2"]
    if candidate_r2 > production_r2:
        return True
    if candidate_r2 < production_r2:
        return False

    return candidate_metrics["rmse"] < production_metrics["rmse"]


def load_production_metadata() -> dict[str, Any] | None:
    if not METADATA_PATH.exists():
        return None
    return read_json(METADATA_PATH)


def write_training_outputs(
    *,
    model_dir: Path,
    final_pipeline,
    metadata: dict[str, Any],
    metrics_path: Path,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_dir / MODEL_PATH.name)
    write_json(model_dir / METADATA_PATH.name, metadata)
    write_json(metrics_path, metadata["metrics"])


def create_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", create_one_hot_encoder()),
        ]
    )
    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ]
    )


def build_candidate_models() -> dict[str, object]:
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=10.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


RF_PARAM_GRID = {
    "model__n_estimators": [200, 300],
    "model__max_depth": [14, 18, 22],
    "model__min_samples_leaf": [1, 2],
}

GBR_PARAM_GRID = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 5],
    "model__learning_rate": [0.05, 0.1],
}

TUNABLE_PARAM_GRIDS = {
    "random_forest": RF_PARAM_GRID,
    "gradient_boosting": GBR_PARAM_GRID,
}


def evaluate_predictions(y_true_log, y_pred_log) -> dict[str, float]:
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def extract_feature_importances(pipeline) -> dict[str, float] | None:
    """Extract feature importances from tree-based models in the pipeline."""
    model = pipeline.named_steps.get("model")
    if not hasattr(model, "feature_importances_"):
        return None
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        feature_names = preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        return None
    importances = model.feature_importances_.tolist()
    return dict(sorted(zip(feature_names, importances, strict=False), key=lambda x: x[1], reverse=True))


def select_best_model(X_train, X_test, y_train, y_test, candidate_models=None, enable_tuning: bool = True):
    candidate_models = candidate_models or build_candidate_models()
    best_name = None
    best_pipeline = None
    best_metrics = None
    all_metrics = {}
    cv_scores_by_model = {}

    for model_name, estimator in candidate_models.items():
        pipeline = Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", estimator)])

        # Cross-validation for reliable comparison
        cv_r2 = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
        cv_scores_by_model[model_name] = {
            "mean_r2": float(cv_r2.mean()),
            "std_r2": float(cv_r2.std()),
            "folds": cv_r2.tolist(),
        }
        logger.info("CV results for %s: mean_r2=%.4f ± %.4f", model_name, cv_r2.mean(), cv_r2.std())

        # Hyperparameter tuning for tree models
        if enable_tuning and model_name in TUNABLE_PARAM_GRIDS:
            logger.info("Running GridSearchCV for %s", model_name)
            grid_search = GridSearchCV(
                pipeline,
                TUNABLE_PARAM_GRIDS[model_name],
                cv=3,
                scoring="r2",
                n_jobs=-1,
                refit=True,
            )
            grid_search.fit(X_train, y_train)
            pipeline = grid_search.best_estimator_
            logger.info(
                "Best params for %s: %s (cv_r2=%.4f)",
                model_name,
                grid_search.best_params_,
                grid_search.best_score_,
            )
        else:
            pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        metrics = evaluate_predictions(y_test, predictions)
        all_metrics[model_name] = metrics

        if best_metrics is None or metrics["r2"] > best_metrics["r2"]:
            best_name = model_name
            best_pipeline = pipeline
            best_metrics = metrics

    if best_name is None or best_pipeline is None or best_metrics is None:
        raise RuntimeError("No candidate model was successfully trained.")

    logger.info("Selected model: %s (r2=%.4f, rmse=%.2f)", best_name, best_metrics["r2"], best_metrics["rmse"])
    return best_name, best_pipeline, best_metrics, all_metrics, cv_scores_by_model


def train_and_save(
    raw_data_path: str | None = None,
    model_dir: Path | None = None,
    metrics_path: Path | None = None,
    enable_tuning: bool = True,
):
    ensure_directories()
    logger.info("Starting training pipeline")

    raw_df = load_raw_dataset(raw_data_path)
    modeling_df = build_modeling_dataframe(raw_df)

    # Validate training data
    validate_training_data(modeling_df)
    logger.info("Training data validated: %d rows, %d columns", len(modeling_df), len(modeling_df.columns))

    X, y = build_training_matrices(modeling_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.1,
        random_state=RANDOM_STATE,
    )

    best_name, best_pipeline, best_metrics, all_metrics, cv_scores = select_best_model(
        X_train, X_test, y_train, y_test, enable_tuning=enable_tuning,
    )

    # Retrain winner on full data
    final_pipeline = Pipeline(
        steps=[("preprocessor", build_preprocessor()), ("model", best_pipeline.named_steps["model"])]
    )
    final_pipeline.fit(X, y)

    target_metrics_path = metrics_path or LATEST_METRICS_PATH

    trained_at = datetime.now(UTC)
    resolved_data_path = (
        Path(raw_data_path).resolve()
        if raw_data_path is not None
        else resolve_raw_data_path().resolve()
    )
    model_version = trained_at.strftime("%Y%m%d%H%M%S%f")

    # Extract feature importances from final pipeline
    feature_importances = extract_feature_importances(final_pipeline)

    metadata = {
        "model_name": best_name,
        "model_version": model_version,
        "trained_at_utc": trained_at.isoformat(),
        "random_state": RANDOM_STATE,
        "source_data_path": str(resolved_data_path),
        "training_rows": int(len(modeling_df)),
        "feature_columns": X.columns.tolist(),
        "metrics": best_metrics,
        "all_model_metrics": all_metrics,
        "cv_scores": cv_scores,
        "ui_options": extract_ui_options(raw_df),
    }
    if feature_importances is not None:
        metadata["feature_importances"] = feature_importances

    if model_dir is not None:
        write_training_outputs(
            model_dir=model_dir,
            final_pipeline=final_pipeline,
            metadata=metadata,
            metrics_path=target_metrics_path,
        )
        register_model_version(metadata, model_dir, make_active=True)
        clear_prediction_caches()
        metadata["promoted_to_production"] = True
        append_training_run(metadata)
        logger.info("Model %s promoted to production (explicit model_dir)", model_version)
        return metadata

    with tempfile.TemporaryDirectory() as temp_dir_name:
        staging_dir = Path(temp_dir_name)
        write_training_outputs(
            model_dir=staging_dir,
            final_pipeline=final_pipeline,
            metadata=metadata,
            metrics_path=target_metrics_path,
        )
        register_model_version(metadata, staging_dir, make_active=False)

    production_metadata = load_production_metadata()
    promoted = candidate_beats_production(metadata, production_metadata)
    if promoted:
        activate_model_version(metadata["model_version"])
    elif production_metadata is not None:
        set_active_registry_version(production_metadata["model_version"])

    metadata["promoted_to_production"] = promoted
    append_training_run(metadata)
    logger.info("Training complete: model=%s version=%s promoted=%s", best_name, model_version, promoted)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and persist the laptop price model.")
    parser.add_argument("--data", dest="data_path", default=None, help="Optional path to laptop_data.csv")
    parser.add_argument("--list-versions", action="store_true", help="List registered model versions")
    parser.add_argument(
        "--activate-version",
        dest="activate_version",
        default=None,
        help="Promote a registered model version to production",
    )
    parser.add_argument(
        "--no-tuning",
        action="store_true",
        help="Skip GridSearchCV hyperparameter tuning for faster training",
    )
    args = parser.parse_args()

    if args.list_versions:
        print(json.dumps(load_registry_index(), indent=2))
        return

    if args.activate_version:
        metadata = activate_model_version(args.activate_version)
        print(json.dumps(metadata, indent=2))
        return

    metadata = train_and_save(raw_data_path=args.data_path, enable_tuning=not args.no_tuning)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
