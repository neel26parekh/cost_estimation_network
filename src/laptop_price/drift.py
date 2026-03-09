from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from .config import (
    CATEGORICAL_COLUMNS,
    DRIFT_ANALYSIS_LIMIT,
    CATEGORICAL_UNSEEN_RATE_THRESHOLD,
    FEATURE_COLUMNS,
    LATEST_DRIFT_REPORT_PATH,
    NUMERIC_COLUMNS,
    NUMERIC_DRIFT_THRESHOLD,
    ensure_directories,
)
from .features import build_inference_dataframe, build_modeling_dataframe, load_raw_dataset
from .monitoring import read_recent_prediction_logs
from .predict import load_metadata


UI_OPTION_KEYS = {
    "Company": "companies",
    "TypeName": "types",
    "Cpu_brand": "cpu_brands",
    "gpu_brand": "gpu_brands",
    "os": "os_options",
}


def build_reference_profile(raw_df: pd.DataFrame) -> dict[str, Any]:
    modeling_df = build_modeling_dataframe(raw_df)
    profile = {"numeric": {}, "categorical": {}}

    for column in NUMERIC_COLUMNS:
        series = modeling_df[column].astype(float)
        profile["numeric"][column] = {
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
        }

    for column in CATEGORICAL_COLUMNS:
        frequencies = modeling_df[column].value_counts(normalize=True)
        profile["categorical"][column] = {
            "top_value": None if frequencies.empty else str(frequencies.index[0]),
            "top_rate": 0.0 if frequencies.empty else float(frequencies.iloc[0]),
        }

    return profile


def build_inference_feature_frame(logs: list[dict[str, Any]]) -> pd.DataFrame:
    if not logs:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    return build_inference_dataframe([entry["features"] for entry in logs])


def analyze_feature_drift(
    inference_df: pd.DataFrame,
    reference_profile: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    numeric_drift: dict[str, Any] = {}
    categorical_drift: dict[str, Any] = {}

    for column in NUMERIC_COLUMNS:
        if inference_df.empty:
            numeric_drift[column] = {
                "detected": False,
                "train_mean": reference_profile["numeric"][column]["mean"],
                "inference_mean": None,
                "mean_shift_in_std": None,
            }
            continue

        train_mean = reference_profile["numeric"][column]["mean"]
        train_std = reference_profile["numeric"][column]["std"]
        inference_mean = float(inference_df[column].astype(float).mean())
        denominator = train_std if train_std > 0 else 1.0
        mean_shift = abs(inference_mean - train_mean) / denominator
        numeric_drift[column] = {
            "detected": mean_shift > NUMERIC_DRIFT_THRESHOLD,
            "train_mean": train_mean,
            "inference_mean": inference_mean,
            "mean_shift_in_std": float(mean_shift),
        }

    ui_options = metadata.get("ui_options", {})
    for column in CATEGORICAL_COLUMNS:
        if inference_df.empty:
            categorical_drift[column] = {
                "detected": False,
                "top_training_value": reference_profile["categorical"][column]["top_value"],
                "top_inference_value": None,
                "unseen_rate": None,
            }
            continue

        allowed_values = set(ui_options.get(UI_OPTION_KEYS[column], []))
        inference_series = inference_df[column].astype(str)
        unseen_rate = float((~inference_series.isin(allowed_values)).mean()) if allowed_values else 0.0
        top_inference = inference_series.value_counts(normalize=True)
        categorical_drift[column] = {
            "detected": unseen_rate > CATEGORICAL_UNSEEN_RATE_THRESHOLD,
            "top_training_value": reference_profile["categorical"][column]["top_value"],
            "top_inference_value": None if top_inference.empty else str(top_inference.index[0]),
            "unseen_rate": unseen_rate,
        }

    overall_detected = any(item["detected"] for item in numeric_drift.values()) or any(
        item["detected"] for item in categorical_drift.values()
    )

    return {
        "detected": overall_detected,
        "numeric": numeric_drift,
        "categorical": categorical_drift,
    }


def generate_drift_report(limit: int | None = None) -> dict[str, Any]:
    ensure_directories()
    logs = read_recent_prediction_logs(limit=limit or DRIFT_ANALYSIS_LIMIT)
    metadata = load_metadata()
    reference_profile = build_reference_profile(load_raw_dataset())
    inference_df = build_inference_feature_frame(logs)
    drift_summary = analyze_feature_drift(inference_df, reference_profile, metadata)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "sample_size": int(len(inference_df)),
        "drift_detected": drift_summary["detected"],
        "numeric_drift": drift_summary["numeric"],
        "categorical_drift": drift_summary["categorical"],
    }
    LATEST_DRIFT_REPORT_PATH.write_text(f"{json.dumps(report, indent=2)}\n", encoding="utf-8")
    return report


def load_latest_drift_report() -> dict[str, Any]:
    if not LATEST_DRIFT_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Drift report not found at {LATEST_DRIFT_REPORT_PATH}. Run `python -m laptop_price.drift` first."
        )
    return json.loads(LATEST_DRIFT_REPORT_PATH.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze prediction logs for feature drift.")
    parser.add_argument("--limit", type=int, default=DRIFT_ANALYSIS_LIMIT, help="Number of recent predictions to analyze")
    args = parser.parse_args()
    print(json.dumps(generate_drift_report(limit=args.limit), indent=2))


if __name__ == "__main__":
    main()