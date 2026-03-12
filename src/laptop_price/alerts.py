from __future__ import annotations

import argparse
import json
from typing import Any

from .config import MIN_DRIFT_ALERT_SAMPLE_SIZE
from .drift import load_latest_drift_report


def evaluate_drift_alert(report: dict[str, Any]) -> dict[str, Any]:
    sample_size = int(report.get("sample_size", 0))
    drift_detected = bool(report.get("drift_detected", False))

    if sample_size < MIN_DRIFT_ALERT_SAMPLE_SIZE:
        return {
            "should_alert": False,
            "reason": "insufficient_sample_size",
            "sample_size": sample_size,
            "minimum_required": MIN_DRIFT_ALERT_SAMPLE_SIZE,
        }

    if drift_detected:
        return {
            "should_alert": True,
            "reason": "drift_detected",
            "sample_size": sample_size,
        }

    return {
        "should_alert": False,
        "reason": "no_drift_detected",
        "sample_size": sample_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate drift report and fail when alert conditions are met.")
    parser.parse_args()

    report = load_latest_drift_report()
    evaluation = evaluate_drift_alert(report)
    print(json.dumps(evaluation, indent=2))

    if evaluation["should_alert"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
