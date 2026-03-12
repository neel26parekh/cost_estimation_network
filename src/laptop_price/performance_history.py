"""Model performance history tracker.

Stores an append-only JSON log of training run metrics over time so that
model quality can be monitored across versions.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from .config import METRICS_DIR

PERFORMANCE_HISTORY_PATH = METRICS_DIR / "performance_history.json"


def load_performance_history() -> list[dict[str, Any]]:
    """Load the full performance history list."""
    if not PERFORMANCE_HISTORY_PATH.exists():
        return []
    return json.loads(PERFORMANCE_HISTORY_PATH.read_text(encoding="utf-8"))


def append_training_run(metadata: dict[str, Any]) -> None:
    """Append a training run entry to the performance history."""
    history = load_performance_history()
    entry = {
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "model_version": metadata["model_version"],
        "model_name": metadata["model_name"],
        "metrics": metadata["metrics"],
        "training_rows": metadata.get("training_rows"),
        "promoted_to_production": metadata.get("promoted_to_production", False),
    }
    if "feature_importances" in metadata:
        entry["feature_importances"] = metadata["feature_importances"]
    if "cv_scores" in metadata:
        entry["cv_scores"] = metadata["cv_scores"]
    history.append(entry)
    PERFORMANCE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    PERFORMANCE_HISTORY_PATH.write_text(
        f"{json.dumps(history, indent=2)}\n", encoding="utf-8"
    )
