from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from .config import PREDICTION_DB_PATH, PREDICTION_LOG_PATH, RECENT_PREDICTIONS_LIMIT, ensure_directories


def ensure_prediction_store() -> None:
    ensure_directories()
    with sqlite3.connect(PREDICTION_DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                request_id TEXT PRIMARY KEY,
                logged_at_utc TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                predicted_price_inr REAL NOT NULL,
                latency_ms REAL NOT NULL,
                features_json TEXT NOT NULL
            )
            """
        )
        connection.commit()


def append_prediction_log(payload: dict[str, Any]) -> None:
    ensure_prediction_store()
    log_record = {
        "logged_at_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with PREDICTION_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(log_record)}\n")

    with sqlite3.connect(PREDICTION_DB_PATH) as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO predictions (
                request_id,
                logged_at_utc,
                model_name,
                model_version,
                predicted_price_inr,
                latency_ms,
                features_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                log_record["request_id"],
                log_record["logged_at_utc"],
                log_record["model_name"],
                log_record["model_version"],
                log_record["predicted_price_inr"],
                log_record["latency_ms"],
                json.dumps(log_record["features"]),
            ),
        )
        connection.commit()


def read_recent_prediction_logs(limit: int | None = None) -> list[dict[str, Any]]:
    max_records = limit or RECENT_PREDICTIONS_LIMIT
    if PREDICTION_DB_PATH.exists():
        ensure_prediction_store()
        with sqlite3.connect(PREDICTION_DB_PATH) as connection:
            rows = connection.execute(
                """
                SELECT logged_at_utc, request_id, model_name, model_version, predicted_price_inr, latency_ms, features_json
                FROM predictions
                ORDER BY logged_at_utc DESC
                LIMIT ?
                """,
                (max_records,),
            ).fetchall()

        return [
            {
                "logged_at_utc": row[0],
                "request_id": row[1],
                "model_name": row[2],
                "model_version": row[3],
                "predicted_price_inr": row[4],
                "latency_ms": row[5],
                "features": json.loads(row[6]),
            }
            for row in rows
        ]

    if not PREDICTION_LOG_PATH.exists():
        return []

    with PREDICTION_LOG_PATH.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle.readlines() if line.strip()]

    recent_lines = lines[-max_records:]
    return [json.loads(line) for line in reversed(recent_lines)]


def summarize_prediction_logs() -> dict[str, Any]:
    recent_logs = read_recent_prediction_logs(limit=RECENT_PREDICTIONS_LIMIT)
    if not recent_logs:
        return {
            "prediction_count": 0,
            "latest_request_id": None,
            "active_model_versions": [],
            "average_latency_ms": None,
        }

    average_latency = sum(item["latency_ms"] for item in recent_logs) / len(recent_logs)
    active_model_versions = sorted({item["model_version"] for item in recent_logs}, reverse=True)
    return {
        "prediction_count": len(recent_logs),
        "latest_request_id": recent_logs[0]["request_id"],
        "active_model_versions": active_model_versions,
        "average_latency_ms": round(average_latency, 2),
    }