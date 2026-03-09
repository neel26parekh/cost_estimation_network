from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import httpx

from .config import METADATA_PATH, MODEL_PATH, ROOT_DIR


def load_local_env_file(env_path: Path | None = None) -> None:
    target = env_path or ROOT_DIR / ".env"
    if not target.exists():
        return

    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip()


def build_auth_headers(api_key: str | None) -> dict[str, str]:
    if not api_key:
        return {}
    return {"x-api-key": api_key}


def default_prediction_payload() -> dict[str, Any]:
    return {
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


def run_preflight_checks() -> dict[str, Any]:
    load_local_env_file()
    env_path = ROOT_DIR / ".env"
    render_path = ROOT_DIR / "render.yaml"
    docker_available = shutil.which("docker") is not None
    git_available = shutil.which("git") is not None

    checks = {
        "env_file_present": env_path.exists(),
        "api_key_configured": bool(os.getenv("API_KEY", "")),
        "model_artifact_present": MODEL_PATH.exists(),
        "metadata_present": METADATA_PATH.exists(),
        "docker_available": docker_available,
        "git_available": git_available,
        "render_config_present": render_path.exists(),
    }

    checks["ready"] = all(
        [
            checks["env_file_present"],
            checks["api_key_configured"],
            checks["model_artifact_present"],
            checks["metadata_present"],
            checks["render_config_present"],
        ]
    )
    return checks


def run_smoke_test(base_url: str, api_key: str | None) -> dict[str, Any]:
    load_local_env_file()
    resolved_api_key = api_key or os.getenv("API_KEY", "")
    headers = build_auth_headers(resolved_api_key)
    with httpx.Client(base_url=base_url, timeout=20.0) as client:
        root_response = client.get("/")
        health_response = client.get("/health")
        ready_response = client.get("/ready")
        metadata_response = client.get("/metadata", headers=headers)
        predict_response = client.post("/predict", headers=headers, json=default_prediction_payload())
        summary_response = client.get("/monitoring/summary", headers=headers)

    results = {
        "root_status": root_response.status_code,
        "health_status": health_response.status_code,
        "ready_status": ready_response.status_code,
        "metadata_status": metadata_response.status_code,
        "predict_status": predict_response.status_code,
        "summary_status": summary_response.status_code,
    }
    results["ok"] = all(status == 200 for status in results.values())
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Operational helpers for deployment preflight and smoke tests.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("preflight", help="Run local deployment readiness checks")

    smoke_parser = subparsers.add_parser("smoke-test", help="Run API smoke tests against a running service")
    smoke_parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="Base URL of the running API")
    smoke_parser.add_argument("--api-key", default=os.getenv("API_KEY", ""), help="API key for protected endpoints")

    args = parser.parse_args()
    if args.command == "preflight":
        result = run_preflight_checks()
    else:
        result = run_smoke_test(base_url=args.base_url, api_key=args.api_key)

    print(json.dumps(result, indent=2))
    if not result.get("ready", result.get("ok", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()