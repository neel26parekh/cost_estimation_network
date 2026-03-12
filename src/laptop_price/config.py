import os
from pathlib import Path


def _resolve_root_dir() -> Path:
    env_root = os.getenv("LAPTOP_PRICE_ROOT")
    if env_root:
        candidate = Path(env_root).resolve()
        if candidate.exists():
            return candidate

    cwd = Path.cwd().resolve()
    cwd_markers = [cwd / "pyproject.toml", cwd / "data", cwd / "laptop_data.csv"]
    if any(marker.exists() for marker in cwd_markers):
        return cwd

    package_root = Path(__file__).resolve().parents[2]
    package_markers = [package_root / "pyproject.toml", package_root / "data", package_root / "laptop_data.csv"]
    if any(marker.exists() for marker in package_markers):
        return package_root

    return cwd


ROOT_DIR = _resolve_root_dir()
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODEL_DIR = ROOT_DIR / "models" / "production"
MODEL_REGISTRY_DIR = ROOT_DIR / "models" / "registry"
REGISTRY_INDEX_PATH = MODEL_REGISTRY_DIR / "index.json"
METRICS_DIR = ROOT_DIR / "reports" / "metrics"
DRIFT_REPORT_DIR = ROOT_DIR / "reports" / "drift"
LOGS_DIR = ROOT_DIR / "logs"

MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
LATEST_METRICS_PATH = METRICS_DIR / "latest_metrics.json"
LATEST_DRIFT_REPORT_PATH = DRIFT_REPORT_DIR / "latest_drift_report.json"
PREDICTION_LOG_PATH = LOGS_DIR / "predictions.jsonl"
PREDICTION_DB_PATH = LOGS_DIR / "predictions.db"

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_KEY = os.getenv("API_KEY", "")
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RECENT_PREDICTIONS_LIMIT = int(os.getenv("RECENT_PREDICTIONS_LIMIT", "20"))
DRIFT_ANALYSIS_LIMIT = int(os.getenv("DRIFT_ANALYSIS_LIMIT", "200"))
NUMERIC_DRIFT_THRESHOLD = float(os.getenv("NUMERIC_DRIFT_THRESHOLD", "0.5"))
CATEGORICAL_UNSEEN_RATE_THRESHOLD = float(os.getenv("CATEGORICAL_UNSEEN_RATE_THRESHOLD", "0.1"))
MIN_DRIFT_ALERT_SAMPLE_SIZE = int(os.getenv("MIN_DRIFT_ALERT_SAMPLE_SIZE", "25"))
CORS_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "*").split(",") if origin.strip()]
MODEL_MAX_AGE_DAYS = int(os.getenv("MODEL_MAX_AGE_DAYS", "30"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

RAW_DATA_CANDIDATES = [
    RAW_DATA_DIR / "laptop_data.csv",
    ROOT_DIR / "laptop_data.csv",
    Path.cwd().resolve() / "data" / "raw" / "laptop_data.csv",
    Path.cwd().resolve() / "laptop_data.csv",
]

RANDOM_STATE = 2
TARGET_COLUMN = "Price"
FEATURE_COLUMNS = [
    "Company",
    "TypeName",
    "Ram",
    "Weight",
    "Touchscreen",
    "Ips",
    "ppi",
    "Cpu_brand",
    "HDD",
    "SSD",
    "gpu_brand",
    "os",
]
CATEGORICAL_COLUMNS = ["Company", "TypeName", "Cpu_brand", "gpu_brand", "os"]
NUMERIC_COLUMNS = [
    "Ram",
    "Weight",
    "Touchscreen",
    "Ips",
    "ppi",
    "HDD",
    "SSD",
]


def ensure_directories() -> None:
    for directory in (RAW_DATA_DIR, MODEL_DIR, MODEL_REGISTRY_DIR, METRICS_DIR, DRIFT_REPORT_DIR, LOGS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def resolve_raw_data_path() -> Path:
    for candidate in RAW_DATA_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find laptop_data.csv. Expected it at data/raw/laptop_data.csv or laptop_data.csv."
    )
