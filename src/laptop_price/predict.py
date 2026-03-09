from __future__ import annotations

import json
from functools import lru_cache

import joblib
import numpy as np

from .config import METADATA_PATH, MODEL_PATH
from .features import build_inference_dataframe
from .logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Run `python -m laptop_price.train` first."
        )
    logger.info("Loading model from %s", MODEL_PATH)
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Model metadata not found at {METADATA_PATH}. Run `python -m laptop_price.train` first."
        )
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def predict_price(record) -> float:
    model = load_model()
    inference_df = build_inference_dataframe([record])
    prediction_log = model.predict(inference_df)[0]
    predicted_price = float(np.exp(prediction_log))
    if not np.isfinite(predicted_price):
        raise ValueError("Prediction result is not finite. Provide a realistic laptop configuration.")
    return predicted_price
