"""Tests for data validation schemas."""

import pandas as pd
import pytest
from pandera.errors import SchemaError

from laptop_price.validation import validate_inference_data, validate_training_data


def _valid_training_row() -> dict:
    return {
        "Company": "Dell",
        "TypeName": "Notebook",
        "Ram": 8,
        "Weight": 2.1,
        "Touchscreen": 0,
        "Ips": 1,
        "ppi": 141.21,
        "Cpu_brand": "Intel Core i5",
        "HDD": 1000,
        "SSD": 256,
        "gpu_brand": "Nvidia",
        "os": "Windows",
        "Price": 55000.0,
    }


def test_valid_training_data_passes() -> None:
    df = pd.DataFrame([_valid_training_row()])
    result = validate_training_data(df)
    assert len(result) == 1


def test_invalid_ram_fails() -> None:
    row = _valid_training_row()
    row["Ram"] = 0  # below minimum
    df = pd.DataFrame([row])
    with pytest.raises(SchemaError):
        validate_training_data(df)


def test_invalid_weight_fails() -> None:
    row = _valid_training_row()
    row["Weight"] = 15.0  # above maximum
    df = pd.DataFrame([row])
    with pytest.raises(SchemaError):
        validate_training_data(df)


def test_negative_price_fails() -> None:
    row = _valid_training_row()
    row["Price"] = -100.0
    df = pd.DataFrame([row])
    with pytest.raises(SchemaError):
        validate_training_data(df)


def test_valid_inference_data_passes() -> None:
    row = _valid_training_row()
    del row["Price"]
    df = pd.DataFrame([row])
    result = validate_inference_data(df)
    assert len(result) == 1
