"""Data validation schemas for training and inference DataFrames."""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema

from .config import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS, TARGET_COLUMN


training_schema = DataFrameSchema(
    columns={
        "Company": Column(str, nullable=False),
        "TypeName": Column(str, nullable=False),
        "Ram": Column(int, pa.Check.in_range(1, 128), nullable=False),
        "Weight": Column(float, pa.Check.in_range(0.1, 10.0), nullable=False),
        "Touchscreen": Column(int, pa.Check.isin([0, 1]), nullable=False),
        "Ips": Column(int, pa.Check.isin([0, 1]), nullable=False),
        "ppi": Column(float, pa.Check.greater_than(0), nullable=False),
        "Cpu_brand": Column(str, nullable=False),
        "HDD": Column(int, pa.Check.greater_than_or_equal_to(0), nullable=False),
        "SSD": Column(int, pa.Check.greater_than_or_equal_to(0), nullable=False),
        "gpu_brand": Column(str, nullable=False),
        "os": Column(str, nullable=False),
        TARGET_COLUMN: Column(float, pa.Check.greater_than(0), nullable=False),
    },
    strict=False,
    coerce=True,
)


inference_schema = DataFrameSchema(
    columns={
        "Company": Column(str, nullable=False),
        "TypeName": Column(str, nullable=False),
        "Ram": Column(int, pa.Check.in_range(1, 128), nullable=False),
        "Weight": Column(float, pa.Check.in_range(0.1, 10.0), nullable=False),
        "Touchscreen": Column(int, pa.Check.isin([0, 1]), nullable=False),
        "Ips": Column(int, pa.Check.isin([0, 1]), nullable=False),
        "ppi": Column(float, pa.Check.greater_than(0), nullable=False),
        "Cpu_brand": Column(str, nullable=False),
        "HDD": Column(int, pa.Check.greater_than_or_equal_to(0), nullable=False),
        "SSD": Column(int, pa.Check.greater_than_or_equal_to(0), nullable=False),
        "gpu_brand": Column(str, nullable=False),
        "os": Column(str, nullable=False),
    },
    strict=False,
    coerce=True,
)


def validate_training_data(df):
    """Validate a training DataFrame, raising pandera.errors.SchemaError on failure."""
    return training_schema.validate(df)


def validate_inference_data(df):
    """Validate an inference DataFrame, raising pandera.errors.SchemaError on failure."""
    return inference_schema.validate(df)
