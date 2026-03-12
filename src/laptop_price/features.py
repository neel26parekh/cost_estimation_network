from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, TARGET_COLUMN, resolve_raw_data_path


def load_raw_dataset(path: str | None = None) -> pd.DataFrame:
    data_path = resolve_raw_data_path() if path is None else path
    return pd.read_csv(data_path)


def compute_ppi(screen_resolution: str, screen_size: float) -> float:
    x_res, y_res = [int(value) for value in screen_resolution.split("x")]
    return float(np.sqrt((x_res**2) + (y_res**2)) / screen_size)


def categorize_cpu(cpu_value: str) -> str:
    cpu_name = " ".join(cpu_value.split()[:3])
    if cpu_name in {"Intel Core i7", "Intel Core i5", "Intel Core i3"}:
        return cpu_name
    if cpu_name.startswith("Intel"):
        return "Other Intel Processor"
    if cpu_name.startswith("AMD"):
        return "AMD Processor"
    return "Other Processor"


def categorize_os(os_value: str) -> str:
    if os_value in {"Windows 10", "Windows 7", "Windows", "Windows 10 S"}:
        return "Windows"
    if os_value in {"Mac", "Mac OS X", "macOS"}:
        return "Mac"
    return "Linux_and_others"


def _parse_memory_components(memory_series: pd.Series) -> pd.DataFrame:
    normalized = memory_series.astype(str).replace("\\.0", "", regex=True)
    normalized = normalized.str.replace("GB", "", regex=False)
    normalized = normalized.str.replace("TB", "000", regex=False)
    split_memory = normalized.str.split("+", n=1, expand=True, regex=False)

    first = split_memory[0].str.strip()
    second = split_memory[1].fillna("0")

    first_numeric = first.str.replace(r"\D", "", regex=True).replace("", "0").astype(int)
    second_numeric = second.str.replace(r"\D", "", regex=True).replace("", "0").astype(int)

    frame = pd.DataFrame(index=memory_series.index)
    frame["HDD"] = (
        first_numeric * first.str.contains("HDD").astype(int)
        + second_numeric * second.str.contains("HDD").astype(int)
    )
    frame["SSD"] = (
        first_numeric * first.str.contains("SSD").astype(int)
        + second_numeric * second.str.contains("SSD").astype(int)
    )
    frame["Hybrid"] = (
        first_numeric * first.str.contains("Hybrid").astype(int)
        + second_numeric * second.str.contains("Hybrid").astype(int)
    )
    frame["Flash_Storage"] = (
        first_numeric * first.str.contains("Flash Storage").astype(int)
        + second_numeric * second.str.contains("Flash Storage").astype(int)
    )
    return frame


def build_modeling_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.drop_duplicates().reset_index(drop=True)
    df["Ram"] = df["Ram"].str.replace("GB", "", regex=False).astype("int64")
    df["Weight"] = df["Weight"].str.replace("kg", "", regex=False).astype("float64")
    df["Touchscreen"] = df["ScreenResolution"].apply(lambda value: 1 if "Touchscreen" in value else 0)
    df["Ips"] = df["ScreenResolution"].apply(lambda value: 1 if "IPS" in value else 0)

    resolution_xy = df["ScreenResolution"].str.extract(r"(?P<x_res>\d+)x(?P<y_res>\d+)")
    df["x_res"] = resolution_xy["x_res"].astype(int)
    df["y_res"] = resolution_xy["y_res"].astype(int)
    df["ppi"] = np.sqrt((df["x_res"] ** 2) + (df["y_res"] ** 2)) / df["Inches"]

    df["Cpu_brand"] = df["Cpu"].apply(categorize_cpu)

    memory_components = _parse_memory_components(df["Memory"])
    df = pd.concat([df, memory_components], axis=1)
    df["gpu_brand"] = df["Gpu"].str.split().str[0]
    df = df[df["gpu_brand"] != "ARM"].copy()
    df["os"] = df["OpSys"].apply(categorize_os)

    keep_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    modeling_df = df[keep_columns].copy()
    modeling_df[TARGET_COLUMN] = modeling_df[TARGET_COLUMN].astype(float)
    return modeling_df.reset_index(drop=True)


def build_training_matrices(modeling_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = modeling_df[FEATURE_COLUMNS].copy()
    y = np.log(modeling_df[TARGET_COLUMN].copy())
    return X, y


def extract_ui_options(raw_df: pd.DataFrame) -> dict[str, list[Any]]:
    modeling_df = build_modeling_dataframe(raw_df)
    resolution_options = sorted(
        {
            match
            for match in raw_df["ScreenResolution"].str.extract(r"(\d+x\d+)", expand=False).dropna().tolist()
        }
    )
    return {
        "companies": sorted(modeling_df["Company"].unique().tolist()),
        "types": sorted(modeling_df["TypeName"].unique().tolist()),
        "ram_options": sorted(int(value) for value in modeling_df["Ram"].unique().tolist()),
        "screen_resolutions": resolution_options,
        "cpu_brands": sorted(modeling_df["Cpu_brand"].unique().tolist()),
        "hdd_options": sorted(int(value) for value in modeling_df["HDD"].unique().tolist()),
        "ssd_options": sorted(int(value) for value in modeling_df["SSD"].unique().tolist()),
        "gpu_brands": sorted(modeling_df["gpu_brand"].unique().tolist()),
        "os_options": sorted(modeling_df["os"].unique().tolist()),
    }


def _to_mapping(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        return record
    if hasattr(record, "model_dump"):
        return record.model_dump()
    if is_dataclass(record):
        return asdict(record)
    raise TypeError(f"Unsupported prediction record type: {type(record)!r}")


def build_inference_dataframe(records: Iterable[Any]) -> pd.DataFrame:
    rows = []
    for record in records:
        payload = _to_mapping(record)
        rows.append(
            {
                "Company": payload["company"],
                "TypeName": payload["type_name"],
                "Ram": int(payload["ram"]),
                "Weight": float(payload["weight"]),
                "Touchscreen": int(bool(payload["touchscreen"])),
                "Ips": int(bool(payload["ips"])),
                "ppi": compute_ppi(payload["screen_resolution"], float(payload["screen_size"])),
                "Cpu_brand": payload["cpu_brand"],
                "HDD": int(payload["hdd"]),
                "SSD": int(payload["ssd"]),
                "gpu_brand": payload["gpu_brand"],
                "os": payload["os"],
            }
        )

    inference_df = pd.DataFrame(rows)
    return inference_df[FEATURE_COLUMNS]


