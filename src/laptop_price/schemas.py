from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    company: str = Field(min_length=1, examples=["Dell"])
    type_name: str = Field(min_length=1, examples=["Notebook"])
    ram: int = Field(gt=0, le=128, examples=[8])
    weight: float = Field(gt=0, le=10, examples=[2.1])
    touchscreen: bool
    ips: bool
    screen_size: float = Field(gt=8, le=25, examples=[15.6])
    screen_resolution: str = Field(pattern=r"^\d{3,4}x\d{3,4}$", examples=["1920x1080"])
    cpu_brand: str = Field(min_length=1, examples=["Intel Core i5"])
    hdd: int = Field(ge=0, le=4000, examples=[1000])
    ssd: int = Field(ge=0, le=4000, examples=[256])
    gpu_brand: str = Field(min_length=1, examples=["Nvidia"])
    os: str = Field(min_length=1, examples=["Windows"])


class PredictionResponse(BaseModel):
    predicted_price_inr: float
    model_name: str
    model_version: str
    request_id: str
    latency_ms: float
    currency: Literal["INR"] = "INR"


class PredictionLogEntry(BaseModel):
    logged_at_utc: str
    request_id: str
    model_name: str
    model_version: str
    predicted_price_inr: float
    latency_ms: float
    features: dict[str, Any]
