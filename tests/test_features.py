import pandas as pd

from laptop_price.features import build_inference_dataframe, build_modeling_dataframe, compute_ppi


def test_build_modeling_dataframe_creates_expected_features() -> None:
    raw_df = pd.DataFrame(
        {
            "Unnamed: 0": [0],
            "Company": ["Dell"],
            "TypeName": ["Notebook"],
            "Inches": [15.6],
            "ScreenResolution": ["IPS Panel Full HD / Touchscreen 1920x1080"],
            "Cpu": ["Intel Core i5 7200U 2.5GHz"],
            "Ram": ["8GB"],
            "Memory": ["256GB SSD+1TB HDD"],
            "Gpu": ["Nvidia GeForce GTX 1050"],
            "OpSys": ["Windows 10"],
            "Weight": ["2.1kg"],
            "Price": [55000.0],
        }
    )

    modeling_df = build_modeling_dataframe(raw_df)
    row = modeling_df.iloc[0]

    assert row["Ram"] == 8
    assert row["Weight"] == 2.1
    assert row["Touchscreen"] == 1
    assert row["Ips"] == 1
    assert row["Cpu_brand"] == "Intel Core i5"
    assert row["HDD"] == 1000
    assert row["SSD"] == 256
    assert row["gpu_brand"] == "Nvidia"
    assert row["os"] == "Windows"
    assert round(row["ppi"], 2) == round(compute_ppi("1920x1080", 15.6), 2)


def test_build_inference_dataframe_matches_training_contract() -> None:
    inference_df = build_inference_dataframe(
        [
            {
                "company": "Dell",
                "type_name": "Notebook",
                "ram": 8,
                "weight": 2.1,
                "touchscreen": True,
                "ips": True,
                "screen_size": 15.6,
                "screen_resolution": "1920x1080",
                "cpu_brand": "Intel Core i5",
                "hdd": 1000,
                "ssd": 256,
                "gpu_brand": "Nvidia",
                "os": "Windows",
            }
        ]
    )

    assert inference_df.shape == (1, 12)
    assert inference_df.iloc[0]["Touchscreen"] == 1
    assert inference_df.iloc[0]["Ips"] == 1
