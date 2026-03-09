import pandas as pd

from laptop_price.drift import analyze_feature_drift, build_reference_profile, build_inference_feature_frame


def test_build_reference_profile_contains_numeric_and_categorical_baselines() -> None:
    raw_df = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1],
            "Company": ["Dell", "HP"],
            "TypeName": ["Notebook", "Notebook"],
            "Inches": [15.6, 14.0],
            "ScreenResolution": ["1920x1080", "1366x768"],
            "Cpu": ["Intel Core i5 7200U 2.5GHz", "Intel Core i3 6006U 2.0GHz"],
            "Ram": ["8GB", "4GB"],
            "Memory": ["256GB SSD+1TB HDD", "128GB SSD"],
            "Gpu": ["Nvidia GeForce GTX 1050", "Intel HD Graphics 620"],
            "OpSys": ["Windows 10", "Windows 10"],
            "Weight": ["2.1kg", "1.6kg"],
            "Price": [55000.0, 40000.0],
        }
    )

    profile = build_reference_profile(raw_df)

    assert "Ram" in profile["numeric"]
    assert "Company" in profile["categorical"]


def test_analyze_feature_drift_flags_numeric_shift_and_unseen_categories() -> None:
    inference_df = build_inference_feature_frame(
        [
            {
                "features": {
                    "company": "Alienware",
                    "type_name": "Notebook",
                    "ram": 64,
                    "weight": 5.0,
                    "touchscreen": False,
                    "ips": True,
                    "screen_size": 15.6,
                    "screen_resolution": "1920x1080",
                    "cpu_brand": "Intel Core i7",
                    "hdd": 2000,
                    "ssd": 1024,
                    "gpu_brand": "Nvidia",
                    "os": "Windows",
                }
            }
        ]
    )
    reference_profile = {
        "numeric": {
            "Ram": {"mean": 8.0, "std": 4.0},
            "Weight": {"mean": 2.0, "std": 0.5},
            "Touchscreen": {"mean": 0.2, "std": 0.4},
            "Ips": {"mean": 0.4, "std": 0.4},
            "ppi": {"mean": 140.0, "std": 20.0},
            "HDD": {"mean": 500.0, "std": 400.0},
            "SSD": {"mean": 256.0, "std": 128.0},
        },
        "categorical": {
            "Company": {"top_value": "Dell", "top_rate": 0.4},
            "TypeName": {"top_value": "Notebook", "top_rate": 0.5},
            "Cpu_brand": {"top_value": "Intel Core i5", "top_rate": 0.5},
            "gpu_brand": {"top_value": "Intel", "top_rate": 0.5},
            "os": {"top_value": "Windows", "top_rate": 0.8},
        },
    }
    metadata = {
        "ui_options": {
            "companies": ["Dell", "HP"],
            "types": ["Notebook"],
            "cpu_brands": ["Intel Core i5", "Intel Core i7"],
            "gpu_brands": ["Intel", "Nvidia"],
            "os_options": ["Windows", "Mac"],
        }
    }

    report = analyze_feature_drift(inference_df, reference_profile, metadata)

    assert report["detected"] is True
    assert report["numeric"]["Ram"]["detected"] is True
    assert report["categorical"]["Company"]["detected"] is True