from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from laptop_price.predict import load_metadata, predict_price  # noqa: E402
from laptop_price.schemas import PredictionRequest  # noqa: E402


def main() -> None:
    st.set_page_config(page_title="Laptop Price Predictor", layout="centered")
    st.title("Laptop Price Predictor")
    st.caption("Laptop price prediction backed by the reproducible training pipeline and saved production artifacts.")

    try:
        metadata = load_metadata()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.info("Run `python -m laptop_price.train` from the project root to generate the model artifacts.")
        return

    options = metadata["ui_options"]

    company = st.selectbox("Brand", options["companies"])
    type_name = st.selectbox("Type", options["types"])
    ram = st.selectbox("RAM (GB)", options["ram_options"])
    weight = st.number_input("Weight of the laptop (kg)", min_value=0.1, value=1.5, step=0.1)
    touchscreen = st.selectbox("Touchscreen", ["No", "Yes"]) == "Yes"
    ips = st.selectbox("IPS", ["No", "Yes"]) == "Yes"
    screen_size = st.number_input("Screen size (inches)", min_value=10.0, value=15.6, step=0.1)
    screen_resolution = st.selectbox("Screen resolution", options["screen_resolutions"])
    cpu_brand = st.selectbox("CPU", options["cpu_brands"])
    hdd = st.selectbox("HDD (GB)", options["hdd_options"])
    ssd = st.selectbox("SSD (GB)", options["ssd_options"])
    gpu_brand = st.selectbox("GPU", options["gpu_brands"])
    operating_system = st.selectbox("OS", options["os_options"])

    if st.button("Predict Price"):
        request = PredictionRequest(
            company=company,
            type_name=type_name,
            ram=ram,
            weight=weight,
            touchscreen=touchscreen,
            ips=ips,
            screen_size=screen_size,
            screen_resolution=screen_resolution,
            cpu_brand=cpu_brand,
            hdd=hdd,
            ssd=ssd,
            gpu_brand=gpu_brand,
            os=operating_system,
        )
        predicted_price = predict_price(request)
        st.success(f"Predicted price: INR {predicted_price:,.0f}")

    with st.expander("Model info"):
        st.json(
            {
                "model_name": metadata["model_name"],
                "model_version": metadata["model_version"],
                "metrics": metadata["metrics"],
            }
        )


if __name__ == "__main__":
    main()
