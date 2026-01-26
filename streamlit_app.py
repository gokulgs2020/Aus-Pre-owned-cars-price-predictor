import numpy as np
import pandas as pd
import streamlit as st
import joblib


@st.cache_resource
def load_artifacts():
    pipe = joblib.load("models/final_price_pipe.joblib")
    feature_cols = joblib.load("models/model_features.joblib")

    bm = pd.read_csv("models/new_price_lookup_bm.csv")
    b = pd.read_csv("models/new_price_lookup_b.csv")
    return pipe, feature_cols, bm, b


def lookup_new_price(brand: str, model: str, bm: pd.DataFrame, b: pd.DataFrame) -> float:
    row_bm = bm[(bm["Brand"] == brand) & (bm["Model"] == model)]
    if len(row_bm) > 0:
        return float(row_bm["New_Price_bm"].iloc[0])

    row_b = b[b["Brand"] == brand]
    if len(row_b) > 0:
        return float(row_b["New_Price_b"].iloc[0])

    return float("nan")


def make_features(
    brand, model, used_or_new, drive_type, body_type,
    transmission, fuel_type,
    age, km, fuel_consumption, cylinders, seats
):
    log_km = np.log1p(km)
    age_km_interaction = (age * km) / 10000

    return pd.DataFrame([{
        "Age": float(age),
        "log_km": float(log_km),
        "FuelConsumption": float(fuel_consumption),
        "CylindersinEngine": int(cylinders),
        "Seats": int(seats),
        "age_kilometer_interaction": float(age_km_interaction),

        "Brand": brand,
        "Model": model,
        "UsedOrNew": used_or_new,
        "DriveType": drive_type,
        "BodyType": body_type,
        "Transmission": transmission,
        "FuelType": fuel_type,
    }])


st.set_page_config(page_title="Preowned Car Price Estimator", layout="centered")
st.title("🚗 Preowned Car Price Estimator (AU)")

pipe, feature_cols, bm_lookup, b_lookup = load_artifacts()

with st.sidebar:
    st.header("Vehicle details")

    brand = st.text_input("Brand", value="Toyota")
    model = st.text_input("Model", value="Camry")
    used_or_new = st.selectbox("UsedOrNew", ["USED", "DEMO", "NEW"], index=0)
    drive_type = st.text_input("DriveType", value="FWD")
    body_type = st.text_input("BodyType", value="Sedan")

    transmission = st.selectbox("Transmission", ["Automatic", "Manual", "Other", "Unknown"], index=0)
    fuel_type = st.selectbox("FuelType", ["Gasoline", "Diesel", "Hybrid", "Electric", "Other", "Unknown"], index=0)

    age = st.number_input("Age (years)", min_value=0.0, max_value=40.0, value=5.0, step=1.0)
    km = st.number_input("Kilometres", min_value=0.0, max_value=400000.0, value=60000.0, step=1000.0)

    fuel_consumption = st.number_input(
        "FuelConsumption (L/100km)",
        min_value=0.0, max_value=30.0, value=7.5, step=0.1
    )
    cylinders = st.number_input("CylindersinEngine", min_value=0, max_value=16, value=4, step=1)
    seats = st.number_input("Seats", min_value=0, max_value=12, value=5, step=1)


if st.button("Estimate price"):
    new_price = lookup_new_price(brand, model, bm_lookup, b_lookup)

    if np.isnan(new_price):
        st.error("Brand/Model not found in training lookups. Try a more common Brand/Model from your dataset.")
        st.stop()

    X = make_features(
        brand, model, used_or_new, drive_type, body_type,
        transmission, fuel_type,
        float(age), float(km), float(fuel_consumption), int(cylinders), int(seats)
    )

    # y = log(retention)
    log_ret = float(pipe.predict(X)[0])
    retention = float(np.exp(log_ret))
    pred_price = float(retention * new_price)

    st.success(f"Estimated Price: **A$ {pred_price:,.0f}**")
    st.caption(f"Implied retention: {retention:.3f} | Proxy New Price: A$ {new_price:,.0f}")
