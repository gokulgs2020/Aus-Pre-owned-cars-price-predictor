import numpy as np
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime


# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    pipe = joblib.load("models/final_price_pipe.joblib")
    feature_cols = joblib.load("models/model_features.joblib")

    bm = pd.read_csv("models/new_price_lookup_bm.csv")
    b = pd.read_csv("models/new_price_lookup_b.csv")

    brand_model_lookup = pd.read_csv("models/brand_model_lookup_50.csv")
    seat_defaults = pd.read_csv("models/brand_model_seat_default.csv")
    multi_seat_lookup = pd.read_csv("models/multi_seat_models.csv")

    # ✅ Filter multi-seat list to only those in dropdown lookup
    multi_seat_lookup = multi_seat_lookup.merge(
        brand_model_lookup,
        on=["Brand", "Model"],
        how="inner"
    )

    # Normalize for safe matching
    for df_ in [bm, b, brand_model_lookup, seat_defaults, multi_seat_lookup]:
        if "Brand" in df_.columns:
            df_["Brand"] = df_["Brand"].astype(str).str.strip()
        if "Model" in df_.columns:
            df_["Model"] = df_["Model"].astype(str).str.strip()

    return pipe, feature_cols, bm, b, brand_model_lookup, seat_defaults, multi_seat_lookup


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

        "Brand": str(brand).strip(),
        "Model": str(model).strip(),
        "UsedOrNew": used_or_new,
        "DriveType": drive_type,
        "BodyType": body_type,
        "Transmission": transmission,
        "FuelType": fuel_type,
    }])


# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="Preowned Car Price Estimator", layout="centered")
st.title("🚗 Preowned Car Price Estimator (AU)")

pipe, feature_cols, bm, b, brand_model_lookup, seat_defaults, multi_seat_lookup = load_artifacts()

with st.sidebar:
    st.header("Vehicle details")

    # Brand dropdown
    brands = sorted(brand_model_lookup["Brand"].dropna().unique().tolist())
    brand = st.selectbox("Brand", brands)

    # Model dropdown filtered by Brand
    models = sorted(
        brand_model_lookup[brand_model_lookup["Brand"] == brand]["Model"]
        .dropna()
        .unique()
        .tolist()
    )
    model = st.selectbox("Model", models)

    # Defaults / simple inputs
    used_or_new = "USED"
    drive_type = st.selectbox("Drive Type", ["FWD", "AWD"], index=0)
    body_type = "Sedan"

    transmission = st.selectbox("Transmission", ["Automatic", "Manual"], index=0)

    # Manufacture year -> age
    current_year = datetime.now().year
    year = st.number_input("Year of Manufacture", min_value=2000, max_value=current_year, value=2020, step=1)
    age = current_year - year

    km = st.number_input("Kilometres", min_value=0, max_value=200000, value=60000, step=5000)

    # -----------------------------
    # Seats logic (only show dropdown for multi-seat models)
    # -----------------------------
    multi_seat_set = set(
        zip(
            multi_seat_lookup["Brand"].astype(str).str.strip(),
            multi_seat_lookup["Model"].astype(str).str.strip(),
        )
    )

    row = seat_defaults[
        (seat_defaults["Brand"] == brand) &
        (seat_defaults["Model"] == model)
    ]

    default_seats = int(row["DefaultSeats"].iloc[0]) if len(row) else 5

    seat_options = [2, 5, 6, 7]

    if (brand, model) in multi_seat_set:
        seats = st.selectbox(
            "Seats",
            seat_options,
            index=seat_options.index(default_seats) if default_seats in seat_options else 1
        )
    else:
        seats = default_seats
        st.caption(f"Seats fixed for this model → auto-set to **{seats}**")

    # -----------------------------
    # Fuel logic (Electric disables cylinders + fuel consumption)
    # -----------------------------
    fuel_type = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Hybrid", "Electric"], index=0)
    is_electric = (fuel_type == "Electric")

    fuel_consumption = st.slider(
        "Fuel Consumption (L/100km)",
        min_value=0.0,
        max_value=20.0,
        value=0.0 if is_electric else 7.5,
        step=0.5,
        disabled=is_electric
    )

    cylinders = st.slider(
        "Cylinders in Engine",
        min_value=0,
        max_value=8,
        value=0 if is_electric else 4,
        step=2,
        disabled=is_electric
    )


# -----------------------------
# Predict
# -----------------------------
if st.button("Estimate price", type="primary"):
    new_price = lookup_new_price(brand, model, bm, b)

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
