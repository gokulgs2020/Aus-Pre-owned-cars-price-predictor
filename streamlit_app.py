import numpy as np
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime
import re


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

    multi_seat_lookup = multi_seat_lookup.merge(
        brand_model_lookup,
        on=["Brand", "Model"],
        how="inner"
    )

    for df_ in [bm, b, brand_model_lookup, seat_defaults, multi_seat_lookup]:
        if "Brand" in df_.columns:
            df_["Brand"] = df_["Brand"].astype(str).str.strip()
        if "Model" in df_.columns:
            df_["Model"] = df_["Model"].astype(str).str.strip()

    return pipe, feature_cols, bm, b, brand_model_lookup, seat_defaults, multi_seat_lookup


# -----------------------------
# Lookups
# -----------------------------
def lookup_new_price(brand, model, bm, b):
    row_bm = bm[(bm["Brand"] == brand) & (bm["Model"] == model)]
    if len(row_bm) > 0:
        return float(row_bm["New_Price_bm"].iloc[0])

    row_b = b[b["Brand"] == brand]
    if len(row_b) > 0:
        return float(row_b["New_Price_b"].iloc[0])

    return float("nan")


# -----------------------------
# Feature Builder
# -----------------------------
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


# -----------------------------
# Carsales Text Parsers
# -----------------------------
def parse_price(text):
    m = re.search(r"\$\s*([0-9]{1,3}(?:,[0-9]{3})+)", text)
    return int(m.group(1).replace(",", "")) if m else None


def parse_kms(text):
    m = re.search(r"([0-9]{1,3}(?:,[0-9]{3})+)\s*km", text, re.IGNORECASE)
    return int(m.group(1).replace(",", "")) if m else None


def parse_year(text):
    m = re.search(r"\b(19[8-9]\d|20[0-2]\d)\b", text)
    return int(m.group(1)) if m else None


# -----------------------------
# Deal Classification
# -----------------------------
def classify_deal(listing_price, pred_price):
    gap = listing_price - pred_price
    gap_pct = gap / max(pred_price, 1)

    if gap_pct > 0.08:
        label = "⚠️ Overpriced"
    elif gap_pct < -0.08:
        label = "🔥 Bargain"
    else:
        label = "✅ Fair Price"

    return gap, gap_pct, label


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Preowned Car Price Estimator", layout="centered")
st.title("🚗 Preowned Car Price Estimator (AU)")

pipe, feature_cols, bm, b, brand_model_lookup, seat_defaults, multi_seat_lookup = load_artifacts()

tab1, tab2 = st.tabs(["🚗 Price Estimator", "🤖 Deal Advisor"])


# =====================================================
# TAB 1 → EXISTING ESTIMATOR (UNCHANGED)
# =====================================================
with tab1:

    with st.sidebar:
        st.header("Vehicle details")

        brands = sorted(brand_model_lookup["Brand"].dropna().unique())
        brand = st.selectbox("Brand", brands)

        models = sorted(
            brand_model_lookup[brand_model_lookup["Brand"] == brand]["Model"].dropna().unique()
        )
        model = st.selectbox("Model", models)

        used_or_new = "USED"
        drive_type = st.selectbox("Drive Type", ["FWD", "AWD"])
        body_type = "Sedan"

        transmission = st.selectbox("Transmission", ["Automatic", "Manual"])

        current_year = datetime.now().year
        year = st.number_input("Year", 2000, current_year, 2020)
        age = current_year - year

        km = st.number_input("Kilometres", 0, 200000, 60000, step=5000)

        row = seat_defaults[(seat_defaults["Brand"] == brand) &
                            (seat_defaults["Model"] == model)]

        seats = int(row["DefaultSeats"].iloc[0]) if len(row) else 5

        fuel_type = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Hybrid", "Electric"])
        is_electric = (fuel_type == "Electric")

        fuel_consumption = st.slider("Fuel Consumption", 0.0, 20.0, 7.5, disabled=is_electric)
        cylinders = st.slider("Cylinders", 0, 8, 4, step=2, disabled=is_electric)

    if st.button("Estimate price", type="primary"):

        new_price = lookup_new_price(brand, model, bm, b)

        X = make_features(
            brand, model, used_or_new, drive_type, body_type,
            transmission, fuel_type,
            age, km, fuel_consumption, cylinders, seats
        )

        log_ret = float(pipe.predict(X)[0])
        retention = float(np.exp(log_ret))
        pred_price = retention * new_price

        st.success(f"Estimated Price: A$ {pred_price:,.0f}")
        st.caption(f"Retention: {retention:.3f} | Proxy New Price: A$ {new_price:,.0f}")


# =====================================================
# TAB 2 → DEAL ADVISOR
# =====================================================
with tab2:

    st.header("Carsales Deal Advisor")

    raw_text = st.text_area("Paste Carsales Listing Text")

    brand_da = st.text_input("Brand")
    model_da = st.text_input("Model")

    if st.button("Analyze Deal"):

        price = parse_price(raw_text)
        kms = parse_kms(raw_text)
        year = parse_year(raw_text)

        if not all([price, kms, year, brand_da, model_da]):
            st.error("Missing values. Please paste richer listing text or fill manually.")
            st.stop()

        current_year = datetime.now().year
        age = current_year - year

        new_price = lookup_new_price(brand_da, model_da, bm, b)

        X = make_features(
            brand_da, model_da, "USED", "FWD", "Sedan",
            "Automatic", "Gasoline",
            age, kms, 7.5, 4, 5
        )

        log_ret = float(pipe.predict(X)[0])
        retention = float(np.exp(log_ret))
        pred_price = retention * new_price

        gap, gap_pct, label = classify_deal(price, pred_price)

        st.metric("Listing Price", f"A$ {price:,.0f}")
        st.metric("Estimated Price", f"A$ {pred_price:,.0f}")
        st.metric("Gap", f"A$ {gap:,.0f} ({gap_pct*100:.1f}%)")

        st.success(label)
