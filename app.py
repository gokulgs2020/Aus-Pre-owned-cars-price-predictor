import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load model + dataset
# -----------------------------
@st.cache_resource
def load_pipe():
    with open("final_price_pipe.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_features():
    with open("model_features.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("car_data_cleaned.csv")

pipe = load_pipe()
feature_cols = load_features()
df = load_data()

st.set_page_config(page_title="Car Price + Retention Curves", layout="wide")
st.title("🚗 Car Price Predictor + Brand×Km Retention Curves")

# -----------------------------
# Dropdown setup
# -----------------------------
brand_counts = df["Brand"].value_counts()
valid_brands = brand_counts[brand_counts >= 100].index
df = df[df["Brand"].isin(valid_brands)].copy()

brand_model_mapping = df.groupby("Brand")["Model"].unique().apply(list).to_dict()

# -----------------------------
# UI inputs
# -----------------------------
c1, c2 = st.columns(2)

with c1:
    brand = st.selectbox("Brand", sorted(valid_brands))
    model_choice = st.selectbox("Model", sorted(brand_model_mapping[brand]))
    year = st.number_input("Year", min_value=1980, max_value=2026, value=2020)
    kms = st.number_input("Kilometres", min_value=0, max_value=500000, value=50000, step=5000)
    used_or_new = st.selectbox("Used/New", ["USED", "NEW"], index=0)

with c2:
    transmission = st.selectbox("Transmission", sorted(df["Transmission"].astype(str).unique()))
    fuel_type = st.selectbox("FuelType", sorted(df["FuelType"].astype(str).unique()))
    drive_type = st.selectbox("DriveType", sorted(df["DriveType"].unique()))
    body_type = st.selectbox("BodyType", sorted(df["BodyType"].unique()))
    cylinders = st.selectbox("CylindersinEngine", sorted(df["CylindersinEngine"].unique()))
    seats = st.selectbox("Seats", sorted(df["Seats"].unique()))
    fuel_consumption = st.number_input("FuelConsumption", min_value=0.0, max_value=30.0, value=float(df["FuelConsumption"].median()))

# -----------------------------
# Build input row for the pipeline
# IMPORTANT: must match your training columns (X columns)
# -----------------------------
age = 2026 - year
age_km_inter = age * (kms / 100_000)  # keep same scaling you used in training

x = pd.DataFrame([{
    "Brand": brand,
    "Year": float(year),
    "Model": model_choice,
    "Car/Suv": "Unknown",
    "Title": "Unknown",
    "UsedOrNew": used_or_new,
    "Transmission": int(transmission),
    "Engine": "Unknown",
    "DriveType": drive_type,
    "FuelType": int(fuel_type),
    "FuelConsumption": float(fuel_consumption),
    "Kilometres": float(kms),
    "ColourExtInt": "Unknown",
    "Location": "Unknown",
    "CylindersinEngine": int(cylinders),
    "BodyType": body_type,
    "Doors": "Unknown",
    "Seats": int(seats),
    "Age": float(age),
    "age_kilometer_interaction": float(age_km_inter),
}])

# align columns to training set
for col in feature_cols:
    if col not in x.columns:
        x[col] = np.nan
x = x[feature_cols]

# -----------------------------
# Predict price
# If you trained log(Price): exp() it back.
# If you trained Price directly: use as-is.
# -----------------------------
if st.button("Predict Price", type="primary"):
    pred = float(pipe.predict(x)[0])
    # If y was log(Price), uncomment:
    # pred = float(np.exp(pred))
    st.success(f"💰 Predicted Price: ${pred:,.0f}")

# -----------------------------
# Retention proxy (Brand×Model near-new price estimate)
# -----------------------------
def estimate_new_price(df, brand, model_choice):
    near_new = df[(df["Brand"] == brand) & (df["Model"] == model_choice) & (df["Kilometres"] <= 10_000)]
    if len(near_new) >= 5:
        return float(near_new["Price"].quantile(0.95))
    near_new_b = df[(df["Brand"] == brand) & (df["Kilometres"] <= 10_000)]
    if len(near_new_b) >= 5:
        return float(near_new_b["Price"].quantile(0.95))
    return np.nan

new_price_proxy = estimate_new_price(df, brand, model_choice)

if not np.isnan(new_price_proxy):
    st.caption(f"New Price Proxy (95th pct of <=10k km): ${new_price_proxy:,.0f}")

# -----------------------------
# Brand×Km curve (sweep km)
# -----------------------------
st.subheader("📉 Brand×Km price + retention curve")

km_max = st.slider("Max kilometres", 50000, 500000, 300000, step=25000)
kms_grid = np.linspace(0, km_max, 35)

curve_rows = []
for km in kms_grid:
    age_km_inter = age * (km / 100_000)

    xk = x.copy()
    xk.loc[0, "Kilometres"] = float(km)
    xk.loc[0, "age_kilometer_interaction"] = float(age_km_inter)

    pred_k = float(pipe.predict(xk)[0])
    # If y was log(Price), uncomment:
    # pred_k = float(np.exp(pred_k))

    retention = np.nan
    if not np.isnan(new_price_proxy) and new_price_proxy > 0:
        retention = pred_k / new_price_proxy

    curve_rows.append((km, pred_k, retention))

curve = pd.DataFrame(curve_rows, columns=["Kilometres", "PredictedPrice", "RetentionProxy"])

# Plot predicted price
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(curve["Kilometres"], curve["PredictedPrice"], marker="o")
ax.set_title(f"Predicted Price vs Kilometres ({brand} {model_choice})")
ax.set_xlabel("Kilometres")
ax.set_ylabel("Predicted Price")
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Plot retention
if curve["RetentionProxy"].notna().any():
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(curve["Kilometres"], curve["RetentionProxy"], marker="o")
    ax2.set_title(f"Retention Proxy vs Kilometres ({brand} {model_choice})")
    ax2.set_xlabel("Kilometres")
    ax2.set_ylabel("Retention (Price / NewPriceProxy)")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

st.dataframe(curve)
