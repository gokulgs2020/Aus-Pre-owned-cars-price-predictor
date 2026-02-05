import os
import json
import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime
from openai import OpenAI


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Preowned Car Price Estimator", layout="centered")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not set")
    st.stop()

client = OpenAI(api_key=api_key)


# -----------------------------
# Helpers
# -----------------------------
def parse_numeric(value):

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        value = value.lower()
        value = value.replace("kms", "")
        value = value.replace("km", "")
        value = value.replace(",", "")
        value = value.replace("$", "")
        value = value.replace("k", "000")

        nums = re.findall(r"\d+", value)

        if nums:
            return float("".join(nums))

    return None


def safe_json_parse(text):

    text = text.strip()

    if text.startswith("```"):
        text = text.split("```")[1]

        if text.startswith("json"):
            text = text[4:]

    return json.loads(text.strip())


# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    pipe = joblib.load("models/final_price_pipe.joblib")
    bm = pd.read_csv("models/new_price_lookup_bm.csv")
    b = pd.read_csv("models/new_price_lookup_b.csv")
    brand_model_lookup = pd.read_csv("models/brand_model_lookup_50.csv")
    return pipe, bm, b, brand_model_lookup


pipe, bm, b, brand_model_lookup = load_artifacts()


# -----------------------------
# Lookup new price
# -----------------------------
def lookup_new_price(brand, model):

    row_bm = bm[(bm["Brand"] == brand) & (bm["Model"] == model)]
    if len(row_bm):
        return float(row_bm["New_Price_bm"].iloc[0])

    row_b = b[b["Brand"] == brand]
    if len(row_b):
        return float(row_b["New_Price_b"].iloc[0])

    return float("nan")


# -----------------------------
# Feature builder
# -----------------------------
def make_features(brand, model, age, km):

    return pd.DataFrame([{
        "Age": age,
        "log_km": np.log1p(km),
        "FuelConsumption": 7.5,
        "CylindersinEngine": 4,
        "Seats": 5,
        "age_kilometer_interaction": (age * km) / 10000,
        "Brand": brand,
        "Model": model,
        "UsedOrNew": "USED",
        "DriveType": "FWD",
        "BodyType": "Sedan",
        "Transmission": "Automatic",
        "FuelType": "Gasoline",
    }])


# -----------------------------
# Session state
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vehicle_data" not in st.session_state:
    st.session_state.vehicle_data = {}


# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(["🚗 Price Estimator", "🤖 AI Deal Advisor"])


# =====================================================
# TAB 1 → ORIGINAL ESTIMATOR
# =====================================================
with tab1:

    st.header("Price Estimator")

    brands = sorted(brand_model_lookup["Brand"].unique())
    brand = st.selectbox("Brand", brands)

    models = sorted(
        brand_model_lookup[brand_model_lookup["Brand"] == brand]["Model"].unique()
    )
    model = st.selectbox("Model", models)

    year = st.number_input("Year", 2000, datetime.now().year, 2020)
    km = st.number_input("Kilometres", 0, 200000, 60000)

    if st.button("Estimate Price"):

        age = datetime.now().year - year
        new_price = lookup_new_price(brand, model)

        if np.isnan(new_price):
            st.error("Brand/Model not in training data")
            st.stop()

        X = make_features(brand, model, age, km)

        log_ret = float(pipe.predict(X)[0])
        retention = float(np.exp(log_ret))
        pred_price = retention * new_price

        st.success(f"Estimated Price: A$ {pred_price:,.0f}")
        st.caption(f"Retention: {retention:.3f} | New Price Proxy: A$ {new_price:,.0f}")


# =====================================================
# TAB 2 → AI DEAL ADVISOR
# =====================================================
with tab2:

    st.header("Conversational AI Deal Advisor")

    if st.button("🔄 Restart Conversation"):
        st.session_state.chat_history = []
        st.session_state.vehicle_data = {}
        st.rerun()

    if st.session_state.vehicle_data:
        st.sidebar.write("Collected Data")
        st.sidebar.json(st.session_state.vehicle_data)

    user_input = st.chat_input("Paste listing or answer questions...")

    if user_input:

        st.session_state.chat_history.append({"role": "user", "content": user_input})

        prompt = f"""
Extract vehicle details from user message.

Current known data:
{st.session_state.vehicle_data}

User message:
{user_input}

Required fields:
Brand, Model, Year, Kilometres

Return ONLY JSON:
{{
 "extracted_data": {{}}
}}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        try:
            parsed = safe_json_parse(response.choices[0].message.content)

            st.session_state.vehicle_data.update(parsed.get("extracted_data", {}))

        except:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "I couldn't understand. Please re-enter."
            })

    # Display chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # -----------------------------
    # Strong Validation
    # -----------------------------
    required = ["Brand", "Model", "Year", "Kilometres"]

    missing = [
        k for k in required
        if k not in st.session_state.vehicle_data
        or st.session_state.vehicle_data[k] in ["", None]
    ]

    if missing:
        st.info(f"I still need: {', '.join(missing)}")

    # -----------------------------
    # Run Pricing
    # -----------------------------
    if len(missing) == 0:

        data = st.session_state.vehicle_data

        brand = data["Brand"]
        model = data["Model"]

        year = parse_numeric(data["Year"])
        kms = parse_numeric(data["Kilometres"])

        if year is None or kms is None:
            st.error("Could not parse Year or Kilometres.")
            st.stop()

        age = datetime.now().year - int(year)
        new_price = lookup_new_price(brand, model)

        if not np.isnan(new_price):

            X = make_features(brand, model, age, kms)

            log_ret = float(pipe.predict(X)[0])
            retention = float(np.exp(log_ret))
            pred_price = retention * new_price

            st.divider()
            st.success(f"Estimated Price: A$ {pred_price:,.0f}")
            st.caption(f"Retention: {retention:.3f} | New Price Proxy: A$ {new_price:,.0f}")
