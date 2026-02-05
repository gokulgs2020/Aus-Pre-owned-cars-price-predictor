import numpy as np
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime
import json


# -----------------------------
# App UI
# -----------------------------
st.set_page_config(page_title="AI Car Deal Advisor", layout="centered")
st.title("🚗 AI Preowned Car Deal Advisor")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    pipe = joblib.load("models/final_price_pipe.joblib")

    bm = pd.read_csv("models/new_price_lookup_bm.csv")
    b = pd.read_csv("models/new_price_lookup_b.csv")

    return pipe, bm, b


pipe, bm, b = load_artifacts()


# -----------------------------
# Lookup new price
# -----------------------------
def lookup_new_price(brand, model):
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
def make_features(brand, model, age, km):
    log_km = np.log1p(km)
    age_km_interaction = (age * km) / 10000

    return pd.DataFrame([{
        "Age": age,
        "log_km": log_km,
        "FuelConsumption": 7.5,
        "CylindersinEngine": 4,
        "Seats": 5,
        "age_kilometer_interaction": age_km_interaction,
        "Brand": brand,
        "Model": model,
        "UsedOrNew": "USED",
        "DriveType": "FWD",
        "BodyType": "Sedan",
        "Transmission": "Automatic",
        "FuelType": "Gasoline",
    }])



# -----------------------------
# Session State
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vehicle_data" not in st.session_state:
    st.session_state.vehicle_data = {}


# -----------------------------
# Chat Input
# -----------------------------
user_input = st.chat_input("Paste listing or answer questions...")

if user_input:

    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # ---------- LLM Prompt ----------
    prompt = f"""
You are collecting vehicle details for valuation.

Current known data:
{st.session_state.vehicle_data}

User message:
{user_input}

Required fields:
Brand
Model
Year
Kilometres

Return ONLY valid JSON:
{{
 "extracted_data": {{}},
 "next_question": ""
}}

If all required fields exist:
next_question = "DONE"
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    output = response.choices[0].message.content

    try:
        parsed = json.loads(output)

        st.session_state.vehicle_data.update(parsed["extracted_data"])
        next_q = parsed["next_question"]

    except:
        next_q = "Sorry, I could not understand. Please re-enter."

    st.session_state.chat_history.append({"role": "assistant", "content": next_q})


# -----------------------------
# Display Chat
# -----------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# -----------------------------
# Run Pricing When Ready
# -----------------------------
required = ["Brand", "Model", "Year", "Kilometres"]

if all(k in st.session_state.vehicle_data for k in required):

    data = st.session_state.vehicle_data

    brand = data["Brand"]
    model = data["Model"]
    year = int(data["Year"])
    kms = float(data["Kilometres"])

    age = datetime.now().year - year

    new_price = lookup_new_price(brand, model)

    if not np.isnan(new_price):

        X = make_features(brand, model, age, kms)

        log_ret = float(pipe.predict(X)[0])
        retention = float(np.exp(log_ret))
        pred_price = retention * new_price

        st.divider()
        st.success(f"Estimated Price: A$ {pred_price:,.0f}")

        st.caption(f"Retention: {retention:.3f} | Proxy New Price: A$ {new_price:,.0f}")
