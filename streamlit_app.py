import os
import json
import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime
from openai import OpenAI

# =====================================================
# CONFIG & CLIENT
# =====================================================
st.set_page_config(page_title="Car Deal Advisor", layout="wide")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OPENAI_API_KEY")
    st.stop()

client = OpenAI(api_key=api_key)

# =====================================================
# HELPERS & ARTIFACTS
# =====================================================
@st.cache_resource
def load_artifacts():
    pipe = joblib.load("models/final_price_pipe.joblib")
    bm = pd.read_csv("models/new_price_lookup_bm.csv")
    b = pd.read_csv("models/new_price_lookup_b.csv")
    lookup = pd.read_csv("models/brand_model_lookup_50.csv")
    return pipe, bm, b, lookup

pipe, bm, b, brand_model_lookup = load_artifacts()

@st.cache_resource
def load_market_sources():
    try:
        with open("data/market_sources.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

MARKET_SOURCES = load_market_sources()

def get_market_sources_for_brand(brand: str):
    brand_l = str(brand).lower()
    result = {"resale": "", "maintenance": "", "reliability": "", "depreciation": ""}
    for entry in MARKET_SOURCES:
        brands = [b.lower() for b in entry.get("brands", [])]
        if brand_l in brands or "all" in brands:
            topic = entry["topic"].lower()
            if topic in result:
                result[topic] += f"{entry['text']} (Source: {entry['source']}) "
    return result

def parse_numeric(value):
    if value in [None, "null", "-", "None"]: return None
    if isinstance(value, (int, float)): return float(value)
    v = str(value).lower().replace("kms", "").replace("km", "").replace(",", "").replace("$", "").strip()
    nums = re.findall(r"\d+", v)
    return float("".join(nums)) if nums else None

def safe_json_parse(text: str):
    try:
        t = (text or "").strip()
        if t.startswith("```"):
            t = t.split("```")[1]
            if t.startswith("json"): t = t[4:]
        return json.loads(t.strip())
    except: return {}

def lookup_new_price(brand, model):
    bn, mn = str(brand).lower().replace(" ", ""), str(model).lower().replace(" ", "")
    bm_copy = bm.copy()
    bm_copy['bn'] = bm_copy['Brand'].str.lower().str.replace(" ", "")
    bm_copy['mn'] = bm_copy['Model'].str.lower().str.replace(" ", "")
    match = bm_copy[(bm_copy['bn'] == bn) & (bm_copy['mn'].str.contains(mn))]
    if not match.empty: return float(match.iloc[0]["New_Price_bm"])
    b_copy = b.copy()
    b_copy['bn'] = b_copy['Brand'].str.lower().str.replace(" ", "")
    match_b = b_copy[b_copy['bn'] == bn]
    return float(match_b.iloc[0]["New_Price_b"]) if not match_b.empty else np.nan

# =====================================================
# SESSION STATE
# =====================================================
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "vehicle_data" not in st.session_state: 
    st.session_state.vehicle_data = {"Brand": "Toyota", "Model": "Corolla", "Year": 2020, "Kilometres": 50000, "Listed Price": 25000}
if "confirmed_plausibility" not in st.session_state: st.session_state.confirmed_plausibility = False

# =====================================================
# UI LAYOUT
# =====================================================
st.title("🤖 AI Car Deal Advisor")

tab1, tab2 = st.tabs(["📊 Price Estimator", "💡 AI Deal Analysis"])

with tab1:
    st.subheader("Manual Input & Refinement")
    
    # Grid Layout for Manual Inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Brand Dropdown (Getting unique brands from lookup)
        available_brands = sorted(brand_model_lookup['Brand'].unique())
        brand_idx = available_brands.index(st.session_state.vehicle_data["Brand"]) if st.session_state.vehicle_data["Brand"] in available_brands else 0
        new_brand = st.selectbox("Brand", available_brands, index=brand_idx)
        
        # Model Dropdown (Filter based on Brand)
        available_models = sorted(brand_model_lookup[brand_model_lookup['Brand'] == new_brand]['Model'].unique())
        model_idx = available_models.index(st.session_state.vehicle_data["Model"]) if st.session_state.vehicle_data["Model"] in available_models else 0
        new_model = st.selectbox("Model", available_models, index=model_idx)

    with col2:
        new_year = st.slider("Year", 2000, 2026, int(st.session_state.vehicle_data["Year"]))
        new_kms = st.number_input("Kilometres", value=int(st.session_state.vehicle_data["Kilometres"]), step=1000)

    with col3:
        new_price = st.number_input("Listed Price (AUD)", value=int(st.session_state.vehicle_data["Listed Price"]), step=500)
        if st.button("Reset All Data"):
            st.session_state.vehicle_data = {"Brand": "Toyota", "Model": "Corolla", "Year": 2020, "Kilometres": 50000, "Listed Price": 25000}
            st.rerun()

    # Sync UI changes back to Session State
    st.session_state.vehicle_data.update({
        "Brand": new_brand, "Model": new_model, "Year": new_year, 
        "Kilometres": new_kms, "Listed Price": new_price
    })

    st.divider()
    st.write("### 💬 Chat Assistant")
    st.caption("Paste a listing here to auto-fill the sliders and boxes above.")

    # Display Chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"].replace("$", "\$"))

    # Smart Chat Input
    user_input = st.chat_input("e.g., 'I actually meant 40,000km' or paste a full ad description")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        extract_prompt = f"Current Data: {st.session_state.vehicle_data}\nMessage: {user_input}\nUpdate JSON. Return ONLY JSON."
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Update car data JSON based on user input. Return ONLY JSON."},
                      {"role": "user", "content": extract_prompt}],
            temperature=0
        )
        extracted = safe_json_parse(resp.choices[0].message.content)
        if extracted:
            st.session_state.vehicle_data.update({k: v for k, v in extracted.items() if v is not None})
        st.rerun()

with tab2:
    v_curr = st.session_state.vehicle_data
    brand, model = str(v_curr["Brand"]), str(v_curr["Model"])
    year, kms, price = v_curr["Year"], v_curr["Kilometres"], v_curr["Listed Price"]

    # Price Calculation
    new_p = lookup_new_price(brand, model)
    if np.isnan(new_p):
        st.error(f"⚠️ Could not find base pricing for {brand} {model}.")
    else:
        age = 2026 - year
        X = pd.DataFrame([{
            "Age": age, "log_km": np.log1p(kms), "Brand": brand, "Model": model,
            "FuelConsumption": 8.0, "CylindersinEngine": 4, "Seats": 5,
            "age_kilometer_interaction": (age * kms) / 10000, "UsedOrNew": "USED",
            "DriveType": "AWD" if "SUV" in model else "FWD", 
            "BodyType": "SUV" if "SUV" in model else "Sedan", 
            "Transmission": "Automatic", "FuelType": "Gasoline"
        }])
        
        pred = np.exp(pipe.predict(X)[0]) * new_p
        gap = ((price - pred) / pred) * 100
        
        # Verdict Header
        if gap < -15: verdict, color = "VERY LOW!", "orange"
        elif gap < -5: verdict, color = "BARGAIN", "green"
        elif gap <= 5: verdict, color = "FAIR PRICED", "blue"
        else: verdict, color = "OVER PRICED!", "red"

        st.markdown(f"## Verdict: :{color}[{verdict}]")
        
        # Narrative Report
        m_ctx = get_market_sources_for_brand(brand)
        report_prompt = f"""
        Listing: {year} {brand} {model}, {kms}km, \${price}. 
        Prediction: \${pred:.0f}. Gap: {gap:.1f}%. 
        Verdict: {verdict}.
        Market Context: {m_ctx}
        
        Write a 3-section professional report. Escape \$ symbols. Cite sources.
        """

        with st.spinner("Analyzing Deal..."):
            report = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": report_prompt}],
                temperature=0.7
            ).choices[0].message.content
            
            st.write(report.replace("$", "\$"))
