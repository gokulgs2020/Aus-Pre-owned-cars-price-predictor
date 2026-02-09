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
# 1. CONFIG & CLIENT
# =====================================================
st.set_page_config(page_title="Car Deal Advisor", layout="wide")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OPENAI_API_KEY")
    st.stop()

client = OpenAI(api_key=api_key)

# =====================================================
# 2. DATA ARTIFACTS & HELPERS
# =====================================================
@st.cache_resource
def load_artifacts():
    try:
        pipe = joblib.load("models/final_price_pipe.joblib")
        bm = pd.read_csv("models/new_price_lookup_bm.csv")
        b = pd.read_csv("models/new_price_lookup_b.csv")
        lookup = pd.read_csv("models/brand_model_lookup_50.csv")
        return pipe, bm, b, lookup
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None, None

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

def parse_numeric(value, default=0):
    if value in [None, "null", "-", "None"]: return default
    if isinstance(value, (int, float)): return float(value)
    v = str(value).lower().replace("kms", "").replace("km", "").replace(",", "").replace("$", "").strip()
    nums = re.findall(r"\d+", v)
    return float("".join(nums)) if nums else default

def safe_json_parse(text: str):
    try:
        t = (text or "").strip()
        if t.startswith("```"):
            t = t.split("```")[1]
            if t.startswith("json"): t = t[4:]
        return json.loads(t.strip())
    except: return {}

def lookup_new_price(brand, model):
    if bm is None or b is None: return np.nan
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
# 3. SESSION STATE
# =====================================================
# Default start values to prevent NoneType errors in Sliders
DEFAULT_DATA = {"Brand": "Toyota", "Model": "Corolla", "Year": 2020, "Kilometres": 50000, "Listed Price": 25000}

if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "vehicle_data" not in st.session_state: 
    st.session_state.vehicle_data = DEFAULT_DATA.copy()

# =====================================================
# 4. TAB 1: PRICE ESTIMATOR (INPUTS)
# =====================================================
st.title("🤖 AI Car Deal Advisor")
tab1, tab2 = st.tabs(["📊 Price Estimator", "💡 AI Deal Analysis"])

with tab1:
    st.subheader("Manual Input & Refinement")
    
    if brand_model_lookup is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 1. Brand Selection
            available_brands = sorted(brand_model_lookup['Brand'].unique())
            current_brand = st.session_state.vehicle_data.get("Brand")
            brand_idx = available_brands.index(current_brand) if current_brand in available_brands else 0
            new_brand = st.selectbox("Brand", available_brands, index=brand_idx)
            
            # 2. Model Selection (Dynamic based on Brand)
            models_for_brand = sorted(brand_model_lookup[brand_model_lookup['Brand'] == new_brand]['Model'].unique())
            current_model = st.session_state.vehicle_data.get("Model")
            model_idx = models_for_brand.index(current_model) if current_model in models_for_brand else 0
            new_model = st.selectbox("Model", models_for_brand, index=model_idx)

        with col2:
            # 3. Year & Kilometres
            # We use parse_numeric with a default to ensure the slider never receives None
            val_year = int(parse_numeric(st.session_state.vehicle_data.get("Year"), 2020))
            new_year = st.slider("Year", 2000, 2026, val_year)
            
            val_kms = int(parse_numeric(st.session_state.vehicle_data.get("Kilometres"), 50000))
            new_kms = st.number_input("Kilometres", value=val_kms, step=1000)

        with col3:
            # 4. Price
            val_price = int(parse_numeric(st.session_state.vehicle_data.get("Listed Price"), 25000))
            new_price = st.number_input("Listed Price (AUD)", value=val_price, step=500)
            
            if st.button("Reset to Default"):
                st.session_state.vehicle_data = DEFAULT_DATA.copy()
                st.rerun()

        # Update Session State based on UI
        st.session_state.vehicle_data.update({
            "Brand": new_brand, "Model": new_model, "Year": new_year, 
            "Kilometres": new_kms, "Listed Price": new_price
        })
    else:
        st.error("Model lookup data is missing. Please check your CSV files.")

    st.divider()
    st.write("### 💬 Chat Assistant")
    
    # Display History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"].replace("$", "\$"))

    # Chat Input for Extraction
    user_input = st.chat_input("Paste a listing or type a correction (e.g., 'it's actually 2022')")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Extracting details..."):
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "Update car data JSON based on user input. Return ONLY JSON."},
                          {"role": "user", "content": f"Current: {st.session_state.vehicle_data}\nInput: {user_input}"}],
                temperature=0
            )
            extracted = safe_json_parse(resp.choices[0].message.content)
            if extracted:
                # Filter out None values from AI to preserve current data
                st.session_state.vehicle_data.update({k: v for k, v in extracted.items() if v is not None})
            st.rerun()

# =====================================================
# 5. TAB 2: AI DEAL ANALYSIS (VERDICT)
# =====================================================
with tab2:
    v_curr = st.session_state.vehicle_data
    brand, model = str(v_curr["Brand"]), str(v_curr["Model"])
    year, kms, price = v_curr["Year"], v_curr["Kilometres"], v_curr["Listed Price"]

    new_p = lookup_new_price(brand, model)
    
    if np.isnan(new_p):
        st.warning(f"⚠️ Market Baseline not found for {brand} {model}. Using brand average.")
        # Attempt brand fallback is handled inside lookup_new_price already
    
    if pipe is not None and not np.isnan(new_p):
        age = 2026 - year
        # Feature Engineering for Model
        X = pd.DataFrame([{
            "Age": age, "log_km": np.log1p(kms), "Brand": brand, "Model": model,
            "FuelConsumption": 8.0, "CylindersinEngine": 4, "Seats": 5,
            "age_kilometer_interaction": (age * kms) / 10000, "UsedOrNew": "USED",
            "DriveType": "AWD" if "SUV" in model.upper() else "FWD",
            "BodyType": "SUV" if "SUV" in model.upper() else "Sedan",
            "Transmission": "Automatic", "FuelType": "Gasoline"
        }])
        
        pred = np.exp(pipe.predict(X)[0]) * new_p
        gap = ((price - pred) / pred) * 100
        
        if gap < -15: verdict, color = "VERY LOW!", "orange"
        elif gap < -5: verdict, color = "BARGAIN", "green"
        elif gap <= 5: verdict, color = "FAIR PRICED", "blue"
        else: verdict, color = "OVER PRICED!", "red"

        st.markdown(f"## Verdict: :{color}[{verdict}]")
        st.metric("Predicted Market Value", f"${pred:,.0f}", f"{gap:.1f}% vs Listing", delta_color="inverse")

        m_ctx = get_market_sources_for_brand(brand)
        report_prompt = f"Listing: {year} {brand} {model}, {kms}km, ${price}. Predicted: ${pred:.0f}. Gap: {gap:.1f}%. Context: {m_ctx}. Write a 3-section report. Cite sources. Escape $ symbols."

        with st.spinner("Generating Deep Analysis..."):
            report = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": report_prompt}],
                temperature=0.7
            ).choices[0].message.content
            st.write(report.replace("$", "\$"))
    else:
        st.info("Ensure all data is correct in the first tab to see the analysis.")