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
st.set_page_config(
    page_title="Car Price Deal Advisor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OPENAI_API_KEY")
    st.stop()

client = OpenAI(api_key=api_key)

# =====================================================
# HELPERS & DATA LOADERS
# =====================================================
def parse_numeric(value):
    if value is None: return None
    if isinstance(value, (int, float)): return float(value)
    if isinstance(value, str):
        v = value.lower().strip().replace("kms", "").replace("km", "").replace(",", "").replace("$", "")
        nums = re.findall(r"\d+", v)
        return float("".join(nums)) if nums else None
    return None

def safe_json_parse(text: str):
    t = (text or "").strip()
    if "```json" in t:
        t = t.split("```json")[1].split("```")[0]
    elif "```" in t:
        t = t.split("```")[1].split("```")[0]
    try:
        return json.loads(t.strip())
    except:
        return {}

@st.cache_resource
def load_artifacts():
    # Note: Ensure these paths are correct for your local setup
    pipe = joblib.load("models/final_price_pipe.joblib")
    bm = pd.read_csv("models/new_price_lookup_bm.csv")
    b = pd.read_csv("models/new_price_lookup_b.csv")
    lookup = pd.read_csv("models/brand_model_lookup_50.csv")
    
    for df_ in (bm, b, lookup):
        for c in ("Brand", "Model"):
            if c in df_.columns:
                df_[c] = df_[c].astype(str).str.strip()
    return pipe, bm, b, lookup

pipe, bm, b, brand_model_lookup = load_artifacts()

@st.cache_resource
def load_market_sources():
    with open("data/market_sources.json", "r", encoding="utf-8") as f:
        return json.load(f)

MARKET_SOURCES = load_market_sources()

def get_market_sources_for_brand(brand: str):
    brand_l = str(brand).lower()
    result = {"resale": [], "maintenance": [], "reliability": [], "depreciation": []}
    for entry in MARKET_SOURCES:
        brands = [b.lower() for b in entry.get("brands", [])]
        if brand_l in brands or "all" in brands:
            topic = entry["topic"].lower()
            if topic in result:
                result[topic].append(f"{entry['text']} (Source: {entry['source']})")
    return result

def lookup_new_price(brand, model):
    bn, mn = str(brand).lower().replace(" ", ""), str(model).lower().replace(" ", "")
    bm_copy = bm.copy()
    bm_copy['bn'] = bm_copy['Brand'].str.lower().str.replace(" ", "")
    bm_copy['mn'] = bm_copy['Model'].str.lower().str.replace(" ", "")
    
    match = bm_copy[(bm_copy['bn'] == bn) & (bm_copy['mn'].str.contains(mn))]
    if not match.empty:
        return float(match.iloc[0]["New_Price_bm"])
    
    b_copy = b.copy()
    b_copy['bn'] = b_copy['Brand'].str.lower().str.replace(" ", "")
    match_b = b_copy[b_copy['bn'] == bn]
    return float(match_b.iloc[0]["New_Price_b"]) if not match_b.empty else np.nan

def validate_data_plausibility(brand, model, year, kms, price):
    warnings = []
    age = max(1, 2026 - year)
    km_per_year = kms / age

    if year >= 2022 and price < 8000:
        warnings.append(f"⚠️ **Price Alert:** AU ${price:,} for a {year} vehicle is significantly below market value. Check for scams.")
    if km_per_year > 25000:
        warnings.append(f"🏎️ **High Usage:** This car averages {int(km_per_year):,} km/year (2x Aus avg).")
    if year <= 2024 and kms < 1000:
        warnings.append(f"🔍 **Low Kms:** Only {kms:,} km on a {year} model. Verify odometer.")
    return warnings

# =====================================================
# SESSION STATE
# =====================================================
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "vehicle_data" not in st.session_state: 
    st.session_state.vehicle_data = {"Brand": None, "Model": None, "Year": None, "Kilometres": None, "Listed Price": None}
if "confirmed_plausibility" not in st.session_state: st.session_state.confirmed_plausibility = False

# =====================================================
# UI LAYOUT
# =====================================================
tab1, tab2 = st.tabs(["Price Estimator", "AI Deal Advisor"])

# TAB 1 - Estimator (Keeping original logic)
with tab1:
    st.header("Quick Market Estimate")
    col1, col2 = st.columns(2)
    with col1:
        sel_brand = st.selectbox("Brand", sorted(brand_model_lookup["Brand"].unique()))
        sel_year = st.number_input("Year", 2000, 2026, 2022)
    with col2:
        models = brand_model_lookup[brand_model_lookup["Brand"] == sel_brand]["Model"].unique()
        sel_model = st.selectbox("Model", sorted(models))
        sel_km = st.number_input("Kilometres", 0, 400000, 30000)
    if st.button("Calculate Estimate"):
        age = 2026 - sel_year
        new_p = lookup_new_price(sel_brand, sel_model)
        if not np.isnan(new_p):
            X = pd.DataFrame([{"Age": age, "log_km": np.log1p(sel_km), "Brand": sel_brand, "Model": sel_model,
                               "FuelConsumption": 7.5, "CylindersinEngine": 4, "Seats": 5,
                               "age_kilometer_interaction": (age * sel_km) / 10000, "UsedOrNew": "USED",
                               "DriveType": "FWD", "BodyType": "Sedan", "Transmission": "Automatic", "FuelType": "Gasoline"}])
            retention = np.exp(pipe.predict(X)[0])
            st.metric("Market Valuation", f"AU ${(retention * new_p):,.0f}")

# TAB 2 - Deal Advisor (THE FIX)
with tab2:
    st.header("AI Deal Advisor")
    
    # 📋 LIVE DASHBOARD
    st.write("### 📋 Current Data Extraction")
    v = st.session_state.vehicle_data
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Brand", v["Brand"] or "-")
    d2.metric("Model", v["Model"] or "-")
    d3.metric("Year", v["Year"] or "-")
    d4.metric("Km", f"{v['Kilometres']:,}" if v["Kilometres"] else "-")
    d5.metric("Price", f"${v['Listed Price']:,}" if v["Listed Price"] else "-")

    if st.button("Reset Everything"):
        st.session_state.chat_history = []
        st.session_state.vehicle_data = {"Brand": None, "Model": None, "Year": None, "Kilometres": None, "Listed Price": None}
        st.session_state.confirmed_plausibility = False
        st.rerun()

    # Display Chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input Handling
    user_input = st.chat_input("Enter details or corrections (e.g., 'oops 20000km')")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Extraction logic
        extract_prompt = f"Current: {st.session_state.vehicle_data}\nNew Input: {user_input}\nUpdate the JSON. Return ONLY JSON."
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You extract car data. If the user corrects a field, update it. Return JSON."},
                      {"role": "user", "content": extract_prompt}],
            temperature=0
        )
        new_extracted = safe_json_parse(resp.choices[0].message.content)
        for key in st.session_state.vehicle_data:
            if new_extracted.get(key):
                st.session_state.vehicle_data[key] = new_extracted[key]
        
        st.session_state.confirmed_plausibility = False
        st.rerun()

    # --- AUTO-PROCESSOR (Runs outside the user_input block) ---
    required = ["Brand", "Model", "Year", "Kilometres", "Listed Price"]
    v_curr = st.session_state.vehicle_data
    missing = [r for r in required if v_curr[r] is None]

    if not any(v_curr.values()):
        st.info("Paste your listing to begin.")
    elif missing:
        st.warning(f"Waiting for: {', '.join(missing)}")
    else:
        # Step 1: Plausibility
        brand, model = str(v_curr["Brand"]), str(v_curr["Model"])
        year, kms = parse_numeric(v_curr["Year"]), parse_numeric(v_curr["Kilometres"])
        price = parse_numeric(v_curr["Listed Price"])
        
        warnings = validate_data_plausibility(brand, model, year, kms, price)
        if warnings and not st.session_state.confirmed_plausibility:
            with st.chat_message("assistant"):
                st.error("### 🛑 Plausibility Warning")
                for w in warnings: st.write(w)
                if st.button("Details are correct, proceed"):
                    st.session_state.confirmed_plausibility = True
                    st.rerun()
            st.stop()

        # Step 2: Final Report (Triggered if all data exists and is confirmed)
        # Check if we already have a report in history to avoid infinite loops
        last_msg = st.session_state.chat_history[-1] if st.session_state.chat_history else {}
        if last_msg.get("role") != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Generating Market Analysis..."):
                    new_p = lookup_new_price(brand, model)
                    if np.isnan(new_p):
                        st.write("Unable to find new price for prediction.")
                    else:
                        age = 2026 - year
                        X = pd.DataFrame([{"Age": age, "log_km": np.log1p(kms), "Brand": brand, "Model": model,
                                           "FuelConsumption": 7.5, "CylindersinEngine": 4, "Seats": 5,
                                           "age_kilometer_interaction": (age * kms) / 10000, "UsedOrNew": "USED",
                                           "DriveType": "FWD", "BodyType": "Sedan", "Transmission": "Automatic", "FuelType": "Gasoline"}])
                        pred = np.exp(pipe.predict(X)[0]) * new_p
                        gap = ((price - pred) / pred) * 100
                        m_ctx = get_market_sources_for_brand(brand)

                        prompt = f"Analyze {year} {brand} {model}. Price Gap: {gap:+.1f}%. Context: {m_ctx}. Use 3 sections: Price, Market Intelligence, and Buyer Advice. Cite sources."
                        report = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7
                        ).choices[0].message.content
                        
                        st.markdown(report)
                        st.session_state.chat_history.append({"role": "assistant", "content": report})