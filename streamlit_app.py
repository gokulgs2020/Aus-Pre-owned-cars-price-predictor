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
# HELPERS & ARTIFACTS
# =====================================================
@st.cache_resource
def load_artifacts():
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
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
    return json.loads(t.strip())

def validate_data_plausibility(brand, model, year, kms, price):
    warnings = []
    age = max(1, 2026 - year) 
    km_per_year = kms / age
    
    if year >= 2022 and price < 10000:
        warnings.append(f"⚠️ **Price Alert:** AU ${price:,} for a {year} model is suspiciously low. Verify if this is a scam.")
    if km_per_year > 25000:
        warnings.append(f"🏎️ **High Usage:** This {brand} has averaged {int(km_per_year):,} km/year; more than 2x the Australian avg (~13,000km/yr; Source: ABS).")
    if year <= 2024 and kms < 1500:
        warnings.append(f"🔍 **Suspiciously Low Kms:** Only {kms:,} km on a {year} model. Please verify the odometer.")
    return warnings

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
    st.session_state.vehicle_data = {"Brand": None, "Model": None, "Year": None, "Kilometres": None, "Listed Price": None}
if "confirmed_plausibility" not in st.session_state: st.session_state.confirmed_plausibility = False

# =====================================================
# UI LAYOUT
# =====================================================
st.title("🤖 Car Price Advisor")

tab1, tab2 = st.tabs(["Price Estimator", "AI Deal Advisor"])

# -----------------------------------------------------
# TAB 1 - QUICK ESTIMATOR
# -----------------------------------------------------
with tab1:
    st.header("Quick Market Estimate")
    col1, col2 = st.columns(2)
    with col1:
        sel_brand = st.selectbox("Brand", sorted(brand_model_lookup["Brand"].unique()), key="tab1_brand")
        sel_year = st.number_input("Year", 2000, 2026, 2022, key="tab1_year")
    with col2:
        models = brand_model_lookup[brand_model_lookup["Brand"] == sel_brand]["Model"].unique()
        sel_model = st.selectbox("Model", sorted(models), key="tab1_model")
        sel_km = st.number_input("Kilometres",min_value=0,max_value=400000,value=30000,step=5000,key="tab1_km")
    
    if st.button("Calculate Estimate"):
        age = 2026 - sel_year
        new_p = lookup_new_price(sel_brand, sel_model)
        if not np.isnan(new_p):
            X_manual = pd.DataFrame([{"Age": age, "log_km": np.log1p(sel_km), "Brand": sel_brand, "Model": sel_model,
                                      "FuelConsumption": 7.5, "CylindersinEngine": 4, "Seats": 5,
                                      "age_kilometer_interaction": (age * sel_km) / 10000, "UsedOrNew": "USED",
                                      "DriveType": "FWD", "BodyType": "Sedan", "Transmission": "Automatic", "FuelType": "Gasoline"}])
            retention = np.exp(pipe.predict(X_manual)[0])
            st.metric("Market Valuation", f"AU ${(retention * new_p):,.0f}")
        else:
            st.warning("New price baseline not found for this selection.")

# -----------------------------------------------------
# TAB 2 - AI DEAL ADVISOR
# -----------------------------------------------------
with tab2:
    st.header("AI Deal Advisor")
    
    # Live Dashboard
    st.write("### 📋 Current Data Extraction")
    v = st.session_state.vehicle_data
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Brand", v["Brand"] or "-")
    d2.metric("Model", v["Model"] or "-")
    d3.metric("Year", v["Year"] or "-")
    d4.metric("Km", f"{v['Kilometres']:,}" if v["Kilometres"] else "-")
    d5.metric("Price", f"${v['Listed Price']:,}" if v["Listed Price"] else "-")

    if st.button("Reset Advisor"):
        st.session_state.chat_history = []
        st.session_state.vehicle_data = {k: None for k in st.session_state.vehicle_data}
        st.session_state.confirmed_plausibility = False
        st.rerun()

    # Display Chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 1. SMART EXTRACTION
    user_input = st.chat_input("Enter details (e.g., '2022 Toyota Corolla 30000km $25000')")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        extract_prompt = f"Current Data: {st.session_state.vehicle_data}\nMessage: {user_input}\nUpdate JSON. Return ONLY JSON."
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a car data assistant. Update the JSON based on the user's message. Return ONLY JSON."},
                      {"role": "user", "content": extract_prompt}],
            temperature=0
        )
        new_data = safe_json_parse(resp.choices[0].message.content)
        for key in st.session_state.vehicle_data:
            if new_data.get(key) is not None: 
                st.session_state.vehicle_data[key] = new_data[key]
        st.session_state.confirmed_plausibility = False
        st.rerun()

    # 2. AUTO-PROCESSOR
    v_curr = st.session_state.vehicle_data
    required = ["Brand", "Model", "Year", "Kilometres", "Listed Price"]
    missing = [r for r in required if v_curr[r] is None or v_curr[r] == "-"]

    if not any(val for val in v_curr.values() if val):
        st.info("Paste your car listing details below to start.")
    elif missing:
        st.warning(f"I'm still missing: **{', '.join(missing)}**")
    else:
        # 3. PLAUSIBILITY CHECK
        brand, model = str(v_curr["Brand"]), str(v_curr["Model"])
        year, kms = parse_numeric(v_curr["Year"]), parse_numeric(v_curr["Kilometres"])
        price = parse_numeric(v_curr["Listed Price"])
        
        warnings = validate_data_plausibility(brand, model, year, kms, price)
        
        if warnings and not st.session_state.confirmed_plausibility:
            with st.chat_message("assistant"):
                st.error("### 🛑 Check these details!")
                for w in warnings: st.write(w)
                if st.button("Yes, these details are correct"):
                    st.session_state.confirmed_plausibility = True
                    st.rerun()
            st.stop()

        # 4. VERDICT & REPORT
        if st.session_state.chat_history and st.session_state.chat_history[-1].get("role") == "user":
            with st.chat_message("assistant"):
                new_p = lookup_new_price(brand, model)
                if np.isnan(new_p):
                    st.write("⚠️ Baseline market price not found for this model.")
                else:
                    age = 2026 - year
                    X_ai = pd.DataFrame([{"Age": age, "log_km": np.log1p(kms), "Brand": brand, "Model": model,
                                       "FuelConsumption": 7.5, "CylindersinEngine": 4, "Seats": 5,
                                       "age_kilometer_interaction": (age * kms) / 10000, "UsedOrNew": "USED",
                                       "DriveType": "FWD", "BodyType": "Sedan", "Transmission": "Automatic", "FuelType": "Gasoline"}])
                    pred = np.exp(pipe.predict(X_ai)[0]) * new_p
                    gap = ((price - pred) / pred) * 100
                    
                    if gap < -15: verdict, color = "VERY LOW!", "orange"
                    elif gap < -5: verdict, color = "BARGAIN", "green"
                    elif gap <= 5: verdict, color = "FAIR PRICED", "blue"
                    else: verdict, color = "OVER PRICED!", "orange"

                    st.markdown(f"### Verdict on the listed price: :{color}[{verdict}]")
                    
                    m_ctx = get_market_sources_for_brand(brand)
                    
                    report_prompt = f"""
                    Analyze this listing: {year} {brand} {model}, {kms:,}km, ${price:,.0f}.
                    Our prediction: \${pred:,.0f} (Gap: {gap:.1f}%). 
                    Verdict: {verdict}.

                    MARKET DATA (CITE SOURCES FROM THIS DATA):
                    Resale: {m_ctx['resale']}
                    Reliability: {m_ctx['reliability']}
                    Maintenance: {m_ctx['maintenance']}
                    Depreciation: {m_ctx['depreciation']}

                    Format the final report strictly as follows:

                    1. **Deal Verdict (2 lines):** Explain the verdict by comparing the gap between our predicted price of \${pred:,.0f} and the listed price of \${price:,.0f}.
                    2. **Brand & Model Analysis (3 lines):** Discuss reliability, maintenance, resale value, and depreciation based on the provided market data, explicitly mentioning the sources.
                    3. **Our View on Listed Price (2 lines):** Provide a professional opinion on whether the asking price is justified given the car's specific details.
                    4. **What You Should Do Next (2-3 lines):** Give actionable advice. If overpriced, suggest waiting or comparing similar listings. If suspiciously low, advise checking for accidents, internal mechanics, or title history.

                    Adapt your tone to {brand}. Provide the response in plain conversational English. Always escape dollar signs with a backslash.
                    """

                    with st.spinner("Synthesizing market report..."):
                        report = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[{"role": "system", "content": "You are a professional Australian auto-analyst. Use only the provided market data sources."},
                                      {"role": "user", "content": report_prompt}],
                            temperature=0.7
                        ).choices[0].message.content
                        
                        st.markdown(report)
                        st.session_state.chat_history.append({"role": "assistant", "content": f"**Verdict: {verdict}**\n\n{report}"})