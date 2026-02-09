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
    # Load model and lookup tables
    pipe = joblib.load("models/final_price_pipe.joblib")
    bm = pd.read_csv("models/new_price_lookup_bm.csv")
    b = pd.read_csv("models/new_price_lookup_b.csv")
    lookup = pd.read_csv("models/brand_model_lookup_50.csv")
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
    try:
        t = (text or "").strip()
        if t.startswith("```"):
            t = t.split("```")[1]
            if t.startswith("json"):
                t = t[4:]
        return json.loads(t.strip())
    except:
        return {}

def validate_data_plausibility(brand, model, year, kms, price):
    warnings = []
    age = max(1, 2026 - year) 
    km_per_year = kms / age
    
    if year >= 2022 and price < 10000:
        warnings.append(f"⚠️ **Price Alert:** AU \${price:,} for a {year} model is suspiciously low. Verify if this is a scam.")
    
    if km_per_year > 25000:
        warnings.append(f"🏎️ **High Usage:** This {brand} has averaged {int(km_per_year):,} km/year. (AU avg ~13,000km).")
    
    if year <= 2024 and kms < 1500:
        warnings.append(f"🔍 **Suspiciously Low Kms:** Only {kms:,} km on a {year} model. Verify odometer.")
    
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
st.title("🤖 AI Car Deal Advisor")

# Create Tabs
tab1, tab2 = st.tabs(["📊 Price Estimator & Chat", "💡 AI Deal Analysis"])

with tab1:
    # 📋 LIVE DASHBOARD
    st.write("### 📋 Extracted Data")
    v = st.session_state.vehicle_data
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Brand", v["Brand"] or "-")
    d2.metric("Model", v["Model"] or "-")
    d3.metric("Year", v["Year"] or "-")
    d4.metric("Km", f"{v['Kilometres']:,}" if v["Kilometres"] else "-")
    d5.metric("Price", f"\${v['Listed Price']:,}" if v["Listed Price"] else "-")

    if st.button("Reset Advisor"):
        st.session_state.chat_history = []
        st.session_state.vehicle_data = {k: None for k in st.session_state.vehicle_data}
        st.session_state.confirmed_plausibility = False
        st.rerun()

    # Display Chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"].replace("$", "\$")) # Prevent LaTeX italics

    # SMART EXTRACTION
    user_input = st.chat_input("Enter details (e.g. 2020 Toyota Kluger 66000kms $50000)")
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

with tab2:
    v_curr = st.session_state.vehicle_data
    required = ["Brand", "Model", "Year", "Kilometres", "Listed Price"]
    missing = [r for r in required if v_curr[r] is None or v_curr[r] == "-"]

    if missing:
        st.info(f"Please provide the following details in the Estimator tab: {', '.join(missing)}")
    else:
        brand, model = str(v_curr["Brand"]), str(v_curr["Model"])
        year, kms = parse_numeric(v_curr["Year"]), parse_numeric(v_curr["Kilometres"])
        price = parse_numeric(v_curr["Listed Price"])
        
        # PLAUSIBILITY CHECK
        warnings = validate_data_plausibility(brand, model, year, kms, price)
        if warnings and not st.session_state.confirmed_plausibility:
            st.error("### 🛑 Data Anomaly Detected")
            for w in warnings: st.write(w)
            if st.button("Yes, these details are correct"):
                st.session_state.confirmed_plausibility = True
                st.rerun()
            st.stop()

        # VERDICT CALCULATION
        new_p = lookup_new_price(brand, model)
        if np.isnan(new_p):
            st.warning("⚠️ Baseline market price not found for this model.")
        else:
            age = 2026 - year
            X = pd.DataFrame([{
                "Age": age, "log_km": np.log1p(kms), "Brand": brand, "Model": model,
                "FuelConsumption": 7.5, "CylindersinEngine": 4, "Seats": 5,
                "age_kilometer_interaction": (age * kms) / 10000, "UsedOrNew": "USED",
                "DriveType": "FWD", "BodyType": "Sedan", "Transmission": "Automatic", "FuelType": "Gasoline"
            }])
            
            pred = np.exp(pipe.predict(X)[0]) * new_p
            gap = ((price - pred) / pred) * 100
            
            # 1-Word Verdict with color
            if gap < -15: verdict, color = "VERY LOW!", "orange"
            elif gap < -5: verdict, color = "BARGAIN", "green"
            elif gap <= 5: verdict, color = "FAIR PRICED", "blue"
            else: verdict, color = "OVER PRICED!", "orange"

            st.markdown(f"## Verdict: :{color}[{verdict}]")
            
            # Market Sources Integration
            m_ctx = get_market_sources_for_brand(brand)
            
            report_prompt = f"""
            Analyze this listing: {year} {brand} {model}, {kms:,}km, ${price:,.0f}.
            Our prediction: ${pred:,.0f} (Gap: {gap:.1f}%). 
            Verdict: {verdict}.

            MARKET DATA (MUST CITE SOURCES FROM THIS DATA):
            Resale: {m_ctx['resale']}
            Reliability: {m_ctx['reliability']}
            Maintenance: {m_ctx['maintenance']}
            Depreciation: {m_ctx['depreciation']}

            Write a professional 3-section report. 
            Section 1: Deal Summary.
            Section 2: Brand & Reliability Citations.
            Section 3: Resale & Maintenance Outlook.

            Rules:
            1. Use plain English.
            2. ALWAYS escape dollar signs with backslash (e.g. \$50,000) to avoid LaTeX errors.
            3. Do not use code blocks or symbols.
            4. Integrate citations (e.g. 'Source: REDBOOK') naturally.
            """

            with st.spinner("Synthesizing market report..."):
                report = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": "You are a professional Australian auto-analyst."},
                              {"role": "user", "content": report_prompt}],
                    temperature=0.7
                ).choices[0].message.content
                
                # Output the report
                st.write(report.replace("$", "\$"))
                
                # Save to history if it's new
                if not st.session_state.chat_history or "Verdict:" not in st.session_state.chat_history[-1]["content"]:
                    st.session_state.chat_history.append({"role": "assistant", "content": f"**Verdict: {verdict}**\n\n{report}"})