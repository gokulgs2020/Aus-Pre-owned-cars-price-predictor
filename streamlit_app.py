import os
import json
import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime
from openai import OpenAI
from pydantic import BaseModel
from typing import List

# =====================================================
# 1. CONFIG & CLIENT
# =====================================================
st.set_page_config(page_title="AI Car Deal Advisor", layout="wide")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OPENAI_API_KEY. Please set the environment variable.")
    st.stop()

client = OpenAI(api_key=api_key)

class AuditReport(BaseModel):
    is_factual: bool
    discrepancies: List[str]
    suggested_fix: str

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
        st.error(f"Error loading model artifacts: {e}")
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
    results = []
    for entry in MARKET_SOURCES:
        brands = [b.lower() for b in entry.get("brands", [])]
        if brand_l in brands or "all" in brands:
            results.append(f"{entry['text']} (Source: {entry['source']})")
    return "\n".join(results) if results else "No specific market citations found."

def parse_numeric(value):
    if value in [None, "null", "-", "None"]: return None
    if isinstance(value, (int, float)): return float(value)
    v = str(value).lower().replace("kms", "").replace("km", "").replace(",", "").replace("$", "").strip()
    nums = re.findall(r"\d+\.?\d*", v)
    return float(nums[0]) if nums else None

def safe_json_parse(text: str):
    try:
        t = (text or "").strip()
        if t.startswith("```"):
            t = t.split("```")[1]
            if t.startswith("json"): t = t[4:]
        return json.loads(t.strip())
    except:
        return {}

def validate_data_plausibility(brand, model, year, kms, price):
    warnings = []
    age = max(1, 2026 - year)
    km_per_year = kms / age
    if year >= 2022 and price < 10000:
        warnings.append(f"⚠️ **Price Alert:** ${price:,.0f} for a {year} model is suspiciously low.")
    if km_per_year > 35000:
        warnings.append(f"🏎️ **High Usage:** Averaging {int(km_per_year):,} km/year.")
    return warnings

def lookup_new_price(brand, model):
    if bm is None or b is None: return np.nan
    bn, mn = str(brand).lower().replace(" ", ""), str(model).lower().replace(" ", "")
    bm_match = bm[(bm['Brand'].str.lower().str.replace(" ", "") == bn) & 
                  (bm['Model'].str.lower().str.replace(" ", "").str.contains(mn))]
    if not bm_match.empty: return float(bm_match.iloc[0]["New_Price_bm"])
    b_match = b[b['Brand'].str.lower().str.replace(" ", "") == bn]
    return float(b_match.iloc[0]["New_Price_b"]) if not b_match.empty else np.nan

# =====================================================
# 3. SESSION STATE
# =====================================================
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "vehicle_data" not in st.session_state: 
    st.session_state.vehicle_data = {"Brand": None, "Model": None, "Year": None, "Kilometres": None, "Listed Price": None}
if "confirmed_plausibility" not in st.session_state: st.session_state.confirmed_plausibility = False

# =====================================================
# 4. MAIN UI
# =====================================================
st.title("🚗 AI Car Deal Advisor (Australia)")

tab1, tab2 = st.tabs(["📊 Price Estimator", "📂 Market Context"])

with st.sidebar:
    st.write("### 📋 Extracted Details")
    vd = st.session_state.vehicle_data
    st.metric("Brand", vd["Brand"] or "-")
    st.metric("Model", vd["Model"] or "-")
    st.metric("Year", vd["Year"] or "-")
    if st.button("Clear & Reset"):
        st.session_state.clear()
        st.rerun()

# Input logic
user_input = st.chat_input("Enter car details (e.g., 2020 Toyota Kluger 66000kms $50000)")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Update car data JSON. Keys: Brand, Model, Year, Kilometres, Listed Price. Return ONLY JSON."},
                  {"role": "user", "content": f"Current: {st.session_state.vehicle_data}\nInput: {user_input}"}],
        temperature=0
    )
    new_data = safe_json_parse(resp.choices[0].message.content)
    st.session_state.vehicle_data.update({k: v for k, v in new_data.items() if v is not None})
    st.session_state.confirmed_plausibility = False
    st.rerun()

# =====================================================
# 5. ANALYSIS ENGINE (Tab 1)
# =====================================================
with tab1:
    v_curr = st.session_state.vehicle_data
    required = ["Brand", "Model", "Year", "Kilometres", "Listed Price"]
    missing = [f for f in required if v_curr[f] is None]

    if not any(v_curr.values()):
        st.info("👋 Enter vehicle details in the chat box below to start the analysis.")
    elif missing:
        st.warning(f"Waiting for: {', '.join(missing)}")
    else:
        brand, model = str(v_curr["Brand"]), str(v_curr["Model"])
        year, kms, price = parse_numeric(v_curr["Year"]), parse_numeric(v_curr["Kilometres"]), parse_numeric(v_curr["Listed Price"])
        
        # Plausibility Check
        warnings = validate_data_plausibility(brand, model, year, kms, price)
        if warnings and not st.session_state.confirmed_plausibility:
            st.error("### 🛑 Data Anomaly Detected")
            for w in warnings: st.write(w)
            if st.button("Details are correct, proceed"):
                st.session_state.confirmed_plausibility = True
                st.rerun()
            st.stop()

        # ML Prediction
        new_p = lookup_new_price(brand, model)
        if np.isnan(new_p):
            st.error("Baseline pricing unavailable for this model.")
        else:
            age = 2026 - year
            X = pd.DataFrame([{
                "Age": age, "log_km": np.log1p(kms), "Brand": brand, "Model": model,
                "FuelConsumption": 8.5, "CylindersinEngine": 6 if "kluger" in model.lower() else 4,
                "Seats": 7 if "kluger" in model.lower() else 5, "age_kilometer_interaction": (age * kms) / 10000,
                "UsedOrNew": "USED", "DriveType": "AWD", "BodyType": "SUV", "Transmission": "Automatic", "FuelType": "Gasoline"
            }])
            
            pred = np.exp(pipe.predict(X)[0]) * new_p
            gap = ((price - pred) / pred) * 100
            
            # 1. VERDICT HEADER
            if gap < -7: verdict, color = "BARGAIN", "green"
            elif gap > 7: verdict, color = "OVERPRICED", "red"
            else: verdict, color = "FAIRLY PRICED", "blue"
            
            st.markdown(f"## Verdict: :{color}[{verdict}]")
            st.divider()

            # 2. ANALYSIS REPORT
            with st.spinner("Generating Professional Report..."):
                m_ctx = get_market_sources_for_brand(brand)
                report_prompt = f"""
                Analyze: {year} {brand} {model}, {kms:,.0f}km, Listed: ${price:,.0f}.
                Model Prediction: ${pred:,.0f} (Gap: {gap:.1f}%). 
                Market Citations: {m_ctx}
                
                REQUIREMENTS:
                - Use exactly 3 sections separated by '---'.
                - Include specific citations from the provided Market Citations.
                - Escape all $ signs like this: \$ (e.g. \$50,000).
                """
                
                raw_report = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "system", "content": "You are a valuation expert. ALWAYS escape dollar signs with backslashes."},
                              {"role": "user", "content": report_prompt}]
                ).choices[0].message.content

                # Fix potential LaTeX italics issues by escaping $
                final_report = re.sub(r'(?<!\\)\$', r'\$', raw_report)
                
                sections = [s.strip() for s in final_report.split("---") if s.strip()]
                labels = ["📢 Summary", "💡 Market Insights & Citations", "📉 Resale Forecast"]
                
                for i, section in enumerate(sections):
                    if i < len(labels):
                        with st.expander(labels[i], expanded=True):
                            st.markdown(section)

# =====================================================
# 6. MARKET DATA (Tab 2)
# =====================================================
with tab2:
    st.subheader("Reference Market Sources")
    if st.session_state.vehicle_data["Brand"]:
        st.write(get_market_sources_for_brand(st.session_state.vehicle_data["Brand"]))
    else:
        st.info("Market data will appear here once a brand is identified.")