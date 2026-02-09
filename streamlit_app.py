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

def parse_numeric(value):
    if value in [None, "null", "-"]: return None
    if isinstance(value, (int, float)): return float(value)
    v = str(value).lower().replace("kms", "").replace("km", "").replace(",", "").replace("$", "").strip()
    nums = re.findall(r"\d+", v)
    return float("".join(nums)) if nums else None

def validate_data_plausibility(brand, model, year, kms, price):
    warnings = []
    age = max(1, 2026 - year)
    km_per_year = kms / age
    
    # 1. Price Floor Check (Scam Detection)
    if year >= 2022 and price < 8000:
        warnings.append(f"⚠️ **Price Alert:** AU ${price:,} for a {year} model is extremely low. This is a common scam indicator.")
    
    # 2. Mileage Checks
    if km_per_year > 35000:
        warnings.append(f"🏎️ **High Usage:** This car averages {int(km_per_year):,} km/year (well above the 13k km Aus avg). Source : abs.gov.au")
    if year <= 2024 and kms < 1000:
        warnings.append(f"🔍 **Suspiciously Low Kms:** Only {kms:,} km on a {year} model. Verify odometer accuracy.")
    
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
st.title("🤖 AI Deal Advisor")

# 📋 LIVE DASHBOARD (Real-time updates)
st.write("### 📋 Current Extraction Status")
v = st.session_state.vehicle_data
d1, d2, d3, d4, d5 = st.columns(5)
d1.metric("Brand", v["Brand"] or "-")
d2.metric("Model", v["Model"] or "-")
d3.metric("Year", v["Year"] or "-")
d4.metric("Km", f"{v['Kilometres']:,}" if v["Kilometres"] else "-")
d5.metric("Price", f"${v['Listed Price']:,}" if v["Listed Price"] else "-")

if st.button("Reset Everything"):
    st.session_state.chat_history = []
    st.session_state.vehicle_data = {k: None for k in st.session_state.vehicle_data}
    st.session_state.confirmed_plausibility = False
    st.rerun()

# Display Chat History
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 1. SMART EXTRACTION
user_input = st.chat_input("Enter details (e.g., 'oops 20000km')")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    extract_prompt = f"Current Data: {st.session_state.vehicle_data}\nMessage: {user_input}\nUpdate JSON fields. Return ONLY JSON."
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You extract car data and handle corrections. Return ONLY JSON."},
                  {"role": "user", "content": extract_prompt}],
        temperature=0
    )
    new_data = json.loads(resp.choices[0].message.content)
    for key in st.session_state.vehicle_data:
        if new_data.get(key) is not None: 
            st.session_state.vehicle_data[key] = new_data[key]
    
    # If data changes, they must confirm plausibility again
    st.session_state.confirmed_plausibility = False
    st.rerun()

# 2. AUTO-PROCESSOR
v_curr = st.session_state.vehicle_data
required = ["Brand", "Model", "Year", "Kilometres", "Listed Price"]
missing = [r for r in required if v_curr[r] is None or v_curr[r] == "-"]

if not any(val for val in v_curr.values() if val):
    st.info("Paste your listing details to begin. I'll automatically check for scams and typos.")
elif missing:
    st.warning(f"Awaiting data for: **{', '.join(missing)}**")
else:
    # 3. THE PLAUSIBILITY CHECK (Reintegrated)
    brand, model = str(v_curr["Brand"]), str(v_curr["Model"])
    year, kms = parse_numeric(v_curr["Year"]), parse_numeric(v_curr["Kilometres"])
    price = parse_numeric(v_curr["Listed Price"])
    
    warnings = validate_data_plausibility(brand, model, year, kms, price)
    
    if warnings and not st.session_state.confirmed_plausibility:
        with st.chat_message("assistant"):
            st.error("### 🛑 Wait, check these details!")
            for w in warnings: st.write(w)
            if st.button("Yes, these details are correct"):
                st.session_state.confirmed_plausibility = True
                st.rerun()
        st.stop() # Halts AI until confirmation is clicked

    # 4. FINAL VERDICT & REPORT
    if st.session_state.chat_history and st.session_state.chat_history[-1].get("role") == "user":
        with st.chat_message("assistant"):
            new_p = lookup_new_price(brand, model)
            if np.isnan(new_p):
                st.write("⚠️ Market baseline not found for this model.")
            else:
                age = 2026 - year
                # Predict price using your ML pipe
                X = pd.DataFrame([{"Age": age, "log_km": np.log1p(kms), "Brand": brand, "Model": model,
                                   "FuelConsumption": 7.5, "CylindersinEngine": 4, "Seats": 5,
                                   "age_kilometer_interaction": (age * kms) / 10000, "UsedOrNew": "USED",
                                   "DriveType": "FWD", "BodyType": "Sedan", "Transmission": "Automatic", "FuelType": "Gasoline"}])
                pred = np.exp(pipe.predict(X)[0]) * new_p
                gap = ((price - pred) / pred) * 100
                
                # VERDICT LOGIC
                if gap < -15: verdict, color = "VERY LOW", "red"
                elif gap < -5: verdict, color = "BARGAIN", "green"
                elif gap <= 5: verdict, color = "FAIR PRICED", "blue"
                else: verdict, color = "OVER PRICED", "orange"

                st.markdown(f"## Verdict: :{color}[{verdict}]")
                st.write(f"Listed at **AU ${price:,.0f}**, which is **{abs(gap):.1f}% {'above' if gap > 0 else 'below'}** predicted value.")

                with st.spinner("Synthesizing market report..."):
                    report = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": f"Analyze {year} {brand} {model}. Gap: {gap:.1f}%. Verdict: {verdict}. Cite sources."}],
                        temperature=0.7
                    ).choices[0].message.content
                    st.markdown(report)
                    st.session_state.chat_history.append({"role": "assistant", "content": f"**Verdict: {verdict}**\n\n{report}"})