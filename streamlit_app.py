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

# Pydantic schema for the Critic Agent (Ensures valid JSON output)
class AuditReport(BaseModel):
    is_factual: bool
    discrepancies: List[str]
    suggested_fix: str

# =====================================================
# 2. DATA ARTIFACTS & HELPERS
# =====================================================
@st.cache_resource
def load_artifacts():
    # Paths assumed based on your project structure
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
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"): t = t[4:]
    return json.loads(t.strip())

def validate_data_plausibility(brand, model, year, kms, price):
    warnings = []
    age = max(1, 2026 - year) # Current year fixed at 2026
    km_per_year = kms / age
    
    if year >= 2022 and price < 10000:
        warnings.append(f"⚠️ **Price Alert:** AU \${price:,} for a {year} model is suspiciously low.")
    if km_per_year > 25000:
        warnings.append(f"🏎️ **High Usage:** Averaging {int(km_per_year):,} km/year. (AU average is ~13,000km).")
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
# 3. CRITIC AGENT (GROUNDING CHECK)
# =====================================================
def run_critic_audit(ml_data, ai_draft_text):
    """Verifies that the AI text matches the ML model's hard data."""
    critic_system_prompt = f"""
    You are a strict Data Auditor. Compare the FACTS against the AI DRAFT.
    FACTS:
    - Listed: \${ml_data['price']:,}
    - Predicted: \${ml_data['pred']:,}
    - Gap: {ml_data['gap']:.1f}% ({ml_data['verdict']})
    
    AUDIT RULES:
    1. If numbers in the DRAFT differ from FACTS, it is a FAIL.
    2. Every single dollar sign MUST be escaped with a backslash (\\$).
    3. Ensure three sections are separated exactly by '---'.
    """
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": critic_system_prompt},
            {"role": "user", "content": ai_draft_text}
        ],
        response_format=AuditReport,
    )
    return completion.choices[0].message.parsed

# =====================================================
# 4. SESSION STATE & UI
# =====================================================
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "vehicle_data" not in st.session_state: 
    st.session_state.vehicle_data = {"Brand": None, "Model": None, "Year": None, "Kilometres": None, "Listed Price": None}
if "confirmed_plausibility" not in st.session_state: st.session_state.confirmed_plausibility = False

st.write("### 📋 Current Vehicle Profile")
v = st.session_state.vehicle_data
cols = st.columns(5)
cols[0].metric("Brand", v["Brand"] or "-")
cols[1].metric("Model", v["Model"] or "-")
cols[2].metric("Year", v["Year"] or "-")
cols[3].metric("Km", f"{v['Kilometres']:,}" if v["Kilometres"] else "-")
cols[4].metric("Price", f"\${v['Listed Price']:,}" if v["Listed Price"] else "-")

if st.button("Clear & Reset"):
    st.session_state.chat_history = []
    st.session_state.vehicle_data = {k: None for k in st.session_state.vehicle_data}
    st.session_state.confirmed_plausibility = False
    st.rerun()

# Display Chat History
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =====================================================
# 5. EXECUTION LOGIC
# =====================================================
user_input = st.chat_input("Enter vehicle details or corrections...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # 5a. Data Extraction
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Update the car data JSON based on user input. Return ONLY JSON."},
                  {"role": "user", "content": f"Current: {st.session_state.vehicle_data}\nInput: {user_input}"}],
        temperature=0
    )
    new_data = safe_json_parse(resp.choices[0].message.content)
    for key in st.session_state.vehicle_data:
        if new_data.get(key) is not None: 
            st.session_state.vehicle_data[key] = new_data[key]
    st.session_state.confirmed_plausibility = False
    st.rerun()

v_curr = st.session_state.vehicle_data
required_fields = ["Brand", "Model", "Year", "Kilometres", "Listed Price"]
missing = [f for f in required_fields if v_curr[f] is None]

if not any(v_curr.values()):
    st.info("👋 Welcome! Please paste a car listing or type the details to get started.")
elif missing:
    st.warning(f"Waiting for: **{', '.join(missing)}**")
else:
    # 5b. Analysis Logic
    brand, model = str(v_curr["Brand"]), str(v_curr["Model"])
    year, kms = parse_numeric(v_curr["Year"]), parse_numeric(v_curr["Kilometres"])
    price = parse_numeric(v_curr["Listed Price"])
    
    # Plausibility Guardrail
    warnings = validate_data_plausibility(brand, model, year, kms, price)
    if warnings and not st.session_state.confirmed_plausibility:
        with st.chat_message("assistant"):
            st.error("### 🛑 Data Anomaly Detected")
            for w in warnings: st.write(w)
            if st.button("Details are correct, proceed"):
                st.session_state.confirmed_plausibility = True
                st.rerun()
        st.stop()

    # Calculation & Report Generation
    if st.session_state.chat_history[-1]["role"] == "user":
        with st.chat_message("assistant"):
            new_p = lookup_new_price(brand, model)
            if np.isnan(new_p):
                st.write("⚠️ Baseline price for this model not found in Australia.")
            else:
                # ML Prediction
                age = 2026 - year
                X = pd.DataFrame([{"Age": age, "log_km": np.log1p(kms), "Brand": brand, "Model": model,
                                   "FuelConsumption": 7.5, "CylindersinEngine": 4, "Seats": 5,
                                   "age_kilometer_interaction": (age * kms) / 10000, "UsedOrNew": "USED",
                                   "DriveType": "FWD", "BodyType": "Sedan", "Transmission": "Automatic", "FuelType": "Gasoline"}])
                pred = np.exp(pipe.predict(X)[0]) * new_p
                gap = ((price - pred) / pred) * 100
                
                # Verdict Styling
                if gap < -15: verdict, color = "VERY LOW!", "orange"
                elif gap < -5: verdict, color = "BARGAIN", "green"
                elif gap <= 5: verdict, color = "FAIRLY PRICED", "blue"
                else: verdict, color = "OVERPRICED!", "orange"

                st.markdown(f"### Verdict: :{color}[{verdict}]")
                
                # Generate Narrative with Critic Agent Audit
                m_ctx = get_market_sources_for_brand(brand)
                report_prompt = f"""
                Analyze: {year} {brand} {model}, {kms:,}km, \${price:,.0f}.
                Model Prediction: \${pred:,.0f} (Gap: {gap:.1f}%). 
                
                MARKET CONTEXT: {m_ctx}
                
                INSTRUCTIONS:
                1. Split into 3 sections using '---'.
                2. Section headers: Verdict Summary, Brand/Model Insights, Resale Parameters.
                3. Escape all \$ symbols with a backslash (\\$).
                """

                with st.spinner("🔍 Auditing Market Data..."):
                    raw_report = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "system", "content": "Professional Australian Auto-Analyst."},
                                  {"role": "user", "content": report_prompt}],
                        temperature=0.7
                    ).choices[0].message.content
                    
                    # Run the Critic Audit
                    ml_data = {"price": price, "pred": pred, "gap": round(gap, 1), "verdict": verdict}
                    audit = run_critic_audit(ml_data, raw_report)
                    
                    final_report = audit.suggested_fix if not audit.is_factual else raw_report
                    
                    # Split and Display with UI Headers
                    sections = final_report.split("---")
                    labels = ["Verdict Summary", "What we found about the brand / model?", "Comments on key resale parameters"]
                    
                    for i, section in enumerate(sections):
                        if i < len(labels):
                            st.markdown(f"#### {labels[i]}")
                            st.write(section.strip())
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": final_report})