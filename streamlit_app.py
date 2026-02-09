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
    current_year = 2026
    age = max(1, current_year - year)
    km_per_year = kms / age
    if year >= 2022 and price < 10000:
        warnings.append(f"⚠️ **Price Alert:** AU ${price:,.0f} for a {year} model is suspiciously low.")
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
# 3. CRITIC AGENT
# =====================================================
def run_critic_audit(ml_data, ai_draft_text):
    critic_system_prompt = f"""
    You are a strict Data Auditor. Compare the FACTS against the AI DRAFT.
    FACTS:
    - Listed Price: ${ml_data['price']:,}
    - Predicted: ${ml_data['pred']:,}
    - Gap: {ml_data['gap']:.1f}% ({ml_data['verdict']})
    
    AUDIT RULES:
    1. If numbers in the DRAFT differ from FACTS, you MUST fail it (is_factual=false).
    2. Provide the corrected narrative or specific numeric fix in 'suggested_fix'.
    3. Every dollar sign ($) in the narrative MUST be escaped like \\$.
    """
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": critic_system_prompt},
                  {"role": "user", "content": ai_draft_text}],
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

st.title("🚗 AI Car Deal Advisor (Australia)")

with st.sidebar:
    st.write("### 📋 Extracted Details")
    vd = st.session_state.vehicle_data
    st.info(f"**Brand:** {vd['Brand'] or '-'}\n\n**Model:** {vd['Model'] or '-'}\n\n**Year:** {vd['Year'] or '-'}")
    if st.button("Clear & Reset"):
        st.session_state.chat_history = []
        st.session_state.vehicle_data = {k: None for k in st.session_state.vehicle_data}
        st.session_state.confirmed_plausibility = False
        st.rerun()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =====================================================
# 5. EXECUTION LOGIC
# =====================================================
user_input = st.chat_input("Enter car details...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Extract car data to JSON. Return ONLY JSON."},
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
    st.info("👋 Please enter car details to start.")
elif missing:
    st.warning(f"Missing: {', '.join(missing)}")
else:
    brand, model = str(v_curr["Brand"]), str(v_curr["Model"])
    year, kms, price = parse_numeric(v_curr["Year"]), parse_numeric(v_curr["Kilometres"]), parse_numeric(v_curr["Listed Price"])
    
    warnings = validate_data_plausibility(brand, model, year, kms, price)
    if warnings and not st.session_state.confirmed_plausibility:
        with st.chat_message("assistant"):
            st.error("### 🛑 Verify Data")
            for w in warnings: st.write(w)
            if st.checkbox("Details are correct"):
                st.session_state.confirmed_plausibility = True
                st.rerun()
        st.stop()

    # --- THE PRICE ESTIMATOR LOGIC (Restored) ---
    new_p = lookup_new_price(brand, model)
    if np.isnan(new_p):
        st.error("Baseline price not found for this model.")
    else:
        age = 2026 - year
        # Basic interactive features for the ML model
        X = pd.DataFrame([{
            "Age": age, "log_km": np.log1p(kms), "Brand": brand, "Model": model,
            "FuelConsumption": 8.0, "CylindersinEngine": 4, "Seats": 5,
            "age_kilometer_interaction": (age * kms) / 10000, "UsedOrNew": "USED",
            "DriveType": "AWD" if "kluger" in model.lower() else "FWD", 
            "BodyType": "SUV" if "kluger" in model.lower() else "Sedan",
            "Transmission": "Automatic", "FuelType": "Gasoline"
        }])
        
        pred = np.exp(pipe.predict(X)[0]) * new_p
        gap = ((price - pred) / pred) * 100
        verdict = "OVERPRICED" if gap > 7 else "BARGAIN" if gap < -7 else "FAIR PRICE"

        # --- REPORT GENERATION & CRITIC ---
        with st.spinner("🔍 Analysing Market..."):
            m_ctx = get_market_sources_for_brand(brand)
            report_prompt = f"Analyze {year} {brand} {model}, {kms}km, ${price}. Predicted: ${pred:.2f} (Gap: {gap:.1f}%). Market Context: {m_ctx}. Use 3 sections separated by '---'."
            
            raw_report = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are an Australian car expert."},
                          {"role": "user", "content": report_prompt}]
            ).choices[0].message.content

            ml_data = {"price": price, "pred": pred, "gap": gap, "verdict": verdict}
            audit = run_critic_audit(ml_data, raw_report)

            # If audit fails, force a rewrite with correct facts
            if not audit.is_factual:
                correction_prompt = f"Rewrite this report. FACTS: Predicted Price ${pred:,.2f}, Gap {gap:.1f}%. Report: {raw_report}"
                final_report = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": correction_prompt}]
                ).choices[0].message.content
            else:
                final_report = raw_report

            # Display
            sections = [s.strip() for s in final_report.split("---") if s.strip()]
            labels = ["📢 Verdict Summary", "💡 Brand Insights", "📉 Market Outlook"]
            for i, section in enumerate(sections):
                if i < len(labels):
                    st.markdown(f"### {labels[i]}")
                    st.write(section.replace(r"\$", "$"))