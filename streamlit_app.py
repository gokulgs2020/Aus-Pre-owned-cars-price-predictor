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

# Use st.secrets for Streamlit Cloud or .env for local
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
    # Support decimals and remove common car-related characters
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
        warnings.append(f"🏎️ **High Usage:** Averaging {int(km_per_year):,} km/year. (AU average is ~13,000km).")
    if year <= 2024 and kms < 500:
        warnings.append(f"🔍 **Low Odometer:** Only {kms:,} km on a {year} model. Verify if this is a demo or error.")
    return warnings

def lookup_new_price(brand, model):
    if bm is None or b is None: return np.nan
    bn, mn = str(brand).lower().replace(" ", ""), str(model).lower().replace(" ", "")
    
    # 1. Try Brand + Model match
    bm_match = bm[
        (bm['Brand'].str.lower().str.replace(" ", "") == bn) & 
        (bm['Model'].str.lower().str.replace(" ", "").str.contains(mn))
    ]
    if not bm_match.empty: 
        return float(bm_match.iloc[0]["New_Price_bm"])
    
    # 2. Fallback to Brand Average
    b_match = b[b['Brand'].str.lower().str.replace(" ", "") == bn]
    if not b_match.empty:
        return float(b_match.iloc[0]["New_Price_b"])
    
    return np.nan

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
    1. If numbers in the DRAFT differ from FACTS, you MUST fail it.
    2. Every single dollar sign ($) MUST be escaped with a backslash (\$).
    3. Ensure exactly three sections separated by '---'.
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

st.title("🚗 AI Car Deal Advisor (Australia)")

# Sidebar for current status
with st.sidebar:
    st.write("### 📋 Extracted Details")
    v = st.session_state.vehicle_data
    st.info(f"**Brand:** {v['Brand'] or '-'}\n\n"
            f"**Model:** {v['Model'] or '-'}\n\n"
            f"**Year:** {v['Year'] or '-'}\n\n"
            f"**Km:** {f'{parse_numeric(v['Kilometres']):,.0f}' if v['Kilometres'] else '-'}\n\n"
            f"**Price:** {f'${parse_numeric(v['Listed Price']):,.0f}' if v['Listed Price'] else '-'}")
    
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
user_input = st.chat_input("Paste a CarSales link or type car details (e.g. 2022 Toyota Corolla 40k km $25000)")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Data Extraction
    with st.spinner("Extracting vehicle details..."):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Extract/Update car data. Keys: Brand, Model, Year, Kilometres, Listed Price. Return ONLY JSON."},
                      {"role": "user", "content": f"Current: {st.session_state.vehicle_data}\nInput: {user_input}"}],
            temperature=0
        )
        new_data = safe_json_parse(resp.choices[0].message.content)
        for key in st.session_state.vehicle_data:
            if new_data.get(key) is not None: 
                st.session_state.vehicle_data[key] = new_data[key]
        
    st.session_state.confirmed_plausibility = False # Reset on new info
    st.rerun()

# Evaluation Logic
v_curr = st.session_state.vehicle_data
required_fields = ["Brand", "Model", "Year", "Kilometres", "Listed Price"]
missing = [f for f in required_fields if v_curr[f] is None]

if not any(v_curr.values()):
    st.info("👋 Welcome! Please paste a car listing to begin analysis.")
elif missing:
    st.warning(f"I'm still missing: **{', '.join(missing)}**. Please provide these details.")
else:
    brand, model = str(v_curr["Brand"]), str(v_curr["Model"])
    year = parse_numeric(v_curr["Year"])
    kms = parse_numeric(v_curr["Kilometres"])
    price = parse_numeric(v_curr["Listed Price"])
    
    # Plausibility Guardrail
    warnings = validate_data_plausibility(brand, model, year, kms, price)
    if warnings and not st.session_state.confirmed_plausibility:
        with st.chat_message("assistant"):
            st.error("### 🛑 Please Verify Data")
            for w in warnings: st.write(w)
            if st.checkbox("These details are correct, proceed with analysis"):
                st.session_state.confirmed_plausibility = True
                st.rerun()
        st.stop()

    # Analysis Generation
    if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user":
        with st.chat_message("assistant"):
            new_p = lookup_new_price(brand, model)
            
            if np.isnan(new_p):
                st.error(f"Could not find baseline pricing for a {brand} {model} in our Australian database.")
            else:
                # ML Prediction
                age = 2026 - year
                X = pd.DataFrame([{
                    "Age": age, "log_km": np.log1p(kms), "Brand": brand, "Model": model,
                    "FuelConsumption": 7.5, "CylindersinEngine": 4, "Seats": 5,
                    "age_kilometer_interaction": (age * kms) / 10000, "UsedOrNew": "USED",
                    "DriveType": "FWD", "BodyType": "Sedan", "Transmission": "Automatic", "FuelType": "Gasoline"
                }])
                
                pred_raw = pipe.predict(X)[0]
                pred = np.exp(pred_raw) * new_p
                gap = ((price - pred) / pred) * 100
                
                # Verdict Styling
                if gap < -15: verdict, color = "VERY LOW! (Check for issues)", "orange"
                elif gap < -5: verdict, color = "BARGAIN", "green"
                elif gap <= 7: verdict, color = "FAIRLY PRICED", "blue"
                else: verdict, color = "OVERPRICED!", "red"

                st.subheader(f"Verdict: :{color}[{verdict}]")
                
                # Narrative & Audit
                m_ctx = get_market_sources_for_brand(brand)
                report_prompt = f"""
                Analyze: {year} {brand} {model}, {kms:,.0f}km, listed at \${price:,.0f}.
                Our Model Predicts: \${pred:,.0f} (Gap: {gap:.1f}%). 
                
                CONTEXT: {m_ctx}
                
                FORMAT: 3 sections separated by '---'. 
                Section 1: Verdict Summary. 
                Section 2: Brand/Model Insights. 
                Section 3: Resale Parameters.
                Use Australian English. Escape all $ with \$.
                """

                with st.spinner("🔍 Performing Critic Audit..."):
                    raw_report = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "system", "content": "You are a professional Australian automotive valuations expert."},
                                  {"role": "user", "content": report_prompt}],
                        temperature=0.7
                    ).choices[0].message.content
                    
                    # Final Audit
                    ml_data = {"price": price, "pred": pred, "gap": round(gap, 1), "verdict": verdict}
                    audit = run_critic_audit(ml_data, raw_report)
                    final_report = audit.suggested_fix if not audit.is_factual else raw_report
                    
                    # UI Rendering
                    sections = final_report.split("---")
                    labels = ["📢 Verdict Summary", "💡 Brand & Model Insights", "📉 Resale & Market Outlook"]
                    
                    for i, section in enumerate(sections):
                        if i < len(labels):
                            st.markdown(f"**{labels[i]}**")
                            st.write(section.strip())
                    
                    st.session_state.chat_history.append({"role": "assistant", "content": final_report})