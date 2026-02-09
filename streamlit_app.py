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
    # Placeholder for your specific paths
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
    brand_l = brand.lower()
    result = {"resale": [], "maintenance": [], "reliability": [], "depreciation": []}
    for entry in MARKET_SOURCES:
        brands = [b.lower() for b in entry.get("brands", [])]
        if brand_l in brands or "all" in brands:
            topic = entry["topic"].lower()
            if topic in result:
                result[topic].append(f"{entry['text']} (Source: {entry['source']})")
    return result

def lookup_new_price(brand, model):
    bn, mn = brand.lower().replace(" ", ""), model.lower().replace(" ", "")
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

# =====================================================
# SESSION STATE
# =====================================================
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "vehicle_data" not in st.session_state: st.session_state.vehicle_data = {}

# =====================================================
# UI LAYOUT
# =====================================================
st.title("🚗 Preowned Car Intelligence")
tab1, tab2 = st.tabs(["Price Estimator", "AI Deal Advisor"])

with tab1:
    st.header("Quick Market Estimate")
    col1, col2 = st.columns(2)
    with col1:
        sel_brand = st.selectbox("Brand", sorted(brand_model_lookup["Brand"].unique()))
        sel_year = st.number_input("Year", 2000, 2025, 2020)
    with col2:
        models = brand_model_lookup[brand_model_lookup["Brand"] == sel_brand]["Model"].unique()
        sel_model = st.selectbox("Model", sorted(models))
        sel_km = st.number_input("Kilometres", 0, 300000, 50000)

    if st.button("Calculate Estimate"):
        age = 2025 - sel_year
        new_p = lookup_new_price(sel_brand, sel_model)
        if np.isnan(new_p):
            st.warning("New price data unavailable for this selection.")
        else:
            # Feature building logic (simplified for snippet)
            X = pd.DataFrame([{"Age": age, "log_km": np.log1p(sel_km), "Brand": sel_brand, "Model": sel_model,
                               "FuelConsumption": 7.5, "CylindersinEngine": 4, "Seats": 5,
                               "age_kilometer_interaction": (age * sel_km) / 10000, "UsedOrNew": "USED",
                               "DriveType": "FWD", "BodyType": "Sedan", "Transmission": "Automatic", "FuelType": "Gasoline"}])
            retention = np.exp(pipe.predict(X)[0])
            est_p = retention * new_p
            st.metric("Estimated Market Value", f"AU ${est_p:,.0f}", f"{retention*100:.1f}% Retention")

with tab2:
    st.header("AI Deal Advisor")
    st.info("Paste a car listing description below. I'll analyze the price, brand reputation, and market trends.")
    
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.vehicle_data = {}
        st.rerun()

    # Display Chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ex: 2019 Toyota Corolla, 45000km, $22000")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 1. Extraction Step
        extract_prompt = f"Extract Brand, Model, Year, Kilometres, and Listed Price from: '{user_input}'. Return JSON only."
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a data extractor. Return JSON: {Brand, Model, Year, Kilometres, Listed Price}"},
                      {"role": "user", "content": extract_prompt}],
            temperature=0
        )
        
        extracted = safe_json_parse(resp.choices[0].message.content)
        for k, v in extracted.items():
            if v: st.session_state.vehicle_data[k] = v
        
        v_data = st.session_state.vehicle_data
        
        # 2. Validation
        required = ["Brand", "Model", "Year", "Kilometres", "Listed Price"]
        missing = [r for r in required if r not in v_data or v_data[r] is None]
        
        if missing:
            msg = f"I've noted the details, but I still need: **{', '.join(missing)}** to give you a full analysis."
            st.session_state.chat_history.append({"role": "assistant", "content": msg})
            with st.chat_message("assistant"): st.write(msg)
        else:
            # 3. Calculation & Strategy
            brand, model = v_data["Brand"], v_data["Model"]
            year, kms = parse_numeric(v_data["Year"]), parse_numeric(v_data["Kilometres"])
            price = parse_numeric(v_data["Listed Price"])
            
            age = 2025 - year
            new_p = lookup_new_price(brand, model)
            
            if np.isnan(new_p):
                ans = "I couldn't find historical pricing for this specific model to run a deep analysis."
            else:
                X_adv = pd.DataFrame([{"Age": age, "log_km": np.log1p(kms), "Brand": brand, "Model": model,
                                       "FuelConsumption": 7.5, "CylindersinEngine": 4, "Seats": 5,
                                       "age_kilometer_interaction": (age * kms) / 10000, "UsedOrNew": "USED",
                                       "DriveType": "FWD", "BodyType": "Sedan", "Transmission": "Automatic", "FuelType": "Gasoline"}])
                pred_p = np.exp(pipe.predict(X_adv)[0]) * new_p
                gap = ((price - pred_p) / pred_p) * 100
                m_ctx = get_market_sources_for_brand(brand)

                # DYNAMIC PROMPT LOGIC
                luxury_brands = ['bmw', 'mercedes', 'audi', 'lexus', 'porsche', 'land rover']
                persona = "Luxury Specialist" if brand.lower() in luxury_brands else "Value Analyst"

                final_prompt = f"""
                You are a {persona} for the Australian car market. 
                Analyze this listing: {year} {brand} {model}, {kms}km, Listed: AU ${price:,.0f}.
                Our Model Predicts: AU ${pred_p:,.0f} (Gap: {gap:+.1f}%).

                MARKET DATA:
                - Resale: {m_ctx['resale']}
                - Reliability: {m_ctx['reliability']}
                - Maintenance: {m_ctx['maintenance']}

                TASK:
                Write a 3-part summary. 
                1. 'Value Assessment': Discuss the {gap:+.1f}% price gap. Is it a deal or a rip-off?
                2. 'Brand Intelligence': Use the MARKET DATA to explain how {brand}'s reputation affects this car's future. 
                   YOU MUST CITE SOURCES (e.g., Source: RedBook).
                3. 'Buyer's Playbook': Give 2 specific pieces of advice for inspecting/negotiating this specific car.

                STRICT RULES:
                - Do NOT use a generic template. 
                - Use {brand}-specific terminology (e.g., "build quality" for luxury, "running costs" for budget).
                - Keep it punchy and avoid repeating the same phrases.
                """

                with st.spinner("Analyzing market data..."):
                    ai_resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": final_prompt}],
                        temperature=0.7 # Higher temperature for variety
                    )
                    ans = ai_resp.choices[0].message.content

            st.session_state.chat_history.append({"role": "assistant", "content": ans})
            with st.chat_message("assistant"):
                st.markdown(ans)