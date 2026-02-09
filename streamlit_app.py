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

def validate_data_plausibility(brand, model, year, kms, price):
    warnings = []
    age = max(1, 2026 - year)
    km_per_year = kms / age

    # Scam Detection: Price vs. Age
    if year >= 2022 and price < 8000:
        warnings.append(f"⚠️ **Price Alert:** AU ${price:,} for a {year} vehicle is significantly below market value. Be extremely cautious of potential scams.")

    # Extreme Mileage: High
    if km_per_year > 50000:
        warnings.append(f"🏎️ **Extreme Usage:** This car has averaged over {int(km_per_year):,} km/year. This is double the Australian average.")

    # Extreme Mileage: Low
    if year <= 2023 and kms < 1000:
        warnings.append(f"🔍 **Suspiciously Low Kms:** Only {kms:,} km on a {year} model. Please verify if the odometer is accurate.")

    return warnings
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
        sel_year = st.number_input("Year", 2000, 2026, 2022)
    with col2:
        models = brand_model_lookup[brand_model_lookup["Brand"] == sel_brand]["Model"].unique()
        sel_model = st.selectbox("Model", sorted(models))
        sel_km = st.number_input("Kilometres", 0, 400000, 30000)

    if st.button("Calculate Estimate"):
        age = 2026 - sel_year
        new_p = lookup_new_price(sel_brand, sel_model)
        if np.isnan(new_p):
            st.warning("New price data unavailable.")
        else:
            X = pd.DataFrame([{"Age": age, "log_km": np.log1p(sel_km), "Brand": sel_brand, "Model": sel_model,
                               "FuelConsumption": 7.5, "CylindersinEngine": 4, "Seats": 5,
                               "age_kilometer_interaction": (age * sel_km) / 10000, "UsedOrNew": "USED",
                               "DriveType": "FWD", "BodyType": "Sedan", "Transmission": "Automatic", "FuelType": "Gasoline"}])
            retention = np.exp(pipe.predict(X)[0])
            est_p = retention * new_p
            st.metric("Market Valuation", f"AU ${est_p:,.0f}", f"{retention*100:.1f}% Retention")

with tab2:
    st.header("AI Deal Advisor")
    
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.vehicle_data = {}
        st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Paste listing details here...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        extract_prompt = f"Extract car details from: '{user_input}'. Return JSON: {{'Brand': str, 'Model': str, 'Year': int, 'Kilometres': int, 'Listed Price': int}}"
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Extract only facts. Return JSON."}, {"role": "user", "content": extract_prompt}],
            temperature=0
        )
        
        extracted = safe_json_parse(resp.choices[0].message.content)
        for k, v in extracted.items():
            if v: st.session_state.vehicle_data[k] = v
        
        v_data = st.session_state.vehicle_data
        required = ["Brand", "Model", "Year", "Kilometres", "Listed Price"]
        missing = [r for r in required if r not in v_data or v_data[r] is None]
        
        if missing:
            msg = f"Almost there! I still need: **{', '.join(missing)}**."
            st.session_state.chat_history.append({"role": "assistant", "content": msg})
            with st.chat_message("assistant"): st.write(msg)
        else:
            # 2. PLAUSIBILITY CHECK (The New Guardrail)
            brand, model = v_data["Brand"], v_data["Model"]
            year, kms = parse_numeric(v_data["Year"]), parse_numeric(v_data["Kilometres"])
            price = parse_numeric(v_data["Listed Price"])
            
            # Generate warnings based on Australian market norms
            sanity_warnings = validate_data_plausibility(brand, model, year, kms, price)
            
            # If warnings exist and the user hasn't clicked "Yes, they are correct" yet
            if sanity_warnings and not st.session_state.get("confirmed_plausibility"):
                with st.chat_message("assistant"):
                    st.error("### 🛑 Wait, let's double-check these details!")
                    for w in sanity_warnings:
                        st.write(w)
                    
                    st.write("If these details are definitely correct (e.g., a salvage car or a rare find), click below to proceed.")
                    if st.button("Yes, these details are correct"):
                        st.session_state.confirmed_plausibility = True
                        st.rerun()
                st.stop() # This halts the script so the AI doesn't generate a report yet

            # --- DATA IS VALIDATED ---
            # Reset the confirmation for the NEXT search so the check runs again for new cars
            if st.session_state.get("confirmed_plausibility"):
                st.session_state.confirmed_plausibility = True 

            age = 2026 - year
            new_p = lookup_new_price(brand, model)
            
            if np.isnan(new_p):
                ans = "I lack enough data to run a full analysis for this specific model."
            else:
                X_adv = pd.DataFrame([{"Age": age, "log_km": np.log1p(kms), "Brand": brand, "Model": model,
                                       "FuelConsumption": 7.5, "CylindersinEngine": 4, "Seats": 5,
                                       "age_kilometer_interaction": (age * kms) / 10000, "UsedOrNew": "USED",
                                       "DriveType": "FWD", "BodyType": "Sedan", "Transmission": "Automatic", "FuelType": "Gasoline"}])
                pred_p = np.exp(pipe.predict(X_adv)[0]) * new_p
                gap = ((price - pred_p) / pred_p) * 100
                m_ctx = get_market_sources_for_brand(brand)

                # DYNAMIC ANALYTIC LOGIC
                deal_type = "suspiciously low" if gap < -15 else "strong bargain" if gap < -5 else "market fair" if gap < 5 else "premium listing"
                luxury = ['bmw', 'mercedes', 'audi', 'lexus', 'porsche', 'land rover']
                persona = "Luxury Portfolio Advisor" if brand.lower() in luxury else "Consumer Value Specialist"

                final_prompt = f"""
                Persona: {persona} (Australia)
                Vehicle: {year} {brand} {model}, {kms:,}km.
                Pricing: Listed AU ${price:,.0f} vs Predicted AU ${pred_p:,.0f} ({gap:+.1f}% variance).
                Deal Context: This is a {deal_type} listing.

                MARKET DATA SOURCES:
                {m_ctx}

                TASK:
                Write a 3-section evaluation.
                - DO NOT use a standard intro like 'For this car...'. 
                - Adapt your vocabulary: for a {brand}, focus on its specific segment (e.g., 'reliability' for Toyota, 'prestige' for BMW).
                - Section 1: 'The Price Narrative'. Interpret the {gap:+.1f}% gap. If it's a {deal_type} deal, tell the buyer what to be wary of or why it's a win.
                - Section 2: 'Segment Insights'. Integrate the MARKET DATA naturally. Cite sources (e.g. Source: RedBook) inline.
                - Section 3: 'Your Move'. Provide 2 non-generic negotiation or inspection tips specific to this {brand} {model}.
                """

                with st.spinner("Analyzing Listing..."):
                    ai_resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "system", "content": "You are a witty, professional auto-analyst. Avoid templates. Be unique every time."},
                                  {"role": "user", "content": final_prompt}],
                        temperature=0.6 #  Creativity for variety
                    )
                    ans = ai_resp.choices[0].message.content

            st.session_state.chat_history.append({"role": "assistant", "content": ans})
            with st.chat_message("assistant"):
                st.markdown(ans)