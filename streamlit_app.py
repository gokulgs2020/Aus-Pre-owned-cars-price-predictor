import os
import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI

# Custom Module Imports
from src.gen_ai.model_utils import load_artifacts, lookup_new_price, calculate_market_prediction, get_market_sources_for_brand
from src.gen_ai.validator import parse_numeric, validate_data_plausibility, validate_model_existence
from src.gen_ai.prompts import SYSTEM_EXTRACTOR, SYSTEM_ANALYST, get_extraction_prompt, get_report_prompt
from src.gen_ai.extractor import call_llm_extractor, safe_json_parse

# =====================================================
# CONFIG & CLIENT
# =====================================================
st.set_page_config(page_title="Car Price Deal Advisor", layout="wide")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OPENAI_API_KEY")
    st.stop()

client = OpenAI(api_key=api_key)
pipe, bm, b, brand_model_lookup = load_artifacts()

# =====================================================
# SESSION STATE
# =====================================================
if "chat_history" not in st.session_state: 
    st.session_state.chat_history = []
if "vehicle_data" not in st.session_state: 
    st.session_state.vehicle_data = {"Brand": None, "Model": None, "Year": None, "Kilometres": None, "Listed Price": None}
if "confirmed_plausibility" not in st.session_state: 
    st.session_state.confirmed_plausibility = False
# Tracks if we just updated the data so we can trigger the analyst
if "just_updated" not in st.session_state:
    st.session_state.just_updated = False

# =====================================================
# UI LAYOUT
# =====================================================
st.title("🤖 Car Price Advisor")
tab1, tab2 = st.tabs(["Price Estimator", "AI Deal Advisor"])

with tab1:
    st.header("Quick Market Estimate")
    col1, col2 = st.columns(2)
    with col1:
        sel_brand = st.selectbox("Brand", sorted(brand_model_lookup["Brand"].unique()), key="tab1_brand")
        sel_year = st.number_input("Year", 2000, 2026, 2022, key="tab1_year")
    with col2:
        models = brand_model_lookup[brand_model_lookup["Brand"] == sel_brand]["Model"].unique()
        sel_model = st.selectbox("Model", sorted(models), key="tab1_model")
        sel_km = st.number_input("Kilometres", 0, 400000, 30000, 5000, key="tab1_km")
    
    if st.button("Calculate Estimate"):
        new_p = lookup_new_price(sel_brand, sel_model, bm, b)
        if not np.isnan(new_p):
            retention = calculate_market_prediction(pipe, sel_brand, sel_model, sel_year, sel_km)
            st.metric("Market Valuation", f"AU ${(retention * new_p):,.0f}")
        else:
            st.warning("New price baseline not found.")

with tab2:
    st.header("AI Deal Advisor")
    v = st.session_state.vehicle_data
    
    # 1. UI METRICS BAR
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
        st.session_state.just_updated = False
        st.rerun()

    # 2. CHAT DISPLAY
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): 
            st.markdown(msg["content"])

    # 3. CHAT INPUT & GUARDRAILS
    user_input = st.chat_input("Enter details (e.g., '2022 Toyota Corolla 30k km $25000')")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # --- GUARDRAIL: IRRELEVANT INPUT CHECK ---
        car_keywords = ['toyota', 'mazda', 'honda', 'hyundai', 'kia', 'km', 'price', '$', '20', 'model', 'car', 'valuation', 'sell', 'buy', 'kluger']
        is_relevant = any(word in user_input.lower() for word in car_keywords) or (len(user_input.split()) > 3)

        if not is_relevant:
            msg = "I can assist with used car price evaluation only! Please enter a valid Brand, Model, Year and KMs."
            st.session_state.chat_history.append({"role": "assistant", "content": msg})
            st.rerun()
        
        # --- DATA EXTRACTION ---
        with st.spinner("Extracting details..."):
            ext_p = get_extraction_prompt(st.session_state.vehicle_data, user_input)
            raw_json = call_llm_extractor(client, SYSTEM_EXTRACTOR, ext_p, expect_json=True)
            
            if raw_json:
                for key in st.session_state.vehicle_data:
                    if raw_json.get(key) is not None: 
                        st.session_state.vehicle_data[key] = raw_json[key]
        
        st.session_state.confirmed_plausibility = False
        st.session_state.just_updated = True # Flag to trigger the analyst on rerun
        st.rerun()

    # 4. VALIDATION
    v_curr = st.session_state.vehicle_data
    if v_curr["Model"]:
        is_valid, reason = validate_model_existence(v_curr["Brand"], v_curr["Model"], brand_model_lookup)
        if not is_valid:
            with st.chat_message("assistant"):
                st.warning(f"❌ Unsupported Model: '{v_curr['Model']}'")
                if st.button("Clear Invalid Model"):
                    st.session_state.vehicle_data["Model"] = None
                    if st.session_state.chat_history: st.session_state.chat_history.pop()
                    st.rerun()
            st.stop()

    # 5. ANALYSIS TRIGGER
    if v_curr["Brand"] and v_curr["Model"] and v_curr["Year"]:
        brand, model = str(v_curr["Brand"]), str(v_curr["Model"])
        year, kms, price = parse_numeric(v_curr["Year"]), parse_numeric(v_curr["Kilometres"]), parse_numeric(v_curr["Listed Price"])
        
        # Plausibility check
        warnings = validate_data_plausibility(brand, model, year, kms, price)
        if warnings and not st.session_state.confirmed_plausibility:
            with st.chat_message("assistant"):
                st.error("### 🛑 Plausibility Warning")
                for w in warnings: st.write(w)
                if st.button("Yes, these details are correct"):
                    st.session_state.confirmed_plausibility = True
                    st.rerun()
            st.stop()

        # RUN THE ANALYST
        # We trigger this if there was just an update OR the last message was from the user
        if st.session_state.just_updated or (st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "user"):
            st.session_state.just_updated = False # Reset the flag
            
            with st.chat_message("assistant"):
                new_p = lookup_new_price(brand, model, bm, b)
                if np.isnan(new_p):
                    st.write(f"⚠️ Baseline market price not found for {brand} {model}.")
                else:
                    retention = calculate_market_prediction(pipe, brand, model, year, kms)
                    pred = retention * new_p
                    gap = ((price - pred) / pred) * 100 if price else 0
                    
                    verdict, color = ("FAIR PRICED", "blue")
                    if gap < -15: verdict, color = "VERY LOW!", "orange"
                    elif gap < -5: verdict, color = "BARGAIN", "green"
                    elif gap > 5: verdict, color = "OVER PRICED!", "orange"

                    st.markdown(f"### Verdict: :{color}[{verdict}]")
                    m_ctx = get_market_sources_for_brand(brand)
                    rep_p = get_report_prompt(year, brand, model, kms, price, pred, gap, verdict, m_ctx)
                    
                    with st.spinner("Analyzing Market Data..."):
                        report = call_llm_extractor(client, SYSTEM_ANALYST, rep_p, temperature=0.7)
                        st.markdown(report)
                        st.session_state.chat_history.append({"role": "assistant", "content": f"**Verdict: {verdict}**\n\n{report}"})