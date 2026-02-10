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
# This flag ensures the report generates immediately after valid extraction
if "trigger_analysis" not in st.session_state:
    st.session_state.trigger_analysis = False

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
    
    # 1. UI METRICS (Always visible)
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Brand", v["Brand"] or "-")
    d2.metric("Model", v["Model"] or "-")
    d3.metric("Year", v["Year"] or "-")
    d4.metric("Km", f"{v['Kilometres']:,}" if v["Kilometres"] else "-")
    d5.metric("Price", f"${v['Listed Price']:,}" if v["Listed Price"] else "-")

    if st.button("Reset Advisor"):
        for key in ["chat_history", "vehicle_data", "confirmed_plausibility", "trigger_analysis"]:
            if key == "vehicle_data": st.session_state[key] = {k: None for k in v}
            elif key == "chat_history": st.session_state[key] = []
            else: st.session_state[key] = False
        st.rerun()

    # 2. CHAT DISPLAY
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    # 3. INPUT PROCESSING
    # 3. CHAT INPUT & SMART GUARDRAILS
    user_input = st.chat_input("Enter details (e.g., '2022 Toyota Corolla 30k km $25000')")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # --- SMART GUARDRAIL ---
        v_curr = st.session_state.vehicle_data
        missing = [k for k, val in v_curr.items() if val is None]
        
        # Allow input if it contains car keywords OR if it's a number and we are missing numeric fields
        car_keywords = ['toyota', 'honda', 'mazda', 'hyundai', 'kia', 'km', 'price', '$', '20', 'model', 'car', 'kluger', 'crv', 'cr-v']
        is_relevant = any(word in user_input.lower() for word in car_keywords) or (len(user_input.split()) > 2)
        
        # New: If the user provides just a number (like '40000') and we are missing KMs or Price, let it pass
        is_numeric_followup = user_input.strip().isdigit() and ('Kilometres' in missing or 'Listed Price' in missing)

        if not (is_relevant or is_numeric_followup):
            msg = "I can assist with used car price evaluation only! Please enter a valid Brand, Model, Year and KMs."
            st.session_state.chat_history.append({"role": "assistant", "content": msg})
            st.rerun()
        
        # --- EXTRACTION ---
        with st.spinner("Updating details..."):
            # The LLM Extractor is great at taking "40000" and knowing it belongs in KMs 
            # because we pass the current vehicle_data context to it.
            ext_p = get_extraction_prompt(st.session_state.vehicle_data, user_input)
            raw_json = call_llm_extractor(client, SYSTEM_EXTRACTOR, ext_p, expect_json=True)
            if raw_json:
                for key in st.session_state.vehicle_data:
                    if raw_json.get(key) is not None: 
                        st.session_state.vehicle_data[key] = raw_json[key]
        
        st.session_state.trigger_analysis = True 
        st.rerun()

    # 4. DATA VALIDATION & PROMPT FOR MISSING INFO
    v_curr = st.session_state.vehicle_data
    
    # Identify what is missing
    missing = [k for k, val in v_curr.items() if val is None]
    
    # If something is missing and the user just typed something
    if missing and st.session_state.trigger_analysis:
        st.session_state.trigger_analysis = False
        with st.chat_message("assistant"):
            missing_str = ", ".join(missing)
            prompt_msg = f"I've got some details, but I'm still missing the **{missing_str}**. Could you please provide those?"
            st.info(prompt_msg)
            st.session_state.chat_history.append({"role": "assistant", "content": prompt_msg})
        st.stop()

    # 5. THE ANALYST ENGINE (Runs only when data is complete)
    if all(v_curr.values()) and st.session_state.trigger_analysis:
        # 1. Capture the boolean AND the resolved name (canonical_name)
        is_valid, result = validate_model_existence(v_curr["Brand"], v_curr["Model"], brand_model_lookup)
        
        if not is_valid:
            with st.chat_message("assistant"):
                # Use the reason (result) to give better feedback
                if result == "rubbish":
                    st.error(f"⚠️ I couldn't understand the model: '{v_curr['Model']}'")
                else:
                    st.warning(f"❌ Unsupported Model: '{v_curr['Model']}'")
                
                if st.button("Clear Invalid Model"):
                    st.session_state.vehicle_data["Model"] = None
                    st.session_state.trigger_analysis = False
                    st.rerun()
            st.stop()
        else:
            # 2. SUCCESS: Update state with the official name from the database
            # This ensures 'crv' becomes 'CR-V' before hitting the ML pipe
            st.session_state.vehicle_data["Model"] = result
            # We refresh the local variable to use the clean name immediately
            model = result

        brand, model = str(v_curr["Brand"]), str(v_curr["Model"])
        year, kms, price = parse_numeric(v_curr["Year"]), parse_numeric(v_curr["Kilometres"]), parse_numeric(v_curr["Listed Price"])
        
        # Plausibility check
        warnings = validate_data_plausibility(brand, model, year, kms, price)
        if warnings and not st.session_state.confirmed_plausibility:
            with st.chat_message("assistant"):
                st.error("### 🛑 Check Details")
                for w in warnings: st.write(w)
                if st.button("Yes, these details are correct"):
                    st.session_state.confirmed_plausibility = True
                    st.rerun()
            st.stop()

        # FINAL ANALYSIS REPORT
        with st.chat_message("assistant"):
            st.session_state.trigger_analysis = False 
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
                
                with st.spinner("Generating Market Report..."):
                    report = call_llm_extractor(client, SYSTEM_ANALYST, rep_p, temperature=0.7)
                    st.markdown(report)
                    st.session_state.chat_history.append({"role": "assistant", "content": f"**Verdict: {verdict}**\n\n{report}"})