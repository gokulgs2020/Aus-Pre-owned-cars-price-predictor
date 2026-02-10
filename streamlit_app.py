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
        st.session_state.chat_history = []
        st.session_state.vehicle_data = {k: None for k in v}
        st.session_state.confirmed_plausibility = False
        st.session_state.trigger_analysis = False
        st.rerun()

    # 2. CHAT DISPLAY
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    # 3. CHAT INPUT & SMART GUARDRAILS
    user_input = st.chat_input("Enter details (e.g., '2022 Toyota Corolla 30k km $25000')")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # --- SMART GUARDRAIL ---
        v_curr = st.session_state.vehicle_data
        missing = [k for k, val in v_curr.items() if val is None]
        
        clean_input = user_input.strip().replace(",", "").replace("$", "")
        car_keywords = ['toyota', 'honda', 'mazda', 'hyundai', 'kia', 'km', 'price', '$', '20', 'model', 'car', 'kluger', 'crv', 'cr-v', 'rav4']
        is_relevant = any(word in user_input.lower() for word in car_keywords) or (len(user_input.split()) > 2)
        
        # Allow numeric follow-ups if price, kms, or year are missing
        is_numeric_followup = clean_input.isdigit() and any(k in missing for k in ['Kilometres', 'Listed Price', 'Year'])

        if not (is_relevant or is_numeric_followup):
            msg = "I can assist with used car price evaluation only! Please enter a valid Brand, Model, Year and KMs."
            st.session_state.chat_history.append({"role": "assistant", "content": msg})
            st.rerun()
        
        # --- EXTRACTION WITH HEURISTIC FALLBACK ---
        with st.spinner("Updating details..."):
            ext_p = get_extraction_prompt(st.session_state.vehicle_data, user_input)
            raw_json = call_llm_extractor(client, SYSTEM_EXTRACTOR, ext_p, expect_json=True)
            
            # Catch Numeric Ambiguity (Disambiguation Logic)
            if raw_json and raw_json.get("ambiguity") == "numeric_collision":
                msg = "I see two numbers here. Could you clarify which one is the **Price** and which is the **Kilometres**?"
                st.session_state.chat_history.append({"role": "assistant", "content": msg})
                st.rerun()

            # Senior Heuristic: If LLM failed to extract a lone numeric follow-up, map it manually
            if is_numeric_followup and raw_json:
                val = int(clean_input)
                if v_curr["Kilometres"] is None and val > 1000: raw_json["Kilometres"] = val
                elif v_curr["Year"] is None and 1990 <= val <= 2026: raw_json["Year"] = val
                elif v_curr["Listed Price"] is None: raw_json["Listed Price"] = val

            if raw_json:
                for key in st.session_state.vehicle_data:
                    if raw_json.get(key) is not None: 
                        val = raw_json[key]
                        if key in ["Year", "Kilometres", "Listed Price"] and val is not None:
                            try: val = int(str(val).replace(",", "").replace("$", ""))
                            except: pass
                        st.session_state.vehicle_data[key] = val
        
        st.session_state.trigger_analysis = True 
        st.rerun()

    # 4. DATA VALIDATION & MISSING INFO PROMPT
    v_curr = st.session_state.vehicle_data
    missing = [k for k, val in v_curr.items() if val is None]

    if missing and st.session_state.trigger_analysis:
        st.session_state.trigger_analysis = False
        with st.chat_message("assistant"):
            msg = f"I've got some details, but I'm still missing the **{', '.join(missing)}**. Could you provide those?"
            st.info(msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg})
        st.stop()

    # 5. THE ANALYST ENGINE
    if all(v_curr.values()) and st.session_state.trigger_analysis:
        # Cross-validate Brand/Model (Handles noise like @##@RDSF)
        is_valid, result = validate_model_existence(v_curr["Brand"], v_curr["Model"], brand_model_lookup)
        
        if not is_valid:
            with st.chat_message("assistant"):
                st.warning(f"❌ {result.get('message', 'Unsupported Model')}")
                if st.button("Clear Invalid Model"):
                    st.session_state.vehicle_data["Model"] = None
                    st.rerun()
            st.stop()
        else:
            # result is a dict: {"brand": "Toyota", "model": "Kluger", "status": "corrected"}
            st.session_state.vehicle_data["Brand"] = result["brand"]
            st.session_state.vehicle_data["Model"] = result["model"]
            if result["status"] == "corrected":
                st.toast(f"🤖 Resolved to {result['brand']} {result['model']}", icon="✅")

        # Prepare for ML Prediction
        brand, model = str(st.session_state.vehicle_data["Brand"]), str(st.session_state.vehicle_data["Model"])
        year, kms, price = parse_numeric(v_curr["Year"]), parse_numeric(v_curr["Kilometres"]), parse_numeric(v_curr["Listed Price"])
        
        # Plausibility Check (Safety logic)
        warnings = validate_data_plausibility(brand, model, year, kms, price)
        if warnings and not st.session_state.confirmed_plausibility:
            with st.chat_message("assistant"):
                st.error("### 🛑 Check Details")
                for w in warnings: st.write(w)
                if st.button("Yes, these are correct"):
                    st.session_state.confirmed_plausibility = True
                    st.rerun()
            st.stop()

        # FINAL REPORT GENERATION
        with st.chat_message("assistant"):
            st.session_state.trigger_analysis = False 
            new_p = lookup_new_price(brand, model, bm, b)
            
            if np.isnan(new_p):
                st.write(f"⚠️ Baseline market baseline not found for {brand} {model}.")
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