import joblib
import pandas as pd
import numpy as np
import streamlit as st
import json
import os

@st.cache_resource
def load_artifacts():
    # 1. Find the project root directory
    # This file: x/src/gen_ai/model_utils.py
    current_file_path = os.path.abspath(__file__)
    
    # Go up 1 level: x/src/gen_ai
    gen_ai_dir = os.path.dirname(current_file_path)
    
    # Go up 2 levels: x/src
    src_dir = os.path.dirname(gen_ai_dir)
    
    # Go up 3 levels to reach the root: x/
    project_root = os.path.dirname(src_dir)
    
    # 2. Define the path to the models folder
    models_dir = os.path.join(project_root, "models")
    
    # 3. Load artifacts
    pipe = joblib.load(os.path.join(models_dir, "final_price_pipe.joblib"))
    bm = pd.read_csv(os.path.join(models_dir, "new_price_lookup_bm.csv"))
    b = pd.read_csv(os.path.join(models_dir, "new_price_lookup_b.csv"))
    lookup = pd.read_csv(os.path.join(models_dir, "brand_model_lookup_50.csv"))
    
    # Clean whitespace
    for df_ in (bm, b, lookup):
        for c in ("Brand", "Model"):
            if c in df_.columns:
                df_[c] = df_[c].astype(str).str.strip()
                
    return pipe, bm, b, lookup

@st.cache_resource
def load_market_sources():
    with open("data/curated_market_sources.json", "r", encoding="utf-8") as f:
        return json.load(f)

def get_market_sources_for_brand(brand: str):
    sources = load_market_sources()
    brand_l = str(brand).lower()
    result = {"resale": "", "maintenance": "", "reliability": "", "depreciation": ""}
    for entry in sources:
        brands = [b.lower() for b in entry.get("brands", [])]
        if brand_l in brands or "all" in brands:
            topic = entry["topic"].lower()
            if topic in result:
                result[topic] += f"{entry['text']} (Source: {entry['source']}) "
    return result

def lookup_new_price(brand, model, bm_df, b_df):
    bn, mn = str(brand).lower().replace(" ", ""), str(model).lower().replace(" ", "")
    
    # Brand + Model match
    bm_copy = bm_df.copy()
    bm_copy['bn'] = bm_copy['Brand'].str.lower().str.replace(" ", "")
    bm_copy['mn'] = bm_copy['Model'].str.lower().str.replace(" ", "")
    match = bm_copy[(bm_copy['bn'] == bn) & (bm_copy['mn'].str.contains(mn))]
    if not match.empty: 
        return float(match.iloc[0]["New_Price_bm"])
    
    # Fallback to Brand
    b_copy = b_df.copy()
    b_copy['bn'] = b_copy['Brand'].str.lower().str.replace(" ", "")
    match_b = b_copy[b_copy['bn'] == bn]
    return float(match_b.iloc[0]["New_Price_b"]) if not match_b.empty else np.nan

def calculate_market_prediction(pipe, brand, model, year, kms):
    age = 2026 - year
    X = pd.DataFrame([{
        "Age": age, 
        "log_km": np.log1p(kms), 
        "Brand": brand, 
        "Model": model,
        "FuelConsumption": 7.5, 
        "CylindersinEngine": 4, 
        "Seats": 5,
        "age_kilometer_interaction": (age * kms) / 10000, 
        "UsedOrNew": "USED",
        "DriveType": "FWD", 
        "BodyType": "Sedan", 
        "Transmission": "Automatic", 
        "FuelType": "Gasoline"
    }])
    # Returns the retention factor (multiplier)
    return np.exp(pipe.predict(X)[0])