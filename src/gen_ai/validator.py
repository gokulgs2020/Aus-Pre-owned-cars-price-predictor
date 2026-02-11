# validator.py
import re
import pandas as pd
import difflib
from datetime import datetime
CURRENT_YEAR = datetime.now().year

def parse_numeric(value):
    if value in [None, "null", "-", "None"]: return None
    if isinstance(value, (int, float)): return float(value)
    v = str(value).lower().replace("kms", "").replace("km", "").replace(",", "").replace("$", "").strip()
    nums = re.findall(r"\d+", v)
    return float("".join(nums)) if nums else None

def validate_data_plausibility(brand, model, year, kms, price):
    warnings = []
    year = parse_numeric(year)
    kms = parse_numeric(kms)
    price = parse_numeric(price)
    
    if not all([year, kms, price]):
        return warnings

    if year < 1950 or year > CURRENT_YEAR:
        warnings.append("Year is implausible")
        return warnings
    
    age = max(1, CURRENT_YEAR - year) 
    km_per_year = kms / age
    
    if year >= CURRENT_YEAR-4 and price < 10000:
        warnings.append(f"⚠️ **Price Alert:** AU ${price:,.0f} for a {year} model is suspiciously low. Verify if this is a scam.")
    if km_per_year > 25000:
        warnings.append(f"🏎️ **High Usage:** This {brand} has averaged {int(km_per_year):,} km/year; more than 2x the Australian avg (~13,000km/yr; Source: ABS).")
    if year <= CURRENT_YEAR-2 and kms < 1500:
        warnings.append(f"🔍 **Suspiciously Low Kms:** Only {kms:,.0f} km on a {year} model. Please verify the odometer.")
    return warnings

def validate_model_existence(brand, model, brand_model_lookup):
    """
    STRICT HIERARCHICAL LOOKUP :
    1. Direct Exact Match (Case-insensitive)
    2. Normalized Match (Alpha-numeric only)
    3. Brand-Locked Fuzzy Match (difflib)
    """
    if not model or not brand:
        return False, {"status": "missing", "message": "Brand and Model are required"}

    # --- HELPER: NORMALIZE ---
    def normalize(text):
        return "".join(char for char in str(text).lower() if char.isalnum())

    user_brand_norm = normalize(brand)
    user_model_norm = normalize(model)
    user_model_raw = str(model).strip().lower()

    # Create working copy of reference data
    temp_df = brand_model_lookup.copy()
    # Create raw lowercase columns for "Soft" matching (handles hyphens/spaces better)
    temp_df['raw_brand'] = temp_df['Brand'].str.strip().str.lower()
    temp_df['raw_model'] = temp_df['Model'].str.strip().str.lower()
    # Create fully normalized columns for "Hard" matching
    temp_df['norm_brand'] = temp_df['raw_brand'].apply(normalize)
    temp_df['norm_model'] = temp_df['raw_model'].apply(normalize)

    # --- STEP 1: RESOLVE BRAND ---
    brand_list_norm = temp_df['norm_brand'].unique()
    resolved_brand_norm = None
    
    if user_brand_norm in brand_list_norm:
        resolved_brand_norm = user_brand_norm
    else:
        # Fuzzy match brand only if user provided one that isn't exact
        brand_matches = difflib.get_close_matches(user_brand_norm, brand_list_norm, n=1, cutoff=0.7)
        if brand_matches:
            resolved_brand_norm = brand_matches[0]

    # --- STEP 2: BRAND-LOCKED SEARCH (NO ESCAPE) ---
    if resolved_brand_norm:
        brand_filtered_df = temp_df[temp_df['norm_brand'] == resolved_brand_norm]
        actual_brand_name = brand_filtered_df.iloc[0]['Brand']
        
        # 2a. Direct Exact Match (e.g., 'CR-V' matches 'CR-V')
        exact_match = brand_filtered_df[brand_filtered_df['raw_model'] == user_model_raw]
        if not exact_match.empty:
            return True, {"brand": actual_brand_name, "model": exact_match.iloc[0]['Model'], "status": "valid"}

        # 2b. Normalized Match (e.g., 'CRV' matches 'CR-V')
        norm_match = brand_filtered_df[brand_filtered_df['norm_model'] == user_model_norm]
        if not norm_match.empty:
            return True, {"brand": actual_brand_name, "model": norm_match.iloc[0]['Model'], "status": "valid"}
        
        # 2c. Fuzzy match ONLY within this brand
        model_list_norm = brand_filtered_df['norm_model'].unique()
        model_matches = difflib.get_close_matches(user_model_norm, model_list_norm, n=1, cutoff=0.6)
        
        if model_matches:
            match_row = brand_filtered_df[brand_filtered_df['norm_model'] == model_matches[0]].iloc[0]
            return True, {
                "brand": actual_brand_name,
                "model": match_row['Model'],
                "status": "corrected"
            }
        else:
            return False, {
                "status": "not_in_db", 
                "message": f"I recognize '{actual_brand_name}', but '{model}' is not in my dataset."
            }

    return False, {"status": "not_in_db", "message": f"Unsupported Brand: '{brand}'"}