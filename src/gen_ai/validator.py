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
    STRICT HIERARCHICAL LOOKUP:
    1. Resolve Brand first.
    2. Search for Model ONLY within that Brand to prevent cross-brand hallucinations.
    """
    if not model:
        return False, {"status": "missing", "message": "Model is required"}
    
    def normalize(text):
        return "".join(char for char in str(text).lower() if char.isalnum())

    user_model_norm = normalize(model)
    user_brand_norm = normalize(brand) if brand else ""

    # Pre-calculate normalized columns
    temp_df = brand_model_lookup.copy()
    temp_df['norm_model'] = temp_df['Model'].apply(normalize)
    temp_df['norm_brand'] = temp_df['Brand'].apply(normalize)

    # --- STEP 1: RESOLVE BRAND FIRST ---
    brand_list = temp_df['norm_brand'].unique()
    
    # Check if the user-provided brand exists (fuzzy match if needed)
    resolved_brand_norm = None
    if user_brand_norm in brand_list:
        resolved_brand_norm = user_brand_norm
    else:
        brand_matches = difflib.get_close_matches(user_brand_norm, brand_list, n=1, cutoff=0.7)
        if brand_matches:
            resolved_brand_norm = brand_matches[0]

    # --- STEP 2: SEARCH WITHIN BRAND ---
    if resolved_brand_norm:
        brand_filtered_df = temp_df[temp_df['norm_brand'] == resolved_brand_norm]
        
        # Check for exact model match within brand
        exact_match = brand_filtered_df[brand_filtered_df['norm_model'] == user_model_norm]
        if not exact_match.empty:
            return True, {
                "brand": exact_match.iloc[0]['Brand'], 
                "model": exact_match.iloc[0]['Model'], 
                "status": "valid"
            }
        
        # Check for fuzzy model match WITHIN brand only
        model_list = brand_filtered_df['norm_model'].unique()
        model_matches = difflib.get_close_matches(user_model_norm, model_list, n=1, cutoff=0.6)
        
        if model_matches:
            match_row = brand_filtered_df[brand_filtered_df['norm_model'] == model_matches[0]].iloc[0]
            return True, {
                "brand": match_row['Brand'],
                "model": match_row['Model'],
                "status": "corrected"
            }
        else:
            # Found the Brand, but Model doesn't exist for it
            return False, {"status": "not_in_db", "message": f"We support {brand}, but couldn't find a model matching '{model}'."}

    # --- STEP 3: FALLBACK (If Brand is unknown/missing) ---
    # Only if we can't find the brand do we look at the whole model list
    all_models = temp_df['norm_model'].unique()
    closest_names = difflib.get_close_matches(user_model_norm, all_models, n=1, cutoff=0.8) # Higher cutoff for safety
    
    if closest_names:
        match_row = temp_df[temp_df['norm_model'] == closest_names[0]].iloc[0]
        return True, {
            "brand": match_row['Brand'],
            "model": match_row['Model'],
            "status": "corrected"
        }

    return False, {"status": "not_in_db", "message": f"Unsupported Vehicle: '{brand} {model}'"}