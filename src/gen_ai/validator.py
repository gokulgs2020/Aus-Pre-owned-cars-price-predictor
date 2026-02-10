# validator.py
import re

def parse_numeric(value):
    if value in [None, "null", "-", "None"]: return None
    if isinstance(value, (int, float)): return float(value)
    v = str(value).lower().replace("kms", "").replace("km", "").replace(",", "").replace("$", "").strip()
    nums = re.findall(r"\d+", v)
    return float("".join(nums)) if nums else None

def validate_data_plausibility(brand, model, year, kms, price):
    warnings = []
    # Ensure values are numeric before calculation
    year = parse_numeric(year)
    kms = parse_numeric(kms)
    price = parse_numeric(price)
    
    if not all([year, kms, price]):
        return warnings

    age = max(1, 2026 - year) 
    km_per_year = kms / age
    
    if year >= 2022 and price < 10000:
        warnings.append(f"⚠️ **Price Alert:** AU ${price:,.0f} for a {year} model is suspiciously low. Verify if this is a scam.")
    if km_per_year > 25000:
        warnings.append(f"🏎️ **High Usage:** This {brand} has averaged {int(km_per_year):,} km/year; more than 2x the Australian avg (~13,000km/yr; Source: ABS).")
    if year <= 2024 and kms < 1500:
        warnings.append(f"🔍 **Suspiciously Low Kms:** Only {kms:,.0f} km on a {year} model. Please verify the odometer.")
    return warnings

import pandas as pd

def validate_model_existence(brand, model, brand_model_lookup):
    """
    Checks if model exists and resolves to canonical name. 
    Handles case-insensitivity and special characters (e.g., 'Santa FE' or 'CR-V').
    """
    if not model:
        return False, {"status": "missing", "message": "Model is required"}
    
    if len(str(model)) < 2 or not any(c.isalpha() for c in str(model)):
        return False, {"status": "rubbish", "message": f"'{model}' isn't a valid name"}

    # Helper: Remove non-alphanumeric characters and lowercase
    def normalize(text):
        return "".join(char for char in str(text).lower() if char.isalnum())

    user_model_norm = normalize(model)
    user_brand_norm = normalize(brand) if brand else ""

    # Check for matches
    # Optimization: Instead of copying the whole DF, we filter first
    # This handles "Santa FE" -> "santafe" matching "Santa Fe" -> "santafe"
    
    # Pre-calculate normalized columns for comparison
    temp_df = brand_model_lookup.copy()
    temp_df['norm_model'] = temp_df['Model'].apply(normalize)
    temp_df['norm_brand'] = temp_df['Brand'].apply(normalize)

    # Search for model matches
    model_matches = temp_df[temp_df['norm_model'] == user_model_norm]

    if not model_matches.empty:
        # Check if the brand matches too
        brand_match = model_matches[model_matches['norm_brand'] == user_brand_norm]
        
        if not brand_match.empty:
            # Perfect Match (ignoring case/spaces)
            return True, {
                "brand": brand_match.iloc[0]['Brand'], 
                "model": brand_match.iloc[0]['Model'], 
                "status": "valid"
            }
        else:
            # Model exists but under a different brand (e.g., user said 'Toyota Santa Fe')
            return True, {
                "brand": model_matches.iloc[0]['Brand'], 
                "model": model_matches.iloc[0]['Model'], 
                "status": "corrected"
            }
    
    return False, {"status": "not_in_db", "message": f"Unsupported Model: '{model}'"}