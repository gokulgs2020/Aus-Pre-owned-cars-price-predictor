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

def validate_model_existence(brand, model, brand_model_lookup):
    """
    Checks if model exists and resolves to canonical name. 
    If brand is noisy or wrong, it attempts to infer the correct brand from the model.
    """
    # 1. Base missing check
    if not model:
        return False, {"status": "missing", "message": "Model is required"}
    
    # 2. Rubbish Check
    if len(str(model)) < 2 or not any(c.isalpha() for c in str(model)):
        return False, {"status": "rubbish", "message": f"'{model}' doesn't look like a valid model name"}

    def normalize(text):
        return "".join(char for char in str(text).lower() if char.isalnum())

    user_model_norm = normalize(model)
    user_brand_norm = normalize(brand) if brand else ""

    # 3. GLOBAL SEARCH (Broadening the scope)
    # Optimization: In production, we pre-calculate these normalized columns
    lookup_copy = brand_model_lookup.copy()
    lookup_copy['norm_model'] = lookup_copy['Model'].apply(normalize)
    lookup_copy['norm_brand'] = lookup_copy['Brand'].apply(normalize)

    # Search for the model anywhere in the Australian database
    matches = lookup_copy[lookup_copy['norm_model'] == user_model_norm]

    if not matches.empty:
        # Check if any match also aligns with the provided brand
        brand_match = matches[matches['norm_brand'] == user_brand_norm]
        
        if not brand_match.empty:
            # Case 1: Perfect match found
            return True, {
                "brand": brand_match.iloc[0]['Brand'], 
                "model": brand_match.iloc[0]['Model'], 
                "status": "valid"
            }
        else:
            # Case 2: Model found but brand is different (Auto-correction)
            return True, {
                "brand": matches.iloc[0]['Brand'], 
                "model": matches.iloc[0]['Model'], 
                "status": "corrected"
            }
    
    # 4. Fallback if no model match is found at all
    return False, {"status": "not_in_db", "message": f"Unsupported Model: '{model}'"}