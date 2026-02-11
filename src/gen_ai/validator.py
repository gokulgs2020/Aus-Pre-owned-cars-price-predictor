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
    # Ensure values are numeric before calculation
    year = parse_numeric(year)
    kms = parse_numeric(kms)
    price = parse_numeric(price)
    
    if not all([year, kms, price]):
        return warnings

    if year < 1950 or year > CURRENT_YEAR:
        warnings.append("Year is implausible")
        return warnings   # stop further logic if invalid
    
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
    Checks if model exists. Resolves exact matches and fuzzy matches 
    (e.g., 'santa' -> 'Santa Fe').
    """
    if not model:
        return False, {"status": "missing", "message": "Model is required"}
    
    # 1. Clean basic noise
    if len(str(model)) < 2 or not any(c.isalpha() for c in str(model)):
        return False, {"status": "rubbish", "message": f"'{model}' isn't a valid name"}

    def normalize(text):
        return "".join(char for char in str(text).lower() if char.isalnum())

    user_model_norm = normalize(model)
    user_brand_norm = normalize(brand) if brand else ""

    # Pre-calculate normalized columns
    temp_df = brand_model_lookup.copy()
    temp_df['norm_model'] = temp_df['Model'].apply(normalize)
    temp_df['norm_brand'] = temp_df['Brand'].apply(normalize)

    # 2. Try Exact Match First
    model_matches = temp_df[temp_df['norm_model'] == user_model_norm]

    # 3. Fuzzy Match Fallback (Handles 'santa' -> 'santafe' or 'corola' -> 'corolla')
    if model_matches.empty:
        all_models = temp_df['norm_model'].unique()
        # n=1 gets the top match, cutoff=0.6 is the similarity threshold
        closest_names = difflib.get_close_matches(user_model_norm, all_models, n=1, cutoff=0.5)
        
        if closest_names:
            model_matches = temp_df[temp_df['norm_model'] == closest_names[0]]
        else:
            return False, {"status": "not_in_db", "message": f"Unsupported Model: '{model}'"}

    # 4. Final Brand Resolution
    brand_match = model_matches[model_matches['norm_brand'] == user_brand_norm]
    
    if not brand_match.empty:
        return True, {
            "brand": brand_match.iloc[0]['Brand'], 
            "model": brand_match.iloc[0]['Model'], 
            "status": "valid"
        }
    else:
        # Corrected: Found the model, but assigning the correct canonical brand
        return True, {
            "brand": model_matches.iloc[0]['Brand'], 
            "model": model_matches.iloc[0]['Model'], 
            "status": "corrected"
        }