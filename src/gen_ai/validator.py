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
    """Checks if the brand/model combo exists and resolves to canonical name."""
    if not brand or not model:
        return False, "missing"
    
    # 1. Rubbish Check
    if len(str(model)) < 2 or not any(c.isalpha() for c in str(model)):
        return False, "rubbish"

    # Helper to normalize strings: lowercase + remove non-alphanumeric
    def normalize(text):
        return "".join(char for char in str(text).lower() if char.isalnum())

    user_model_norm = normalize(model)
    
    # 2. Database Lookup with Normalization
    # Filter for the correct brand first (case-insensitive)
    brand_mask = brand_model_lookup["Brand"].str.lower() == brand.lower()
    valid_models = brand_model_lookup[brand_mask]["Model"].unique()
    
    # Compare normalized versions
    for canonical_m in valid_models:
        if normalize(canonical_m) == user_model_norm:
            # RETURN TRUE + THE CANONICAL NAME 
            # (e.g., if user gave 'CRV', this returns 'CR-V')
            return True, canonical_m
    
    return False, "not_in_db"