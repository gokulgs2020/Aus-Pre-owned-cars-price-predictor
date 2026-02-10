import re
import pandas as pd
import numpy as np
from datetime import datetime

def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def extract_int(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.extract(r"(\d+)")[0]
    return pd.to_numeric(s, errors="coerce")

def extract_float(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.extract(r"(\d+\.?\d*)")[0]
    return pd.to_numeric(s, errors="coerce")

def normalize_transmission(x: str) -> str:
    if pd.isna(x):
        return "Unknown"
    x = str(x).lower()
    if "auto" in x:
        return "Automatic"
    if "man" in x:
        return "Manual"
    return "Other"

def normalize_fuel(x: str) -> str:
    if pd.isna(x):
        return "Unknown"
    x = str(x).lower()
    if "diesel" in x:
        return "Diesel"
    if "electric" in x:
        return "Electric"
    if "hybrid" in x:
        return "Hybrid"
    if "petrol" in x or "gas" in x or "unleaded" in x or "premium" in x:
        return "Gasoline"
    return "Other"

def clean_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- numeric coercions ---
    if "Kilometres" in df.columns:
        df["Kilometres"] = _to_numeric(df["Kilometres"])
    if "Price" in df.columns:
        df["Price"] = _to_numeric(df["Price"])
    if "Year" in df.columns:
        df["Year"] = _to_numeric(df["Year"])

    # --- Age handling ---
    if "Age" in df.columns:
        df["Age"] = _to_numeric(df["Age"])

    # If Age is missing but Year exists, create Age
    if "Age" not in df.columns and "Year" in df.columns:
        current_year = datetime.now().year
        df["Age"] = current_year - df["Year"]

    # Clean Age (avoid negatives)
    if "Age" in df.columns:
        df["Age"] = df["Age"].clip(lower=0)

    # Seats: extract digits
    if "Seats" in df.columns:
        df["Seats"] = extract_int(df["Seats"]).fillna(0).astype(int)

    # Fuel consumption: extract float
    if "FuelConsumption" in df.columns:
        df["FuelConsumption"] = extract_float(df["FuelConsumption"])
        df["FuelConsumption"] = df["FuelConsumption"].fillna(df["FuelConsumption"].median())

    # Cylinders: extract digits
    if "CylindersinEngine" in df.columns:
        df["CylindersinEngine"] = extract_int(df["CylindersinEngine"]).fillna(0).astype(int)

    # Normalize categoricals
    if "Transmission" in df.columns:
        df["Transmission"] = df["Transmission"].apply(normalize_transmission)

    if "FuelType" in df.columns:
        df["FuelType"] = df["FuelType"].apply(normalize_fuel)

    # Trim string cols
    for c in ["Brand", "Model", "BodyType", "DriveType", "UsedOrNew"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Drop rows with missing critical values
    must_have = ["Brand", "Model", "Kilometres", "Age", "Price"]
    for c in must_have:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df = df.dropna(subset=must_have).copy()

    return df
