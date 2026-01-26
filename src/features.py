import numpy as np
import pandas as pd
from .config import NEAR_NEW_KM, RET_CLIP_LOW, RET_CLIP_HIGH

def apply_filters(df: pd.DataFrame, max_price: int, max_km: int, min_year: int) -> pd.DataFrame:
    df = df.copy()
    if "Year" in df.columns:
        df = df[df["Year"] >= min_year]
    df = df[(df["Price"] <= max_price) & (df["Kilometres"] <= max_km)]
    return df

def build_new_price_lookups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    near_new = df[df["Kilometres"] <= NEAR_NEW_KM].copy()

    # Brand-Model proxy
    bm = (
        near_new.groupby(["Brand", "Model"])["Price"]
        .quantile(0.95)
        .reset_index()
        .rename(columns={"Price": "New_Price_bm"})
    )

    # Brand proxy fallback
    b = (
        near_new.groupby(["Brand"])["Price"]
        .quantile(0.95)
        .reset_index()
        .rename(columns={"Price": "New_Price_b"})
    )

    return bm, b

def add_features(df: pd.DataFrame, bm_lookup: pd.DataFrame, b_lookup: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Merge lookups
    df = df.merge(bm_lookup, on=["Brand", "Model"], how="left")
    df = df.merge(b_lookup, on=["Brand"], how="left")

    df["New_Price"] = df["New_Price_bm"].fillna(df["New_Price_b"])
    df = df[df["New_Price"].notna()].copy()

    # Retention + log target
    df["retention"] = (df["Price"] / df["New_Price"]).clip(lower=RET_CLIP_LOW, upper=RET_CLIP_HIGH)
    df["y"] = np.log(df["retention"])

    # Non-linear km + interaction
    df["log_km"] = np.log1p(df["Kilometres"])
    df["age_kilometer_interaction"] = (df["Age"] * df["Kilometres"]) / 10000

    # Group key to avoid leakage (same Brand+Model in train & test)
    df["BrandModelGroup"] = df["Brand"].astype(str) + "||" + df["Model"].astype(str)

    return df
