import os
import json
import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime
from openai import OpenAI

# =====================================================
# CONFIG (FIRST STREAMLIT CALL)
# =====================================================
st.set_page_config(
    page_title="Preowned Car Price Estimator",
    layout="centered"
)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not set")
    st.stop()

client = OpenAI(api_key=api_key)

# =====================================================
# HELPERS
# =====================================================
def parse_numeric(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        v = value.lower().strip()
        if v.startswith("my") and v[2:].isdigit():
            return float("20" + v[2:])
        v = (
            v.replace("kms", "")
             .replace("km", "")
             .replace(",", "")
             .replace("$", "")
             .replace("k", "000")
        )
        nums = re.findall(r"\d+", v)
        return float("".join(nums)) if nums else None

    return None

def safe_json_parse(text: str):
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
    return json.loads(t.strip())

def clean_llm_markdown(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\u200b", "").replace("\u00a0", " ")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n\n".join(lines)

def normalize_text(x):
    return (
        str(x).lower().strip()
        .replace("-", "")
        .replace(" ", "")
        .replace("_", "")
    )

# =====================================================
# LOAD ML ARTIFACTS
# =====================================================
@st.cache_resource
def load_artifacts():
    pipe = joblib.load("models/final_price_pipe.joblib")
    bm = pd.read_csv("models/new_price_lookup_bm.csv")
    b = pd.read_csv("models/new_price_lookup_b.csv")
    lookup = pd.read_csv("models/brand_model_lookup_50.csv")

    for df_ in (bm, b, lookup):
        for c in ("Brand", "Model"):
            if c in df_.columns:
                df_[c] = df_[c].astype(str).str.strip()

    return pipe, bm, b, lookup

pipe, bm, b, brand_model_lookup = load_artifacts()

# =====================================================
# LOAD MARKET SOURCES (CITATIONS)
# =====================================================
@st.cache_resource
def load_market_sources():
    with open("data/market_sources.json", "r", encoding="utf-8") as f:
        return json.load(f)

MARKET_SOURCES = load_market_sources()

def get_market_sources_for_brand(brand: str):
    brand_l = brand.lower()
    result = {
        "resale": [],
        "maintenance": [],
        "reliability": [],
        "depreciation": []
    }

    for entry in MARKET_SOURCES:
        brands = [b.lower() for b in entry.get("brands", [])]
        if brand_l in brands or "all" in brands:
            topic = entry["topic"].lower()
            if topic in result:
                result[topic].append(
                    f"{entry['text']} (Source: {entry['source']})"
                )

    return result

# =====================================================
# PRICE LOOKUP
# =====================================================
def lookup_new_price(brand, model):
    bn, mn = normalize_text(brand), normalize_text(model)

    bm_ = bm.assign(
        Brand_n=bm["Brand"].apply(normalize_text),
        Model_n=bm["Model"].apply(normalize_text)
    )

    row = bm_[
        (bm_["Brand_n"] == bn) &
        (bm_["Model_n"].str.contains(mn))
    ]

    if not row.empty:
        return float(row.iloc[0]["New_Price_bm"])

    b_ = b.assign(Brand_n=b["Brand"].apply(normalize_text))
    row = b_[b_["Brand_n"] == bn]

    return float(row.iloc[0]["New_Price_b"]) if not row.empty else np.nan

# =====================================================
# FEATURE BUILDER
# =====================================================
def make_features(brand, model, age, km):
    return pd.DataFrame([{
        "Age": age,
        "log_km": np.log1p(km),
        "FuelConsumption": 7.5,
        "CylindersinEngine": 4,
        "Seats": 5,
        "age_kilometer_interaction": (age * km) / 10000,
        "Brand": brand,
        "Model": model,
        "UsedOrNew": "USED",
        "DriveType": "FWD",
        "BodyType": "Sedan",
        "Transmission": "Automatic",
        "FuelType": "Gasoline",
    }])

# =====================================================
# SESSION STATE
# =====================================================
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("vehicle_data", {})

# =====================================================
# UI
# =====================================================
tab1, tab2 = st.tabs(["🚗 Price Estimator", "🤖 Deal Advisor"])

# =====================================================
# TAB 1 — PRICE ESTIMATOR
# =====================================================
with tab1:
    st.header("Price Estimator")

    brand = st.selectbox(
        "Brand",
        sorted(brand_model_lookup["Brand"].unique())
    )

    model = st.selectbox(
        "Model",
        sorted(
            brand_model_lookup[
                brand_model_lookup["Brand"] == brand
            ]["Model"].unique()
        )
    )

    year = st.number_input("Year", 2000, datetime.now().year, 2020)
    km = st.number_input("Kilometres", 0, 200000, 60000)

    if st.button("Estimate Price", key="estimate_price"):
        age = datetime.now().year - year
        new_price = lookup_new_price(brand, model)

        if np.isnan(new_price):
            st.error("No market data for this Brand / Model.")
        else:
            X = make_features(brand, model, age, km)
            retention = float(np.exp(pipe.predict(X)[0]))
            pred_price = retention * new_price

            st.success(f"Estimated Price: AU $ {pred_price:,.0f}")
            st.caption(
                f"Retention: {retention*100:.1f}% | Typical new price: AU $ {new_price:,.0f}"
            )

# =====================================================
# TAB 2 — DEAL ADVISOR (GEN AI)
# =====================================================
with tab2:
    st.header("Deal Advisor")

    if st.button("🔄 Restart", key="restart_chat"):
        st.session_state.chat_history = []
        st.session_state.vehicle_data = {}
        st.rerun()

    user_input = st.chat_input(
        "Paste listing (Brand, Model, Year, Kms, Listed price in AUD)"
    )

    if user_input:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )

        extract_prompt = f"""
Extract vehicle details from the text.

Return JSON only:
{{
  "extracted_data": {{
    "Brand": "",
    "Model": "",
    "Year": "",
    "Kilometres": "",
    "Listed Price": ""
  }}
}}

Text:
{user_input}
"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": extract_prompt}],
            temperature=0
        )

        data = safe_json_parse(resp.choices[0].message.content)
        st.session_state.vehicle_data.update(data["extracted_data"])

    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    # -------------------------------------------------
    # VALIDATION
    # -------------------------------------------------
    v = st.session_state.vehicle_data
    brand = v.get("Brand")
    model = v.get("Model")
    year = parse_numeric(v.get("Year"))
    kms = parse_numeric(v.get("Kilometres"))
    listed_price = parse_numeric(v.get("Listed Price"))

    missing = [
        k for k, val in {
            "Brand": brand,
            "Model": model,
            "Year": year,
            "Kilometres": kms,
            "Listed Price": listed_price
        }.items() if not val
    ]

    if missing:
        st.chat_message("assistant").write(
            f"I still need: {', '.join(missing)}."
        )
        st.stop()

    age = datetime.now().year - int(year)
    new_price = lookup_new_price(brand, model)

    if np.isnan(new_price):
        st.chat_message("assistant").write(
            "I don’t have sufficient market data for this Brand / Model."
        )
        st.stop()

    X = make_features(brand, model, age, kms)
    retention = float(np.exp(pipe.predict(X)[0]))
    pred_price = retention * new_price
    gap_pct = round(((listed_price - pred_price) / pred_price) * 100, 1)

    market_ctx = get_market_sources_for_brand(brand)

    explanation_prompt = f"""
You are advising a buyer on THIS listing.

VEHICLE:
{brand} {model}, {age} years, {kms} km

PRICES:
Predicted: AU ${int(pred_price)}
Listed: AU ${int(listed_price)}
Gap: {gap_pct}%

MARKET CONTEXT (CITE INLINE):
Resale:
{" ".join(market_ctx["resale"][:1])}

Reliability:
{" ".join(market_ctx["reliability"][:1])}

Maintenance:
{" ".join(market_ctx["maintenance"][:1])}

Depreciation:
{" ".join(market_ctx["depreciation"][:1])}

FORMAT STRICTLY:

### 💰 Does this listed price make sense?
### 📊 How the listed price compares
### 🧭 What you should do next
"""

    with st.spinner("Generating explanation…"):
        expl = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": explanation_prompt}],
            temperature=0.4
        )

    st.divider()
    st.markdown("### 🧠 Market Explanation")
    st.markdown(clean_llm_markdown(expl.choices[0].message.content))


