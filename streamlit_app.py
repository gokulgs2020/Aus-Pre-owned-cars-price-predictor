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
# CONFIG
# =====================================================
st.set_page_config(page_title="Preowned Car Deal Advisor", layout="centered")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not set")
    st.stop()

client = OpenAI(api_key=api_key)

# =====================================================
# LOAD MARKET SOURCES (CURATED KB)
# =====================================================
@st.cache_resource
def load_market_sources():
    with open("data/market_sources.json", "r", encoding="utf-8") as f:
        return json.load(f)

MARKET_SOURCES = load_market_sources()

# =====================================================
# HELPERS
# =====================================================
def parse_numeric(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        x = x.lower().replace(",", "").replace("$", "").replace("kms", "").replace("km", "")
        x = x.replace("k", "000")
        nums = re.findall(r"\d+", x)
        if nums:
            return float("".join(nums))
    return None


def normalize_text(x):
    return str(x).lower().replace(" ", "").replace("-", "")


def select_sources(brand, topics):
    """
    Selects relevant market source snippets by brand and topic.
    """
    brand_l = brand.lower()
    selected = []

    for row in MARKET_SOURCES:
        if row["topic"].lower() in topics:
            brands = [b.lower() for b in row["brands"]]
            if "all" in brands or brand_l in brands:
                selected.append(row)

    return selected

# =====================================================
# LOAD MODEL ARTIFACTS
# =====================================================
@st.cache_resource
def load_artifacts():
    pipe = joblib.load("models/final_price_pipe.joblib")
    bm = pd.read_csv("models/new_price_lookup_bm.csv")
    b = pd.read_csv("models/new_price_lookup_b.csv")
    lookup = pd.read_csv("models/brand_model_lookup_50.csv")
    return pipe, bm, b, lookup

pipe, bm, b, brand_model_lookup = load_artifacts()

# =====================================================
# PRICE LOOKUP
# =====================================================
def lookup_new_price(brand, model):
    bn = normalize_text(brand)
    mn = normalize_text(model)

    bm["Brand_n"] = bm["Brand"].apply(normalize_text)
    bm["Model_n"] = bm["Model"].apply(normalize_text)
    b["Brand_n"] = b["Brand"].apply(normalize_text)

    row = bm[(bm["Brand_n"] == bn) & (bm["Model_n"].str.contains(mn))]
    if not row.empty:
        return float(row["New_Price_bm"].iloc[0])

    row = b[b["Brand_n"] == bn]
    if not row.empty:
        return float(row["New_Price_b"].iloc[0])

    return np.nan

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
if "vehicle" not in st.session_state:
    st.session_state.vehicle = {}

# =====================================================
# UI
# =====================================================
st.title("🚗 Preowned Car Deal Advisor")

user_text = st.chat_input(
    "Paste listing (e.g. 'Honda CR-V 2020, 60,000 km, $35,000')"
)

# =====================================================
# STEP 1: EXTRACT VEHICLE DATA
# =====================================================
if user_text:
    extract_prompt = f"""
Extract vehicle details from the text below.
Return STRICT JSON only.

Text:
{user_text}

{{
  "Brand": "",
  "Model": "",
  "Year": "",
  "Kilometres": "",
  "ListedPrice": ""
}}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": extract_prompt}],
        temperature=0
    )

    st.session_state.vehicle = json.loads(resp.choices[0].message.content)

# =====================================================
# STEP 2: VALIDATE
# =====================================================
v = st.session_state.vehicle

brand = v.get("Brand")
model = v.get("Model")
year = parse_numeric(v.get("Year"))
kms = parse_numeric(v.get("Kilometres"))
listed_price = parse_numeric(v.get("ListedPrice"))

missing = []
if not brand: missing.append("Brand")
if not model: missing.append("Model")
if year is None: missing.append("Year")
if kms is None: missing.append("Kilometres")
if listed_price is None: missing.append("Listed Price")

if missing:
    st.info(f"Please include: {', '.join(missing)}")
    st.stop()

# =====================================================
# STEP 3: PRICE PREDICTION
# =====================================================
age = datetime.now().year - int(year)
new_price = lookup_new_price(brand, model)

if np.isnan(new_price):
    st.error("Insufficient pricing data for this Brand / Model.")
    st.stop()

X = make_features(brand, model, age, kms)
retention = float(np.exp(pipe.predict(X)[0]))
pred_price = float(retention * new_price)
gap_pct = round((listed_price - pred_price) / pred_price * 100, 1)

st.success(
    f"Predicted value: AU ${pred_price:,.0f} "
    f"(Listed: AU ${int(listed_price):,}, Gap: {gap_pct}%)"
)

# =====================================================
# STEP 4: SELECT SOURCES
# =====================================================
sources = select_sources(
    brand,
    topics=["reliability", "maintenance", "resale", "depreciation"]
)

sources_text = "\n\n".join(
    f"- {s['topic']} ({s['source']}): {s['text']}"
    for s in sources
)

# =====================================================
# STEP 5: LLM EXPLANATION (JSON OUTPUT)
# =====================================================
explain_prompt = f"""
You are an automotive market advisor.

Use ONLY the information provided.
Cite sources in parentheses.

Vehicle:
{brand} {model}, {age} years, {kms} km

Pricing:
Predicted Price: AU ${int(pred_price)}
Retention: {round(retention*100,1)}%
Listed Price: AU ${int(listed_price)}
Gap: {gap_pct}%

Market sources:
{sources_text}

Return STRICT JSON:

{{
  "verdict": "",
  "why_price": ["", ""],
  "gap_reason": ["", ""],
  "next_steps": ["", ""]
}}
"""

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": explain_prompt}],
    temperature=0.4
)

explanation = json.loads(resp.choices[0].message.content)

# =====================================================
# STEP 6: RENDER
# =====================================================
st.markdown("## 🧠 Market Explanation")

st.markdown(f"**Verdict:** {explanation['verdict']}")

st.markdown("**Why this price makes sense (or does not)**")
for b in explanation["why_price"]:
    st.markdown(f"- {b}")

st.markdown("**How the listed price compares**")
for b in explanation["gap_reason"]:
    st.markdown(f"- {b}")

st.markdown("**What you should do next**")
for b in explanation["next_steps"]:
    st.markdown(f"- {b}")
