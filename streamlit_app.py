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
st.set_page_config(page_title="Preowned Car Price Estimator", layout="centered")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not set")
    st.stop()

client = OpenAI(api_key=api_key)

# =====================================================
# SOURCE KNOWLEDGE BASE (PROVIDED BY YOU)
# =====================================================
SOURCE_DATA = [
    {
        "source": "Drive.com.au",
        "topic": "Resale Value",
        "brands": ["Nissan","Tesla","Toyota","Suzuki","Hyundai","Isuzu","Mazda","Skoda","Volvo","Jeep","Subaru"],
        "text": "In 2023, several brands including Toyota and Tesla demonstrated strong resale value in Australia, particularly for popular SUVs and EVs."
    },
    {
        "source": "Drive.com.au",
        "topic": "Depreciation",
        "brands": ["Toyota","Hyundai","Mazda"],
        "text": "Toyota generally shows lower depreciation than many competitors, supported by demand and brand reputation."
    },
    {
        "source": "Drive.com.au",
        "topic": "Reliability",
        "brands": ["Toyota","Nissan","Subaru"],
        "text": "Brands like Toyota and Subaru are recognised for dependable performance and longevity."
    },
    {
        "source": "Drive.com.au",
        "topic": "Maintenance",
        "brands": ["Toyota","Hyundai","Mazda"],
        "text": "Toyota and Hyundai are often favoured for lower maintenance costs, supporting used-car appeal."
    },
    {
        "source": "JR Auto Service",
        "topic": "Maintenance",
        "brands": ["Toyota","Mazda","Hyundai","Kia"],
        "text": "Toyota, Mazda and Hyundai are recognised for relatively low maintenance expenses over ownership."
    },
    {
        "source": "MyCarChoice",
        "topic": "Reliability",
        "brands": ["Toyota","Subaru","Honda","Mazda"],
        "text": "Toyota and Subaru lead in reliability perception among used car buyers, supporting buyer confidence."
    },
    {
        "source": "MyCarChoice",
        "topic": "Resale Value",
        "brands": ["Toyota","Mazda","Honda","Subaru","Mitsubishi","Ford","Kia","Volkswagen","Hyundai"],
        "text": "Toyota consistently maintains strong resale value due to reliability and low maintenance costs."
    },
    {
        "source": "MyCarChoice",
        "topic": "Depreciation",
        "brands": ["Toyota","Mazda","Honda","Kia","Hyundai"],
        "text": "Toyota generally depreciates less than peers, while Mazda and Honda also retain value well."
    }
]

# =====================================================
# HELPERS
# =====================================================
def parse_numeric(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        v = value.lower().replace(",", "").replace("$", "").replace("km", "").replace("kms", "")
        v = v.replace("k", "000")
        nums = re.findall(r"\d+", v)
        return float("".join(nums)) if nums else None
    return None


def safe_json_parse(text: str):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1].strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return json.loads(t)


def normalize_text(x):
    return str(x).strip().lower()


# =====================================================
# SOURCE LOOKUP ENGINE (DETERMINISTIC)
# =====================================================
def lookup_source(topic, brand):
    brand = normalize_text(brand)
    for item in SOURCE_DATA:
        if item["topic"].lower() == topic.lower():
            if brand in [b.lower() for b in item["brands"]]:
                return {
                    "text": item["text"],
                    "source": item["source"]
                }
    # fallback
    for item in SOURCE_DATA:
        if item["topic"].lower() == topic.lower():
            return {
                "text": item["text"],
                "source": item["source"]
            }
    return {
        "text": "General market observations apply; brand-specific data is limited.",
        "source": "General market commentary"
    }


def tool_reliability(brand, model):
    return lookup_source("Reliability", brand)


def tool_maintenance(brand, model):
    return lookup_source("Maintenance", brand)


def tool_resale(brand, model):
    return lookup_source("Resale Value", brand)


def tool_depreciation(brand, model):
    return lookup_source("Depreciation", brand)


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
    brand_n = normalize_text(brand)
    model_n = normalize_text(model)

    bm["Brand_n"] = bm["Brand"].apply(normalize_text)
    bm["Model_n"] = bm["Model"].apply(normalize_text)
    b["Brand_n"] = b["Brand"].apply(normalize_text)

    m = bm[(bm["Brand_n"] == brand_n) & (bm["Model_n"].str.contains(model_n))]
    if not m.empty:
        return float(m["New_Price_bm"].iloc[0])

    m = b[b["Brand_n"] == brand_n]
    if not m.empty:
        return float(m["New_Price_b"].iloc[0])

    return np.nan


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
st.session_state.setdefault("vehicle_data", {})

# =====================================================
# UI
# =====================================================
tab1, tab2 = st.tabs(["🚗 Price Estimator", "🤖 Deal Advisor"])

# =====================================================
# TAB 1
# =====================================================
with tab1:
    st.header("Price Estimator")

    brand = st.selectbox("Brand", sorted(brand_model_lookup["Brand"].unique()))
    model = st.selectbox(
        "Model",
        sorted(brand_model_lookup[brand_model_lookup["Brand"] == brand]["Model"].unique())
    )

    year = st.number_input("Year", 2000, datetime.now().year, 2020)
    km = st.number_input("Kilometres", 0, 300000, 60000)

    if st.button("Estimate Price"):
        age = datetime.now().year - int(year)
        new_price = lookup_new_price(brand, model)

        if np.isnan(new_price):
            st.error("No market data available.")
        else:
            X = make_features(brand, model, age, km)
            retention = float(np.exp(pipe.predict(X)[0]))
            price = retention * new_price
            st.success(f"Estimated Price: AU ${price:,.0f}")


# =====================================================
# TAB 2 — DEAL ADVISOR
# =====================================================
with tab2:
    st.header("Deal Advisor")

    user_input = st.chat_input(
        "Paste listing details (Brand, Model, Year, Kilometres, Listed price in AUD)"
    )

    if user_input:
        extract_prompt = f"""
Extract vehicle details and return STRICT JSON only.

{{
  "Brand": "",
  "Model": "",
  "Year": "",
  "Kilometres": "",
  "ListedPrice": ""
}}

Message:
{user_input}
"""
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": extract_prompt}],
            temperature=0
        )

        st.session_state.vehicle_data = safe_json_parse(
            resp.choices[0].message.content
        )

    d = st.session_state.vehicle_data
    if not d:
        st.stop()

    brand = d["Brand"]
    model = d["Model"]
    year = int(parse_numeric(d["Year"]))
    km = int(parse_numeric(d["Kilometres"]))
    listed_price = int(parse_numeric(d["ListedPrice"]))

    age = datetime.now().year - year
    new_price = lookup_new_price(brand, model)

    X = make_features(brand, model, age, km)
    retention = float(np.exp(pipe.predict(X)[0]))
    predicted = retention * new_price
    gap_pct = round((listed_price - predicted) / predicted * 100, 1)

    reliability = tool_reliability(brand, model)
    maintenance = tool_maintenance(brand, model)
    resale = tool_resale(brand, model)
    depreciation = tool_depreciation(brand, model)

    explanation_prompt = f"""
Return STRICT JSON only.

{{
  "verdict": "",
  "price_rationale": ["", ""],
  "gap_analysis": ["", ""],
  "next_steps": ["", ""],
  "citations": {{
    "reliability": "",
    "maintenance": "",
    "resale": "",
    "depreciation": ""
  }}
}}

ReliabilityText: {reliability["text"]}
ReliabilitySource: {reliability["source"]}

MaintenanceText: {maintenance["text"]}
MaintenanceSource: {maintenance["source"]}

ResaleText: {resale["text"]}
ResaleSource: {resale["source"]}

DepreciationText: {depreciation["text"]}
DepreciationSource: {depreciation["source"]}
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": explanation_prompt}],
        temperature=0.3
    )

    expl = safe_json_parse(resp.choices[0].message.content)

    st.markdown("### 🧠 Market Explanation")
    st.markdown(f"**Verdict:** {expl['verdict']}")

    for k in ["price_rationale", "gap_analysis", "next_steps"]:
        for b in expl[k]:
            st.markdown(f"- {b}")

    st.markdown("### 📌 Sources")
    for k, v in expl["citations"].items():
        st.markdown(f"- **{k.title()}**: {v}")
