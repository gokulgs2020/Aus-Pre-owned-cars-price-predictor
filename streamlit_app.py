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
    if not text:
        raise ValueError("Empty LLM response")

    t = text.strip()

    # remove ```json fences if present
    if t.startswith("```"):
        parts = t.split("```")
        t = parts[1] if len(parts) > 1 else parts[0]
        t = t.strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()

    return json.loads(t)


# =====================================================
# PERCEPTION TOOLS
# =====================================================
def tool_reliability(brand, model):
    if brand.lower() == "toyota":
        return "Toyota vehicles are widely regarded for strong long-term mechanical reliability."
    return "No unusual reliability risks compared with typical vehicles in this category."


def tool_maintenance(brand, model):
    if brand.lower() == "toyota":
        return "Toyota vehicles benefit from widespread service availability and relatively low maintenance friction."
    return "Maintenance requirements are broadly in line with segment expectations."


def tool_resale(brand, model):
    if brand.lower() == "toyota":
        return "Toyota models typically command stronger used-market demand, supporting resale value."
    return "Resale value reflects general market demand rather than strong brand premiums."


def tool_depreciation(brand, model):
    return "Vehicles typically experience higher depreciation in early years followed by gradual stabilisation."


# =====================================================
# LOAD ARTIFACTS
# =====================================================
@st.cache_resource
def load_artifacts():
    pipe = joblib.load("models/final_price_pipe.joblib")
    bm = pd.read_csv("models/new_price_lookup_bm.csv")
    b = pd.read_csv("models/new_price_lookup_b.csv")
    lookup = pd.read_csv("models/brand_model_lookup_50.csv")
    return pipe, bm, b, lookup


pipe, bm, b, brand_model_lookup = load_artifacts()


def normalize_text(x):
    return str(x).lower().replace(" ", "").replace("-", "").replace("_", "")


def lookup_new_price(brand, model):
    brand_n = normalize_text(brand)
    model_n = normalize_text(model)

    bm["Brand_n"] = bm["Brand"].apply(normalize_text)
    bm["Model_n"] = bm["Model"].apply(normalize_text)
    b["Brand_n"] = b["Brand"].apply(normalize_text)

    match = bm[(bm["Brand_n"] == brand_n) & (bm["Model_n"].str.contains(model_n))]
    if not match.empty:
        return float(match["New_Price_bm"].iloc[0])

    match = b[b["Brand_n"] == brand_n]
    if not match.empty:
        return float(match["New_Price_b"].iloc[0])

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
# TAB 1 — PRICE ESTIMATOR
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

    if st.button("Estimate Price", key="estimate_price"):
        age = datetime.now().year - year
        new_price = lookup_new_price(brand, model)

        if np.isnan(new_price):
            st.error("No market data available for this Brand / Model.")
        else:
            X = make_features(brand, model, age, km)
            retention = float(np.exp(pipe.predict(X)[0]))
            price = retention * new_price

            st.success(f"Estimated Price: AU ${price:,.0f}")
            st.caption(f"Retention: {retention*100:.1f}% | New price ≈ AU ${new_price:,.0f}")


# =====================================================
# TAB 2 — DEAL ADVISOR
# =====================================================
with tab2:
    st.header("Deal Advisor")

    if st.button("🔄 Restart Conversation", key="restart_chat"):
        st.session_state.vehicle_data = {}
        st.rerun()

    user_input = st.chat_input(
        "Paste listing details (Brand, Model, Year, Kilometres, Listed price in AUD)"
    )

    # -----------------------------
    # 1. Extract listing details
    # -----------------------------
    if user_input:
        extract_prompt = f"""
Extract vehicle details and return STRICT JSON only.

Message:
{user_input}

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

        try:
            st.session_state.vehicle_data = safe_json_parse(
                resp.choices[0].message.content
            )
        except Exception:
            st.error("Could not extract listing details.")
            st.code(resp.choices[0].message.content)
            st.stop()

    data = st.session_state.vehicle_data

    brand = data.get("Brand")
    model = data.get("Model")
    year = parse_numeric(data.get("Year"))
    km = parse_numeric(data.get("Kilometres"))
    listed_price = parse_numeric(data.get("ListedPrice"))

    missing = []
    if not brand: missing.append("Brand")
    if not model: missing.append("Model")
    if year is None: missing.append("Year")
    if km is None: missing.append("Kilometres")
    if listed_price is None: missing.append("Listed Price")

    if missing:
        st.chat_message("assistant").write(f"I still need: {', '.join(missing)}.")
        st.stop()

    # -----------------------------
    # 2. Run valuation
    # -----------------------------
    age = datetime.now().year - int(year)
    new_price = lookup_new_price(brand, model)

    if np.isnan(new_price):
        st.chat_message("assistant").write(
            "I don’t have sufficient market data for this vehicle."
        )
        st.stop()

    X = make_features(brand, model, age, km)
    retention = float(np.exp(pipe.predict(X)[0]))
    predicted_price = retention * new_price
    gap_pct = round((listed_price - predicted_price) / predicted_price * 100, 1)

    st.chat_message("assistant").write(
        f"Got it 👍 {brand} {model}, {km:,} km, listed at AU ${int(listed_price):,}. Evaluating…"
    )

    # -----------------------------
    # 3. Ask LLM for structured explanation
    # -----------------------------
    explanation_prompt = f"""
Return STRICT JSON only. No markdown, no prose.

{{
  "verdict": "",
  "price_rationale": ["", ""],
  "gap_analysis": ["", ""],
  "next_steps": ["", ""]
}}

Brand: {brand}
Model: {model}
Age: {age}
Kilometres: {km}
PredictedPrice: {int(predicted_price)}
RetentionPercent: {round(retention*100,1)}
ListedPrice: {int(listed_price)}
GapPercent: {gap_pct}

Reliability: {tool_reliability(brand, model)}
Maintenance: {tool_maintenance(brand, model)}
Resale: {tool_resale(brand, model)}
Depreciation: {tool_depreciation(brand, model)}
"""

    with st.spinner("Analysing the deal…"):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": explanation_prompt}],
            temperature=0.4
        )

    try:
        expl = safe_json_parse(resp.choices[0].message.content)
    except Exception:
        st.error("Model returned invalid JSON.")
        st.code(resp.choices[0].message.content)
        st.stop()

    # -----------------------------
    # 4. Render UI
    # -----------------------------
    st.divider()
    st.markdown("### 🧠 Market Explanation")

    st.markdown(f"**Verdict:** {expl['verdict']}")

    st.markdown("**Why this price makes sense (or does not)**")
    for b in expl["price_rationale"]:
        st.markdown(f"- {b}")

    st.markdown("**How the listed price compares**")
    for b in expl["gap_analysis"]:
        st.markdown(f"- {b}")

    st.markdown("**What you should do next**")
    for b in expl["next_steps"]:
        st.markdown(f"- {b}")
