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

        # handle MY20, MY21
        if v.startswith("my") and v[2:].isdigit():
            return float("20" + v[2:])

        v = v.replace("kms", "").replace("km", "")
        v = v.replace(",", "").replace("$", "")
        v = v.replace("k", "000")

        nums = re.findall(r"\d+", v)
        if nums:
            return float("".join(nums))

    return None


def safe_json_parse(text):
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
    return json.loads(t.strip())

# =====================================================
# PERCEPTION TOOLS
# =====================================================
def tool_reliability(brand, model):
    if brand.lower() == "toyota":
        return (
            "Toyota models are widely regarded for mechanical reliability and "
            "long-term durability, with fewer major ownership issues than many peers."
        )
    return (
        "There are no strong signals suggesting unusual reliability risks; "
        "expectations generally align with segment norms."
    )


def tool_maintenance(brand, model):
    if brand.lower() == "toyota":
        return (
            "Toyota benefits from a broad service network, readily available spare parts, "
            "and relatively low ownership friction."
        )
    return (
        "Maintenance and servicing requirements are broadly in line with segment expectations."
    )


def tool_resale(brand, model):
    if brand.lower() == "toyota":
        return (
            "Toyota vehicles typically command strong used-market demand, "
            "supporting above-average resale value."
        )
    return (
        "Resale demand generally reflects overall segment dynamics rather than brand-specific premiums."
    )


def tool_depreciation(brand, model):
    return (
        "Vehicles in this segment typically experience steeper depreciation in early years, "
        "followed by gradual value stabilisation over time."
    )

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

# =====================================================
# PRICE LOOKUP
# =====================================================
def lookup_new_price(brand, model):
    row_bm = bm[(bm["Brand"] == brand) & (bm["Model"] == model)]
    if len(row_bm):
        return float(row_bm["New_Price_bm"].iloc[0])

    row_b = b[b["Brand"] == brand]
    if len(row_b):
        return float(row_b["New_Price_b"].iloc[0])

    return float("nan")

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
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vehicle_data" not in st.session_state:
    st.session_state.vehicle_data = {}

# =====================================================
# UI TABS
# =====================================================
tab1, tab2 = st.tabs(["🚗 Price Estimator", "🤖 Deal Advisor"])

# =====================================================
# TAB 1 — ESTIMATOR
# =====================================================
with tab1:
    st.header("Price Estimator")

    brands = sorted(brand_model_lookup["Brand"].unique())
    brand = st.selectbox("Brand", brands)

    models = sorted(
        brand_model_lookup[brand_model_lookup["Brand"] == brand]["Model"].unique()
    )
    model = st.selectbox("Model", models)

    year = st.number_input("Year", 2000, datetime.now().year, 2020)
    km = st.number_input("Kilometres", 0, 200000, 60000)

    if st.button("Estimate Price"):
        age = datetime.now().year - year
        new_price = lookup_new_price(brand, model)

        if np.isnan(new_price):
            st.error("Brand/Model not in training data")
        else:
            X = make_features(brand, model, age, km)
            log_ret = float(pipe.predict(X)[0])
            retention = np.exp(log_ret)
            pred_price = retention * new_price

            st.success(f"Estimated Price: AU $ {pred_price:,.0f}")
            st.caption(
                f"Retention %: {retention:.2d} | Typical Price of a new car: AU $ {new_price:,.0f}"
            )

# =====================================================
# TAB 2 — DEAL ADVISOR
# =====================================================
with tab2:
    st.header("Deal Advisor")

    if st.button("🔄 Restart Conversation"):
        st.session_state.chat_history = []
        st.session_state.vehicle_data = {}
        st.rerun()

    user_input = st.chat_input("Paste the details of your the car listing you are interested in, kindly include Brand, Model, Kms driven and listed price")

    if user_input:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )

        extract_prompt = f"""
Extract vehicle details from the message.

Current known data:
{st.session_state.vehicle_data}

Message:
{user_input}

Required:
Brand, Model, Year, Kilometres, Listed Price

Return JSON only:
{{"extracted_data": {{}}}}
"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": extract_prompt}],
            temperature=0
        )

        try:
            parsed_llm = safe_json_parse(resp.choices[0].message.content)
            st.session_state.vehicle_data.update(
                parsed_llm.get("extracted_data", {})
            )
        except:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "I couldn’t understand that. Could you rephrase please?"
            })

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # -------------------------------------------------
    # VALIDATION & GATING
    # -------------------------------------------------
    parsed = {}
    invalid_fields = []

    # Brand & Model
    for field in ["Brand", "Model"]:
        v = st.session_state.vehicle_data.get(field)
        if isinstance(v, str) and v.strip():
            parsed[field] = v.strip()
        else:
            invalid_fields.append(field)

    # Year & Kilometres
    year = parse_numeric(st.session_state.vehicle_data.get("Year"))
    kms = parse_numeric(st.session_state.vehicle_data.get("Kilometres"))

    if year is None:
        invalid_fields.append("Year")
    else:
        parsed["Year"] = int(year)

    if kms is None:
        invalid_fields.append("Kilometres")
    else:
        parsed["Kilometres"] = int(kms)

    if invalid_fields:
        st.chat_message("assistant").write(
            f"Please include the following to continue: {', '.join(invalid_fields)}"
        )
    else:
        brand = parsed["Brand"]
        model = parsed["Model"]
        year = parsed["Year"]
        kms = parsed["Kilometres"]

        age = datetime.now().year - year
        new_price = lookup_new_price(brand, model)

        if np.isnan(new_price):
            st.chat_message("assistant").write(
                "I don’t have sufficient market data for this Brand / Model."
            )
        else:
            X = make_features(brand, model, age, kms)
            log_ret = float(pipe.predict(X)[0])
            retention = np.exp(log_ret)
            pred_price = retention * new_price

            # -------------------------------------------------
        # LISTED PRICE — MUST BE EXPLICIT
        # -------------------------------------------------
        extracted_lp = parse_numeric(st.session_state.vehicle_data.get("Listed Price"))

        if extracted_lp is not None:
            st.info(f"Listed price detected from listing: AU $ {int(extracted_lp):,}")
            listed_price = float(extracted_lp)
            manual_override = False
        else:
            st.warning("⚠️ Listed price not found in the message.")
            listed_price = st.number_input(
                "Please enter the seller's listed price",
                min_value=0,
                step=500
            )
            manual_override = True

        # Stop execution until user provides it
        if not listed_price or listed_price <= 0:
            st.stop()


            gap_pct = round(
                (listed_price - pred_price) / pred_price * 100, 1
            )

            explanation_prompt = f"""
You are an automotive market advisor.

VEHICLE
Brand: {brand}
Model: {model}
Age: {age}
Kilometres: {kms}

PRICING
New Price: {int(new_price)}
Predicted Price: {int(pred_price)}
Retention (%): {round(retention*100,1)}
Listed Price: {int(listed_price)}
Gap (%): {gap_pct}

RELIABILITY
{tool_reliability(brand, model)}

MAINTENANCE
{tool_maintenance(brand, model)}

RESALE
{tool_resale(brand, model)}

DEPRECIATION
{tool_depreciation(brand, model)}

Explain in not more than 2-3 lines each, use bullet points as needed.
Price, classify deal (Great Bargain/Good offer/On Par with evaluation/Slightly Overpriced/Very expensive) based on the gap,
explain the gap, and advise next steps 
"""

            with st.spinner("Generating explanation…"):
                expl = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": explanation_prompt}],
                    temperature=0.3
                )

            st.divider()
            st.markdown("### 🧠 Market Explanation")
            st.markdown(expl.choices[0].message.content)
