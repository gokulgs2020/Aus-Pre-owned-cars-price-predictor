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
# CONFIG (must be first Streamlit call)
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


def safe_json_parse(text: str):
    """
    Parses JSON returned by the LLM, handling ```json fences if present.
    """
    t = (text or "").strip()
    if not t:
        raise ValueError("Empty LLM response")

    if t.startswith("```"):
        parts = t.split("```")
        t = parts[1] if len(parts) > 1 else parts[0]
        t = t.strip()
        if t.startswith("json"):
            t = t[4:].strip()

    return json.loads(t.strip())


def clean_llm_markdown(text: str) -> str:
    """
    Keeps markdown structure, removes empty lines, avoids per-character line breaks.
    """
    if not text:
        return ""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


# =====================================================
# PERCEPTION TOOLS (controlled text)
# =====================================================
def tool_reliability(brand, model):
    if str(brand).lower() == "toyota":
        return (
            "Toyota models are widely regarded for mechanical reliability and long-term durability, "
            "with fewer major ownership issues than many peers."
        )
    return (
        "There are no strong signals suggesting unusual reliability risks; expectations generally align with segment norms."
    )


def tool_maintenance(brand, model):
    if str(brand).lower() == "toyota":
        return (
            "Toyota benefits from a broad service network, readily available spare parts, and relatively low ownership friction."
        )
    return "Maintenance and servicing requirements are broadly in line with segment expectations."


def tool_resale(brand, model):
    if str(brand).lower() == "toyota":
        return "Toyota vehicles typically command strong used-market demand, supporting above-average resale value."
    return "Resale demand generally reflects overall segment dynamics rather than brand-specific premiums."


def tool_depreciation(brand, model):
    return (
        "Vehicles in this segment typically experience steeper depreciation in early years, followed by gradual value stabilisation over time."
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

    # normalize keys to reduce lookup mismatches
    for df_ in (bm, b, lookup):
        if "Brand" in df_.columns:
            df_["Brand"] = df_["Brand"].astype(str).str.strip()
        if "Model" in df_.columns:
            df_["Model"] = df_["Model"].astype(str).str.strip()

    return pipe, bm, b, lookup


pipe, bm, b, brand_model_lookup = load_artifacts()

def normalize_text(x):
    if not x:
        return ""
    return (
        str(x)
        .lower()
        .strip()
        .replace("-", "")
        .replace(" ", "")
        .replace("_", "")
    )


# =====================================================
# PRICE LOOKUP
# =====================================================
def lookup_new_price(brand, model):
    brand_n = normalize_text(brand)
    model_n = normalize_text(model)

    bm_copy = bm.copy()
    b_copy = b.copy()

    bm_copy["Brand_n"] = bm_copy["Brand"].apply(normalize_text)
    bm_copy["Model_n"] = bm_copy["Model"].apply(normalize_text)
    b_copy["Brand_n"] = b_copy["Brand"].apply(normalize_text)

    # 1️⃣ Model-level match (contains handles trims)
    model_match = bm_copy[
        (bm_copy["Brand_n"] == brand_n) &
        (bm_copy["Model_n"].str.contains(model_n))
    ]

    if not model_match.empty:
        return float(model_match["New_Price_bm"].iloc[0])

    # 2️⃣ Brand-level fallback
    brand_match = b_copy[b_copy["Brand_n"] == brand_n]

    if not brand_match.empty:
        return float(brand_match["New_Price_b"].iloc[0])

    return float("nan")



# =====================================================
# FEATURE BUILDER (keep as-is for your model)
# =====================================================
def make_features(brand, model, age, km):
    return pd.DataFrame([{
        "Age": float(age),
        "log_km": float(np.log1p(km)),
        "FuelConsumption": 7.5,
        "CylindersinEngine": 4,
        "Seats": 5,
        "age_kilometer_interaction": (float(age) * float(km)) / 10000.0,
        "Brand": str(brand).strip(),
        "Model": str(model).strip(),
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

    brands = sorted(brand_model_lookup["Brand"].dropna().unique())
    brand = st.selectbox("Brand", brands)

    models = sorted(
        brand_model_lookup[brand_model_lookup["Brand"] == brand]["Model"]
        .dropna()
        .unique()
    )
    model = st.selectbox("Model", models)

    year = st.number_input("Year", 2000, datetime.now().year, 2020)
    km = st.number_input("Kilometres", 0, 200000, 60000)

    if st.button("Estimate Price"):
        age = datetime.now().year - int(year)
        new_price = lookup_new_price(brand, model)

        if np.isnan(new_price):
            st.error("Brand/Model not in training data")
        else:
            X = make_features(brand, model, age, km)
            log_ret = float(pipe.predict(X)[0])
            retention = float(np.exp(log_ret))
            pred_price = float(retention * new_price)

            st.success(f"Estimated Price: AU $ {pred_price:,.0f}")
            st.caption(
                f"Retention: {retention * 100:.1f}% | Typical new price: AU $ {new_price:,.0f}"
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

    user_input = st.chat_input(
        "Paste the listing details (Brand, Model, Year, Kms, Listed price in AUD)."
    )

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        extract_prompt = f"""
Extract vehicle details from the message.

Current known data:
{st.session_state.vehicle_data}

Message:
{user_input}

Required fields (extract if present):
Brand, Model, Year, Kilometres, Listed Price

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
Do not include markdown code fences.
"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": extract_prompt}],
            temperature=0
        )

        try:
            parsed_llm = safe_json_parse(resp.choices[0].message.content)
            st.session_state.vehicle_data.update(parsed_llm.get("extracted_data", {}))
        except Exception:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "I couldn’t extract the details. Please rephrase with Brand, Model, Year, Kms, and Listed Price."
            })

    # show chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # -------------------------------------------------
    # VALIDATION & GATING (including Listed Price)
    # -------------------------------------------------
    parsed = {}
    missing_fields = []

    # Brand & Model
    for field in ["Brand", "Model"]:
        v = st.session_state.vehicle_data.get(field)
        if isinstance(v, str) and v.strip():
            parsed[field] = v.strip()
        else:
            missing_fields.append(field)

    # Year, Kilometres, Listed Price (numeric)
    year = parse_numeric(st.session_state.vehicle_data.get("Year"))
    kms = parse_numeric(st.session_state.vehicle_data.get("Kilometres"))
    listed_price = parse_numeric(st.session_state.vehicle_data.get("Listed Price"))

    if year is None:
        missing_fields.append("Year")
    else:
        parsed["Year"] = int(year)

    if kms is None:
        missing_fields.append("Kilometres")
    else:
        parsed["Kilometres"] = int(kms)

    if listed_price is None:
        missing_fields.append("Listed Price")
    else:
        parsed["Listed Price"] = int(listed_price)

    if missing_fields:
        st.chat_message("assistant").write(
            f"I still need: {', '.join(missing_fields)}."
        )
        st.stop()

    # -------------------------------------------------
    # RUN VALUATION + EXPLANATION
    # -------------------------------------------------
    brand = parsed["Brand"]
    model = parsed["Model"]
    year = parsed["Year"]
    kms = parsed["Kilometres"]
    listed_price = parsed["Listed Price"]

    age = datetime.now().year - int(year)
    new_price = lookup_new_price(brand, model)


    if np.isnan(new_price):
        st.chat_message("assistant").write(
            "I don’t have sufficient market data for this Brand / Model. Try a more common Brand/Model."
        )
        st.stop()

    st.chat_message("assistant").write(
        f"Got it 👍 {brand} {model}, {kms:,} km, listed at AU $ {listed_price:,}. Evaluating…"
    )

    X = make_features(brand, model, age, kms)
    log_ret = float(pipe.predict(X)[0])
    retention = float(np.exp(log_ret))
    pred_price = float(retention * new_price)
    gap_pct = round(((listed_price - pred_price) / pred_price) * 100, 1)

    explanation_prompt = f"""
You are an automotive market advisor supporting a used-car buyer.

STRICT RULES:
- Use ONLY the information provided below
- Do NOT invent facts, statistics, URLs, or assumptions
- Keep the response professional, concise, and structured
- Use clean Markdown exactly as requested

VEHICLE DETAILS
Brand: {brand}
Model: {model}
Vehicle Age (years): {age}
Kilometres Driven: {kms}

PRICING CONTEXT
Typical New Price: AU ${int(new_price)}
Model-Predicted Price: AU ${int(pred_price)}
Implied Retention: {round(retention*100,1)}%
Seller Listed Price: AU ${int(listed_price)}
Price Gap vs Prediction: {gap_pct}%

MARKET CONTEXT (REFERENCE INFORMATION)
Reliability:
{tool_reliability(brand, model)}

Maintenance & After-Sales:
{tool_maintenance(brand, model)}

Resale Perception:
{tool_resale(brand, model)}

Depreciation Pattern:
{tool_depreciation(brand, model)}

OUTPUT RULES:
- Use headings and bullets exactly in the format below
- Each section MUST have exactly 2 bullet points
- Each bullet point MUST be one sentence only

FORMAT RULES (STRICT):

- Use full sentences.
- Do NOT split numbers across lines.
- Use standard currency formatting (e.g. AU $25,125).
- Use clear section headings.
- Do NOT insert extra line breaks inside numbers.
- Do NOT use excessive markdown.


FORMAT (mandatory):

#### 🧾 Verdict on the deal
**<ONE of: Great Bargain | Good Offer | Fair / On Par with Market | Slightly Overpriced | Significantly Overpriced>**

#### 💰 Does the listed price make sense?
You are explaining THIS specific vehicle to THIS buyer.

You must:
- Refer explicitly to the vehicle by brand and model
- Refer explicitly to the predicted price, retention %, age and kilometres
- Avoid generic language like "this segment" without tying it back to the vehicle
- Speak as if advising a real buyer evaluating this exact listing

When explaining prices, always anchor statements to:
“for this {brand} {model} with {age} years and {kms} km…”

#### 📊 How the listed price compares
- ...
- ...

#### 🧭 What you should do next
- ...
- ...
"""

    with st.spinner("Generating explanation…"):
        expl = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": explanation_prompt}],
            temperature=0.4
        )

    st.divider()
    st.markdown("### 🧠 Market Explanation")

    cleaned = clean_llm_markdown(expl.choices[0].message.content)
    st.markdown(cleaned,unsafe_allow_html=False)
