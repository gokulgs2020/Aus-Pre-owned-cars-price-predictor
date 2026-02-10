# prompts.py

SYSTEM_EXTRACTOR = "You are a car data assistant. Update the JSON based on the user's message. Return ONLY JSON."

def get_extraction_prompt(current_data, user_input):
    return f"""
    TODAY'S YEAR: 2026. 
    Current Data: {current_data}
    Message: {user_input}
    
    Update the JSON. If the user mentions age (e.g., '4 years old'), calculate the Year by subtracting that from 2026. 
    Return ONLY JSON.
    """

SYSTEM_ANALYST = "You are a professional Australian market auto-analyst. Use only the provided market data sources."

def get_report_prompt(year, brand, model, kms, price, pred, gap, verdict, m_ctx):
    return f"""
    Analyze this listing: {year} {brand} {model}, {kms:,.0f}km, \${price:,.0f}.
    Our prediction: \${pred:,.0f} (Gap: {gap:.1f}%). 
    Verdict: {verdict}.

    MARKET DATA (CITE SOURCES FROM THIS DATA):
    Resale: {m_ctx['resale']}
    Reliability: {m_ctx['reliability']}
    Maintenance: {m_ctx['maintenance']}
    Depreciation: {m_ctx['depreciation']}

    Format the final report strictly as follows:

    1. **Deal Verdict (2 lines):** Explain the verdict by comparing the gap between our predicted price of \${pred:,.0f} and the listed price of \${price:,.0f}.
    2. **Brand & Model Analysis (3 lines):** Discuss reliability, maintenance, resale value, and depreciation based on the provided market data, explicitly mentioning the sources.
    3. **Our View on Listed Price (2 lines):** Provide a professional opinion on whether the asking price is justified given the car's specific details.
    4. **What You Should Do Next (2-3 lines):** Give actionable advice. If overpriced, suggest waiting or comparing similar listings. If suspiciously low, advise checking for accidents, internal mechanics, or title history.

    Adapt your tone to {brand}. Provide the response in plain conversational English. Always escape dollar signs with a backslash.
    """