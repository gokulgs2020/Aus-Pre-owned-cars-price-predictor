import json

def call_critic_agent(client, vehicle_data, ml_prediction, gap_percentage, report_text):
    """
    Strengthened Auditor: Uses Chain of Verification to detect entity drift and numeric errors.
    """
    CRITIC_SYSTEM_PROMPT = """
    You are a Senior Data Auditor. Your mission is to reject any report that contains factual 
    drift, hallucinations, or logic errors. 

    STRICT AUDIT RULES:
    1. ENTITY DRIFT: Ensure the Brand and Model in the report match the GROUND TRUTH exactly. 
       (e.g., If Ground Truth is 'Renault' but report says 'Holden', hallucination_score = 0).
    2. NUMERIC DRIFT: Check that the Year, Kms, and Price quoted in the report match the 
       Ground Truth. No 'rounding errors' allowed.
    3. VERDICT CONSISTENCY: 
       - Gap < -5%: Must be 'BARGAIN' or 'VERY LOW'
       - -5% <= Gap <= 5%: Must be 'FAIR PRICED'
       - Gap > 5%: Must be 'OVER PRICED'
    4. FEATURE HALLUCINATION: Identify if the AI 'invented' car features (e.g., 'leather seats', 
       'sunroof') that were not provided in the vehicle_data.

    CHAIN OF VERIFICATION (Internal Monologue):
    - Identify the Brand/Model in the Report. Does it match Ground Truth?
    - Identify the Price and Kms in the Report. Do they match Ground Truth?
    - Calculate if the Verdict matches the Math Gap.

    OUTPUT FORMAT:
    You must output ONLY a valid JSON object:
    {
        "hallucination_score": float (0-1),
        "recall_score": float (0-1),
        "precision_score": float (0-1),
        "found_hallucinations": [list],
        "missing_facts": [list],
        "verdict_consistent": bool,
        "audit_justification": "Short summary of findings"
    }
    """

    user_prompt = f"""
    GROUND TRUTH DATA (THE BIBLE):
    - Brand: {vehicle_data['Brand']}
    - Model: {vehicle_data['Model']}
    - Year: {vehicle_data['Year']}
    - Kilometres: {vehicle_data['Kilometres']}
    - Listed Price: ${vehicle_data['Listed Price']}
    - ML Predicted Market Value: ${ml_prediction:,.0f}
    - Calculated Gap %: {gap_percentage:.1f}%

    REPORT TO AUDIT:
    ---
    {report_text}
    ---
    """

    response = client.chat.completions.create(
        model="gpt-4o", # Upgraded to 4o for higher reasoning capability during audit
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    return json.loads(response.choices[0].message.content)