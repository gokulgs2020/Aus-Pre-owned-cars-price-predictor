import json

def call_critic_agent(client, vehicle_data, ml_prediction, gap_percentage, report_text):
    """
    Evaluates the generated report for Hallucination, Recall, and Precision.
    """
    CRITIC_SYSTEM_PROMPT = """
    You are a professional AI Auditor for a car valuation tool. 
    Your job is to compare a 'Generated Report' against 'Ground Truth Data'.
    
    EVALUATION CRITERIA:
    1. HALLUCINATION: Does the report mention features (e.g. 'Sunroof', 'New Tires') NOT provided in the data?
    2. RECALL: Did the report mention all 4 key metrics (Year, Brand, Model, Kms)?
    3. PRECISION: Is the verdict consistent with the Gap %? 
       (e.g., if Gap > 5%, Verdict must be 'Overpriced')

    OUTPUT FORMAT:
    You must output ONLY a valid JSON object with this structure:
    {
        "hallucination_score": float (0 to 1, where 1 is no hallucinations),
        "recall_score": float (0 to 1, where 1 is all facts used),
        "precision_score": float (0 to 1, where 1 is perfectly consistent),
        "found_hallucinations": [list of strings],
        "missing_facts": [list of strings],
        "verdict_consistent": bool
    }
    """

    user_prompt = f"""
    GROUND TRUTH DATA:
    - Vehicle: {vehicle_data['Year']} {vehicle_data['Brand']} {vehicle_data['Model']}
    - Kilometres: {vehicle_data['Kilometres']}
    - Listed Price: ${vehicle_data['Listed Price']}
    - ML Predicted Price: ${ml_prediction:,.0f}
    - Math Gap: {gap_percentage:.1f}%

    GENERATED REPORT:
    ---
    {report_text}
    ---
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini", # Use a faster/cheaper model for auditing
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    return json.loads(response.choices[0].message.content)