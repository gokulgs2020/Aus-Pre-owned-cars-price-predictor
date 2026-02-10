# extractor.py
import json

def safe_json_parse(text: str):
    t = (text or "").strip()
    if t.startswith("```"):
        # Remove markdown code blocks
        t = t.split("```")[1]
        if t.startswith("json"):
            t = t[4:]
    try:
        return json.loads(t.strip())
    except json.JSONDecodeError:
        return {}

def call_llm_extractor(client, system_prompt, user_prompt):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    return safe_json_parse(resp.choices[0].message.content)