import anthropic
import pandas as pd
import json
import time
import re

raw = pd.read_csv("final/data_processed/cleaned_data.csv")

client = anthropic.Anthropic(api_key='your api')

VALID_FRAMES = {
    "ECONOMIC", "SECURITY", "TECHNOLOGICAL",
    "ENVIRONMENTAL", "INFRASTRUCTURE", "ELECTRICITY", "OTHER"
}
VALID_SENTIMENTS = {"POSITIVE", "NEGATIVE", "NEUTRAL"}

def classify_frame(text):
    prompt = f"""You are analyzing elite framing of data center policy.

Read the following government statement or news article and identify:
1. The PRIMARY frame used
2. The overall sentiment toward data centers

FRAME — choose ONE:
1. ECONOMIC: focuses on jobs, investment, GDP growth, economic development, tax revenue
2. ENVIRONMENTAL: focuses on carbon footprint, sustainability, pollution, water usage
3. INFRASTRUCTURE: focuses on physical development, regional planning, connectivity
4. ELECTRICITY: focuses on electricity consumption, power grid, energy costs, utility rates
5. OTHER: does not fit above categories

SENTIMENT — choose ONE:
- POSITIVE: favorable or supportive tone toward data centers
- NEGATIVE: critical, concerned, or oppositional tone
- NEUTRAL: factual or balanced with no clear evaluative stance

Text:
{text[:1500]}

Respond ONLY in JSON format with no extra text or markdown:
{{
    "frame": "ECONOMIC or SECURITY or TECHNOLOGICAL or ENVIRONMENTAL or INFRASTRUCTURE or ELECTRICITY or OTHER",
    "frame_confidence": 0.0,
    "key_phrases": ["phrase1", "phrase2", "phrase3"],
    "frame_reasoning": "one sentence explanation",
    "sentiment": "POSITIVE or NEGATIVE or NEUTRAL",
    "sentiment_confidence": 0.0,
    "sentiment_reasoning": "one sentence explanation"
}}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}]
        )

        raw_text = response.content[0].text.strip()
        raw_text = re.sub(r"```json|```", "", raw_text).strip()
        result = json.loads(raw_text)

        if result.get("frame") not in VALID_FRAMES:
            result["frame"] = "OTHER"
        if result.get("sentiment") not in VALID_SENTIMENTS:
            result["sentiment"] = "NEUTRAL"

        return result

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print("Raw response:", response.content[0].text[:200])
        return {
            "frame": "ERROR",
            "frame_confidence": 0.0,
            "key_phrases": [],
            "frame_reasoning": f"JSON parse error: {e}",
            "sentiment": "ERROR",
            "sentiment_confidence": 0.0,
            "sentiment_reasoning": f"JSON parse error: {e}"
        }
    except Exception as e:
        print(f"API error: {e}")
        return {
            "frame": "ERROR",
            "frame_confidence": 0.0,
            "key_phrases": [],
            "frame_reasoning": str(e),
            "sentiment": "ERROR",
            "sentiment_confidence": 0.0,
            "sentiment_reasoning": str(e)
        }


results = []
for i, row in raw.iterrows():
    print(f"Processing {i+1}/{len(raw)}...")
    result = classify_frame(row["clean_text"])
    result["index"] = i
    results.append(result)
    time.sleep(0.5)

df_out = raw.copy()
df_out["frame"]               = [r.get("frame", "ERROR")               for r in results]
df_out["frame_confidence"]    = [r.get("frame_confidence", 0.0)         for r in results]
df_out["key_phrases"]         = [r.get("key_phrases", [])               for r in results]
df_out["frame_reasoning"]     = [r.get("frame_reasoning", "")           for r in results]
df_out["sentiment"]           = [r.get("sentiment", "ERROR")            for r in results]
df_out["sentiment_confidence"]= [r.get("sentiment_confidence", 0.0)     for r in results]
df_out["sentiment_reasoning"] = [r.get("sentiment_reasoning", "")       for r in results]

df_out.to_csv("final/data_processed/framing_sentiment_results.csv", index=False)

print("\n--- Frame Distribution ---")
print(df_out["frame"].value_counts())
print("\n--- Sentiment Distribution ---")
print(df_out["sentiment"].value_counts())
print("\n--- Sentiment by Frame ---")
print(df_out.groupby("frame")["sentiment"].value_counts().unstack(fill_value=0))

n_errors = (df_out["frame"] == "ERROR").sum()
if n_errors > 0:
    print(f"\nWarning: {n_errors} documents failed to classify")
    print(df_out[df_out["frame"] == "ERROR"][["clean_text", "frame_reasoning"]])
