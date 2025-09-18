import requests
import time

API_ENDPOINT = "https://api.perplexity.ai/chat/completions"
API_KEY = os.getenv("PERPLEXITY_API_KEY")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}
MODEL_NAME = "sonar"

def generate_low_answer_and_scores(question):
    prompt = (
        f"You are asked an Excel interview question but want to respond with an incomplete, incorrect, or 'I don't know' kind of answer.\n"
        f"Reply concisely to this question: '{question}'.\n"
        f"Then provide scores as JSON (0-10) for: correctness, clarity, terminology, efficiency.\n"
        f"Output ONLY valid JSON like:\n"
        f'{{"answer": "your answer here", "correctness": 1.0, "clarity": 1.0, "terminology": 1.0, "efficiency": 1.0}}\n'
    )
    
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "Be concise and output only JSON."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150,
    }
    
    response = requests.post(API_ENDPOINT, headers=HEADERS, json=data)
    if response.status_code == 200:
        result = response.json()
        try:
            content = result["choices"][0]["message"]["content"].strip()
            return content  # JSON string
        except Exception:
            return None
    else:
        print(f"API error {response.status_code}: {response.text}")
        return None

# Example usage with your dataset
import pandas as pd
import json

df = pd.read_json("excel_qa_clean_core_scores.json")
augmented_samples = []

for idx, row in df.iterrows():
    question = row["question"]
    type_ = row["type"]

    response_str = generate_low_answer_and_scores(question)
    time.sleep(0.5)  # API rate limit

    if response_str:
        try:
            obj = json.loads(response_str)
            answer = obj.get("answer", "")
            correctness = float(obj.get("correctness", 1.0))
            clarity = float(obj.get("clarity", 1.0))
            terminology = float(obj.get("terminology", 1.0))
            efficiency = float(obj.get("efficiency", 1.0))

            augmented_samples.append({
                "question": question,
                "answer": answer,
                "type": type_,
                "correctness": correctness,
                "clarity": clarity,
                "terminology": terminology,
                "efficiency": efficiency,
            })
        except json.JSONDecodeError:
            print(f"Failed to parse JSON for question: {question}")
        except Exception as e:
            print(f"Error processing response for question: {question} - {e}")

# Create DataFrame and merge
df_aug = pd.DataFrame(augmented_samples)
df_final = pd.concat([df, df_aug], ignore_index=True)
df_final.to_json("excel_qa_augmented_dynamic_with_scores.json", indent=2)
print(f"Augmented dataset size: {len(df_final)}")
