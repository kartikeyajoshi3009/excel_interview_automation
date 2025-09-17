import requests
import pandas as pd
import json
import time
import os

API_ENDPOINT = os.getenv("PERPLEXITY_API_ENDPOINT")
API_KEY = os.getenv("PERPLEXITY_API_KEY")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

MODEL_NAME = "sonar"

def generate_scores(question, answer, type_):
    prompt = f"""
You are an expert Excel interviewer. Evaluate this answer on criteria (0-10):
correctness, clarity, terminology, efficiency.
Return only JSON like:
{{
  "correctness": 0.0-10.0,
  "clarity": 0.0-10.0,
  "terminology": 0.0-10.0,
  "efficiency": 0.0-10.0
}}

Question: "{question}"
Answer: "{answer}"
Type: "{type_}"
"""
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "Be precise and concise."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 150,
    }
    response = requests.post(API_ENDPOINT, headers=HEADERS, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"API error {response.status_code}: {response.text}")
        return None

def main():
    df = pd.read_csv("excel_qa_cleaned.csv")
    results = []
    for idx, row in df.iterrows():
        question = row.get("Question", "")
        answer = row.get("Answer", "")
        type_ = row.get("Type", "") or row.get("Level", "")

        if not answer or not isinstance(answer, str) or not answer.strip():
            print(f"Skipping row {idx+1} due to empty answer")
            continue

        print(f"Processing row {idx+1}: {question[:50]}...")
        scores = generate_scores(question, answer, type_)

        if scores:
            record = {
                "question": question,
                "answer": answer,
                "type": type_,
                "scores": scores,
            }
            results.append(record)

        time.sleep(0.5)

    with open("excel_qa_with_scores.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Completed annotating {len(results)} records")

if __name__ == "__main__":
    main()
