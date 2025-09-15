import os
import time
import json
import requests
import pandas as pd
from dotenv import load_dotenv
import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto")
# Create prompt with question, answer, level
def create_prompt(question, answer, level):
    prompt = f"""
You are an Excel interview evaluator. 
Evaluate the following answer to the question on four criteria:
- correctness: Is it factually correct? (0 to 10)
- clarity: Is it explained clearly and structured? (0 to 10)
- terminology: Does it use the most correct and relevant terminologies? (0 to 10)
- efficiency: Does it use the most efficient Excel approach? (0 to 10)
Return only JSON in this format:
{{
  "correctness": 0.0-10.0,
  "clarity": 0.0-10.0,
  "terminology": 0.0-10.0,
  "keywords": "relevant keywords used",
  "efficiency": 0.0-10.0
}}

Question: "{question}"
Answer: "{answer}"
Level: "{level}"
"""
    return prompt

# Send prompt to HuggingFace API get response as JSON
def query_huggingface_api(prompt, model_name, hf_api_token=None):
    headers = {"Authorization": f"Bearer {hf_api_token}"} if hf_api_token else {}

    response = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, return_full_text=False)

    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code}, {response.text}")

    result = response.json()

    # Extract generated text
    if isinstance(result, list):
        generated_text = result[0].get("generated_text", "")
    elif isinstance(result, dict):
        generated_text = result.get("generated_text", "")
    else:
        generated_text = str(result)

    # Try extracting JSON from response
    json_start = generated_text.find('{')
    json_end = generated_text.rfind('}') + 1
    if json_start != -1 and json_end != -1:
        json_str = generated_text[json_start:json_end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            raise Exception("Failed to parse JSON from model output.")
    else:
        raise Exception("No JSON object found in model output.")

if __name__ == "__main__":
    load_dotenv()
    api_token = os.getenv("EXCEL_INTERVIEW_AT")

    # ✅ Updated model
    model_name = "HuggingFaceH4/zephyr-7b-beta"

    csv_file = "excel_qa_cleaned.csv"
    df = pd.read_csv(csv_file)

    annotated_records = []

    for idx, row in df.iterrows():
        question = row.get('Question', '')
        answer = row.get('Answer', '')
        level = row.get('Level', '')

        if pd.isna(answer) or answer.strip() == '':
            print(f"Skipping row {idx+1} due to empty answer\n")
            continue

        messages = create_prompt(question, answer, level)
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            evaluation = query_huggingface_api(prompt, model_name, api_token)

            if evaluation:
                record = {
                    "question": question,
                    "answer": answer,
                    "level": level,
                    "correctness": evaluation.get("correctness", None),
                    "clarity": evaluation.get("clarity", None),
                    "terminology": evaluation.get("terminology", None),
                    "efficiency": evaluation.get("efficiency", None),
                    "keywords": evaluation.get("keywords", "")
                }
                annotated_records.append(record)

                print(f"Row {idx+1} annotated successfully")
            else:
                print(f"Row {idx+1} returned no evaluation")

        except Exception as e:
            print(f"Error evaluating row {idx+1}: {str(e)}")

        time.sleep(2)

    # Save annotated data as JSON
    with open("annotated_excel_qa.json", "w", encoding="utf-8") as f_json:
        json.dump(annotated_records, f_json, indent=2)

    # Save annotated data as CSV
    annotated_df = pd.DataFrame(annotated_records)
    annotated_df.to_csv("annotated_excel_qa.csv", index=False)

    print("✅ Annotated data saved to annotated_excel_qa.json and annotated_excel_qa.csv")
