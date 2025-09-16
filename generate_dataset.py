from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import time
import json
import re
from dotenv import load_dotenv

# Load the correct T5-small model
model_name = "bigscience/bloomz-560m"# Correct identifier
print("Loading bloomz-560m model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def create_prompt(question, answer, level):
    return f"""
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

def query_local_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Raw model response: {response}")  # Debug output
    
    # Extract numbers from response - only return if we get valid scores
    try:
        numbers = re.findall(r'\d+(?:\.\d+)?', response)
        numbers = [float(n) for n in numbers if 0 <= float(n) <= 10]  # Only valid scores 0-10
        
        # We need at least 4 valid scores
        if len(numbers) >= 4:
            # Extract keywords (meaningful words from response)
            words = re.findall(r'[a-zA-Z]+', response.lower())
            keywords = ' '.join([w for w in words if len(w) > 3 and w not in ['question', 'answer', 'level', 'rate', 'excel']][:5])
            
            return {
                "correctness": numbers[0],
                "clarity": numbers[1],
                "terminology": numbers[2],
                "efficiency": numbers[3],
                "keywords": keywords if keywords else None
            }
    except Exception as e:
        print(f"Parsing error: {e}")
    
    # Return None if we can't extract valid scores
    return None

if __name__ == "__main__":
    load_dotenv()
    
    csv_file = "excel_qa_cleaned.csv"
    df = pd.read_csv(csv_file)
    annotated_records = []
    failed_count = 0
    
    print(f"Processing {len(df)} rows with t5-small...")
    
    for idx, row in df.iterrows():
        question = row.get('Question', '')
        answer = row.get('Answer', '')
        level = row.get('Level', '')
        
        if pd.isna(answer) or not answer.strip():
            print(f"Skipping row {idx+1}: empty answer")
            continue
            
        prompt = create_prompt(question, answer, level)
        evaluation = query_local_model(prompt)
        
        if evaluation is not None:
            record = {
                "question": question,
                "answer": answer,
                "level": level,
                "correctness": evaluation.get("correctness"),
                "clarity": evaluation.get("clarity"), 
                "terminology": evaluation.get("terminology"),
                "efficiency": evaluation.get("efficiency"),
                "keywords": evaluation.get("keywords")
            }
            annotated_records.append(record)
            print(f"✅ Row {idx+1} annotated successfully: {evaluation.get('correctness')}/{evaluation.get('clarity')}/{evaluation.get('terminology')}/{evaluation.get('efficiency')}")
        else:
            failed_count += 1
            print(f"❌ Row {idx+1} failed: No valid scores extracted")
        
        time.sleep(0.1)
    
    # Save only successful annotations
    if annotated_records:
        pd.DataFrame(annotated_records).to_csv("annotated_excel_qa_bloomz.csv", index=False)
        with open("annotated_excel_qa_bloomz.json", "w") as f:
            json.dump(annotated_records, f, indent=2)
        
        print(f"\n✅ Annotation complete!")
        print(f"✅ Successfully annotated: {len(annotated_records)} records")
        print(f"❌ Failed annotations: {failed_count} records")
        print(f"📈 Success rate: {len(annotated_records)/(len(df)-failed_count)*100:.1f}%")
    else:
        print(f"\n❌ No valid annotations generated. Consider using a different model.")
