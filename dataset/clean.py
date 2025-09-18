import json

input_file = "excel_qa_with_scores.json"
output_file = "excel_qa_clean_core_scores.json"

with open(input_file, "r") as f:
    data = json.load(f)

cleaned = []
for record in data:
    question = record.get("question")
    answer = record.get("answer")
    type_ = record.get("type")

    scores_info = record.get("scores")

    if not scores_info:
        continue

    try:
        # Extract the JSON string inside choices[0].message.content
        choices = scores_info.get("choices", [])
        if not choices:
            continue
        message_content = choices[0].get("message", {}).get("content", "")
        score_dict = json.loads(message_content)

        # Extract our needed fields
        correctness = score_dict.get("correctness")
        clarity = score_dict.get("clarity")
        terminology = score_dict.get("terminology")
        efficiency = score_dict.get("efficiency")

        cleaned.append({
            "question": question,
            "answer": answer,
            "type": type_,
            "correctness": correctness,
            "clarity": clarity,
            "terminology": terminology,
            "efficiency": efficiency
        })
    except Exception as e:
        print(f"Error parsing scores for question: {question}\n{e}")

with open(output_file, "w") as f:
    json.dump(cleaned, f, indent=2)

print(f"Extracted core scores for {len(cleaned)} records saved to {output_file}")
