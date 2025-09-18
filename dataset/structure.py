import json

input_json_file = "excel_qa_augmented_dynamic_with_scores.json"
output_json_file = "cleaned_output.json"

with open(input_json_file, "r") as f:
    data = json.load(f)

# Get lists of keys (row indices)
indices = list(data["question"].keys())

# Build records by iterating row keys
records = []
for idx in indices:
    rec = {
        "question": data["question"][idx],
        "answer": data["answer"][idx],
        "type": data["type"][idx],
        "correctness": data["correctness"][idx],
        "clarity": data["clarity"][idx],
        "terminology": data["terminology"][idx],
        "efficiency": data["efficiency"][idx],
    }
    records.append(rec)

# Save as JSON list of records
with open(output_json_file, "w") as f:
    json.dump(records, f, indent=2)

print(f"Converted {len(records)} records to flattened JSON saved as {output_json_file}")
