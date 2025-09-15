import requests
import json

# Define a function to create the prompt with question, answer, and level
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
  "correctness": 0.0,
  "clarity": 0.0,
  "terminology": 0.0,
  "keywords": "relevant keywords used",
  "efficiency": 0.0
}}

Question: "{question}"
Answer: "{answer}"
Level: "{level}"
"""
    return prompt

# Define a function to send the prompt to HuggingFace Inference API and get the JSON response
def query_huggingface_api(prompt, model_name, hf_api_token=None):
    headers = {}
    if hf_api_token:
        headers["Authorization"] = f"Bearer {hf_api_token}"
        
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 200,
            "temperature": 0.1,
            "return_full_text": False
        }
    }
    
    url = f"https://api-inference.huggingface.co/models/{model_name}"
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code}, {response.text}")
    
    result = response.json()
    
    # Extract generated text either from list or dict response
    if isinstance(result, list):
        generated_text = result[0].get("generated_text", "")
    elif isinstance(result, dict):
        generated_text = result.get("generated_text", "")
    else:
        generated_text = str(result)
        
    # Attempt to extract JSON from generated_text
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

# Example usage:
if __name__ == "__main__":
    question = "What is VLOOKUP?"
    answer = "VLOOKUP searches a value in the first column of a range and returns a value in the same row from another column."
    level = "Intermediate Level Questions"
    model_name = "microsoft/DialoGPT-medium"  # Replace with your preferred HuggingFace model
    hf_api_token = "your_huggingface_api_token"  # Replace with your actual API key or None for public access
    
    prompt = create_prompt(question, answer, level)
    evaluation = query_huggingface_api(prompt, model_name, hf_api_token)
    
    print("Evaluation Output:")
    print(json.dumps(evaluation, indent=2))
