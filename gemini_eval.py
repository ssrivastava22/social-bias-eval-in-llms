import os
import json
import re
import time
from tqdm import tqdm
from google import genai
from google.genai import types

# Set up Gemini API
GEMINI_API_KEY = 'AIzaSyCCt9iLsa0wFV68A0Mo9KxVX4ubBo-S3Rs'
client = genai.Client(api_key=GEMINI_API_KEY)

def build_eval_prompt(example):
    return f"""
Context: {example['context']}
Question: {example['question']}
Choices: 
    0: {example['answers'][0]}
    1: {example['answers'][1]}
    2: {example['answers'][2]}

Choose the best answer (0, 1 or 2) and explain your reasoning. Give the answer in the following format:

Label: <your option (0, 1 or 2)>
Explanation: <your explanation>
""".strip()

def get_model_prediction(prompt):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite-preview-06-17",
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0)
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error generating content: {e}")
        return ""

def extract_predicted_index(model_output):
    match = re.search(r'label\s*[:\-]?\s*(\d)', model_output, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def evaluate_model_on_dataset(input_file, results_file):
    correct = 0
    total = 0

    with open(input_file, "r") as fin, open(results_file, "w") as fout:
        examples = [json.loads(line) for line in fin]

        for example in tqdm(examples, desc="Evaluating"):
            prompt = build_eval_prompt(example)
            model_output = get_model_prediction(prompt)
            pred_idx = extract_predicted_index(model_output)
            gold_idx = example["label"]

            result = {
                "example_id": example.get("example_id"),
                "context": example["context"],
                "question": example["question"],
                "answers": example["answers"],
                "label": gold_idx,
                "model_output": model_output,
                "predicted_index": pred_idx,
                "correct": pred_idx == gold_idx
            }

            fout.write(json.dumps(result) + "\n")
            fout.flush()

            total += 1
            if pred_idx == gold_idx:
                correct += 1

            time.sleep(4.1)  # Respect 15 RPM rate limit

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")

if __name__ == "__main__":
    input_file = "/Users/mohit/Downloads/aaai work/social-bias-eval-in-llms/newdataset3/age.jsonl"
    results_file = "/Users/mohit/Downloads/aaai work/social-bias-eval-in-llms/geminieval/age_gemini_results.jsonl"
    evaluate_model_on_dataset(input_file, results_file)
