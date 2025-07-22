import os
import json
from openai import OpenAI
from tqdm import tqdm
import re
import argparse

client = OpenAI()

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
"""

def get_model_prediction(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def extract_predicted_index(model_output):
    # Try to extract 'Answer: <label>' from the model output
    match = re.search(r'label\s*[:\-]?\s*(\d)', model_output, re.IGNORECASE)
    if match:
        return int(match.group(1))

def evaluate_model_on_dataset(input_file, results_file):
    correct = 0
    total = 0
    # Open the output file once, and write each result as soon as it's ready
    with open(input_file, "r") as fin, open(results_file, "w") as fout:
        for line in tqdm(fin, desc="Evaluating"):
            example = json.loads(line)
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

    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.3f} ({correct}/{total})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evaluate_model_on_dataset(args.input, args.output) 