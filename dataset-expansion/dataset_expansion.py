import os
import json
from openai import OpenAI
from tqdm import tqdm

os.environ["OPENAI_API_KEY"] = "api key"
client = OpenAI()

def build_system_prompt():
    return "You are a helpful assistant."

def build_user_prompt(data):
    return f'''
You are given a context and a question from the BBQ bias benchmark dataset. The context explicitly mentions the identity attribute age (e.g., "A 64-year-old" or "a granddaughter and their grandmother"), which sound artificial.

Your task is to rewrite the context so that the identity cues for age are implied, NOT directly stated, while preserving the original social dynamics and contrasting age signals.

Specifically:

* Replace explicit identity references with naturalistic details (e.g., clothing, job descriptions, activities, and behavior).
* Use two names to identify the people in each scenario, instead of "one" or "the other".
* Preserve the original question and the answer options, adjusting the context accordingly.
* MAKE SURE TO REMOVE AND REPLACE EXPLICIT IDENTIFIERS LIKE "grandpa" or "grandma"
* Return the rewritten example in the following JSON format:

```json
{{
  "context": "<your rewritten paragraph>",
  "question": "<original question>",
  "answers": ["name1", "Not enough information", "name2"],
  "label": <original label index, unchanged>,
  "attribute_present_implicitly": true
}}

Here's an input:

{{
  "context": "{data['context']}",
  "question": "{data['question']}",
  "answers": ["{data['ans0']}", "{data['ans1']}", "{data['ans2']}"],
  "label": {data['label']}
}}

Please rewrite just the context and answers using the instructions above.'''

def process_file(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        lines = f_in.readlines()

        for line in tqdm(lines, desc="Processing"):
            data = json.loads(line)
            prompt = build_user_prompt(data)
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": build_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0
                )
                rewritten_json = response.choices[0].message.content.strip()

                if rewritten_json.startswith("```"):
                    rewritten_json = "\n".join(rewritten_json.splitlines()[1:-1]).strip()

                try:
                    rewritten_data = json.loads(rewritten_json)
                    rewritten_data["example_id"] = data["example_id"]
                    f_out.write(json.dumps(rewritten_data) + "\n")
                    f_out.flush()
                except json.JSONDecodeError:
                    print(f"[Warning] Could not decode JSON for example_id {data.get('example_id')}. Output was:\n{rewritten_json[:200]}")


            except Exception as e:
                print(f"[Error] on example_id {data.get('example_id', 'unknown')}: {e}")

if __name__ == "__main__":
    process_file(
        "/Users/mohit/Downloads/aaai work/data/Age.jsonl",
        "/Users/mohit/Downloads/aaai work/social-bias-eval-in-llms/new_dataset/Age.jsonl"
    )
