import os
import json
from openai import OpenAI
from tqdm import tqdm
import time

os.environ["OPENAI_API_KEY"] = "your-api-key-here"
client = OpenAI()

def build_system_prompt():
    return "You are a helpful assistant."

def build_user_prompt(data):
    return f'''
You are given an example from the BBQ Bias Benchmark dataset that includes a context, a question, a list of answer choices, and a correct answer label.

The context often explicitly states identity attributes like race and gender (e.g., "a Black man and an Asian woman"), which makes the example feel unnatural.

Your task is to rewrite the context to make identity cues implicit, while keeping the original contrast and social dynamics.

Do the following:

1. Replace explicit identity mentions (e.g., "a Black man", "an Asian woman") with naturalistic indicators like:
   - Names typically associated with race and gender
   - Job descriptions
   - Clothing
   - Activities or lived experiences
   - Interpersonal interactions

2. Use names that subtly convey race and gender (e.g., "Darnell" for a Black man, "Mei" for an Asian woman, "Luis" for a Latino man, "Aaliyah" for a Black woman, etc.).

3. Preserve any realistic or contextual cues that help the social contrast (e.g., workplace behavior, social setting, or everyday activities).

4. Do not change the question or label. Rewrite the answers to reflect the use of names that imply race and gender, instead of phrases like “the Black man” or “the Asian woman.”

Output your rewritten version in the following JSON format:

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

Please rewrite just the context and answers using the instructions above.
'''

def process_file(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        lines = f_in.readlines()

        for line in tqdm(lines, desc="Processing"):
            data = json.loads(line)
            prompt = build_user_prompt(data)

            max_retries = 5
            retry_wait = 60  # default wait in seconds if unspecified

            for attempt in range(max_retries):
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

                    break  # success, exit retry loop

                except Exception as e:
                    error_str = str(e)
                    if "rate_limit_exceeded" in error_str or "Rate limit" in error_str:
                        wait_time = retry_wait

                        # Try to extract "Please try again in Xs" automatically
                        import re
                        match = re.search(r'Please try again in (\d+(\.\d+)?)s', error_str)
                        if match:
                            wait_time = float(match.group(1)) + 5  # add buffer

                        print(f"[Rate Limit] Waiting {wait_time}s before retrying for example_id {data.get('example_id')} (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f"[Error] on example_id {data.get('example_id', 'unknown')}: {e}")
                        break  # do not retry on non-rate-limit errors


if __name__ == "__main__":
    process_file(
        "/Users/aarushiwagh/Documents/Georgia Tech/Sem_2/MLHH/original_data/Sexual_orientation.jsonl",
        "/Users/aarushiwagh/Documents/Georgia Tech/Sem_2/MLHH/extended_data/Sexual_orientation_2.jsonl"
    )