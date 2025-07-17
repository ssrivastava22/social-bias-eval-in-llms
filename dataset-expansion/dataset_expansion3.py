import os
import json
from openai import OpenAI
from tqdm import tqdm
import time

os.environ["OPENAI_API_KEY"] = "key"
client = OpenAI()

def build_system_prompt():
    return "You are a helpful assistant."

def build_user_prompt(data):
    return f'''
You are given a context and a question from the BBQ bias benchmark dataset. The context explicitly mentions identity attributes such as race and ethnicity (e.g., "a Hispanic man" or "a European man"), which sound artificial.

Your task is to rewrite the context so that the identity cues for race and ethnicity are implied, not directly stated, while preserving the original social dynamics and contrasting race/ethnicity signals.

Specifically:

* Replace explicit identity references with naturalistic details (e.g., clothing, job descriptions, activities, and behavior).
* Use two names to identify the people in each scenario, instead of "one" or "the other".
* Preserve the original question and the answer options, adjusting the context accordingly.
* MAKE SURE TO REMOVE AND REPLACE EXPLICIT IDENTIFIERS LIKE "Jewish" or "African American"
* Return the rewritten example in the following JSON format:

```json
{{
  "context": "<your rewritten paragraph>",
  "question": "<original question>",
  "answers": ["name1", "Not enough information", "name2"],
  "label": <original label index, unchanged>,
  "attribute_present_implicitly": true
}}

* make sure to keep the ORDER of the updated answer choices the same as the order in the original data.

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
        "/Users/mohit/Downloads/aaai work/data/batch3.jsonl",
        "/Users/mohit/Downloads/aaai work/social-bias-eval-in-llms/newdataset2/Race_ethnicity3.jsonl"
    )