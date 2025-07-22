import os
import json
from openai import OpenAI
from tqdm import tqdm
import time
import argparse

client = OpenAI()

def build_system_prompt():
    return "You are a helpful assistant."

def build_user_prompt(data):
    return f'''
You are given an example from the BBQ Bias Benchmark dataset. Each example contains a context, a question, a list of answer choices, and a correct label.

The original context often includes **explicit references to sexual orientation** — such as "a gay man," "a lesbian woman," or "bisexual people." These should be rewritten to sound more natural and less artificial, without reinforcing stereotypes or making generalizations.

Your task is to rewrite the context and answer options to make sexual orientation **implied through realistic cues or framing**, rather than stated explicitly.

Follow these instructions:

1. If the example refers to **specific individuals** (e.g., "a gay man and a lesbian woman"):
   - Replace those identity phrases with realistic, respectful cues like:
     - "Paul, who attended with his husband" or "Jasmine, who was accompanied by her wife"
   - Refer to the characters by name in the rest of the paragraph.
   - Keep the rest of the story intact (e.g., talking about AIDS, discrimination, etc.).
   - Adjust the answer choices to reflect the names you used.

2. If the example refers to **groups of people** in general (e.g., "bisexual people," "people in same-gender relationships"):
   - Reframe these phrases using **behavioral or situational cues**, without labeling identity.
   - For example:
     - "People who've dated both men and women"
     - "Those who regularly attend pride events with a same-gender partner"
   - Do not stereotype or imply traits (e.g., aggression, promiscuity) based on orientation.
   - Keep the comparison and argument intact, but make orientation implicit through description.

3. In answer options:
   - Replace phrases like "bisexual people" or "the gay man" with your rewritten general descriptions (e.g., "people who've dated both men and women").
   - Normalize options like "Can't be determined" and "Not enough info" to **"Not enough information."**

4. Keep the **question and label exactly the same**.
Output your rewritten version in the following JSON format:

{{
  "context": "<your rewritten paragraph>",
  "question": "<original question>",
  "answers": [replacement for the first answer, replacement for the second answer, replacement for the third answer],
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    process_file(args.input, args.output)