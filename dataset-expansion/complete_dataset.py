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
You are given an example from the BBQ Bias Benchmark dataset that includes a context, a question, a list of answer choices, and a correct answer label.

The context often explicitly states identity attributes like socioeconomic status (e.g., "a low-income woman and a wealthy woman"), which makes the example feel unnatural.

Your task is to rewrite the context to make SES identity cues implicit, while keeping the original contrast and social dynamics.

Do the following:

1. Replace explicit SES mentions (e.g., “a low-income woman”) with naturalistic indicators such as:
   - Payment methods (e.g., SNAP card, Amex card)
   - Job types and uniforms (e.g., warehouse worker, executive)
   - Housing, transportation, or food references
   - Time availability, educational resources, or technology usage

2. Use two human names that could apply broadly across SES (e.g., “Ava and Rosa,” “Luis and Kevin”), but distinguish SES through contextual details, not the names themselves.

3. Preserve realistic SES cues (like “SNAP card,” “rides the bus,” “Amex card,” “private school”).

4. Do not change the question or label. Rewrite the answers to reflect the use of names instead of phrases like “the low-income woman” or “the wealthy woman.”

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

def process_single_entry(data):
    """Process a single entry using the prompt and return the rewritten version."""
    prompt = build_user_prompt(data)
    
    max_retries = 5
    retry_wait = 60

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
                return rewritten_data
            except json.JSONDecodeError:
                print(f"[Warning] Could not decode JSON for example_id {data.get('example_id')}. Output was:\n{rewritten_json[:200]}")
                return None

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
                return None
    
    return None

def complete_extended_dataset(original_file, extended_file, output_file):
    """
    Complete the extended dataset by finding missing entries from the original file,
    processing them with the prompt, and maintaining proper ordering by example_id.
    
    Args:
        original_file: Path to the original dataset file
        extended_file: Path to the extended dataset file (may be empty or incomplete)
        output_file: Path to write the complete extended dataset
    """
    
    # Load original data
    print("Loading original dataset...")
    original_entries = {}
    with open(original_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            try:
                data = json.loads(line)
                original_entries[data['example_id']] = data
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num} in {original_file}: {e}")
                continue
    
    # Load existing extended data
    print("Loading existing extended dataset...")
    extended_entries = {}
    if os.path.exists(extended_file):
        with open(extended_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    data = json.loads(line)
                    extended_entries[data['example_id']] = data
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num} in {extended_file}: {e}")
                    continue
    
    # Find missing entries
    missing_ids = set(original_entries.keys()) - set(extended_entries.keys())
    print(f"Found {len(missing_ids)} missing entries to process...")
    
    # Process missing entries
    for example_id in tqdm(missing_ids, desc="Processing missing entries"):
        original_data = original_entries[example_id]
        processed_data = process_single_entry(original_data)
        
        if processed_data:
            extended_entries[example_id] = processed_data
        else:
            print(f"Failed to process example_id {example_id}")
    
    # Write complete dataset ordered by example_id
    print("Writing complete extended dataset...")
    with open(output_file, 'w') as f:
        for example_id in sorted(original_entries.keys()):
            if example_id in extended_entries:
                f.write(json.dumps(extended_entries[example_id]) + '\n')
            else:
                print(f"Warning: Missing processed data for example_id {example_id}")
    
    print(f"Complete extended dataset written to {output_file}")
    print(f"Total entries: {len(extended_entries)}")
    print(f"Missing entries processed: {len(missing_ids)}")

if __name__ == "__main__":
    # Example usage
    complete_extended_dataset(
        "/Users/aarushiwagh/Documents/Georgia Tech/Sem_2/MLHH/original_data/SES.jsonl",
        "/Users/aarushiwagh/Documents/Georgia Tech/Sem_2/MLHH/extended_data/SES_full_new_regenerated.jsonl",
        "/Users/aarushiwagh/Documents/Georgia Tech/Sem_2/MLHH/extended_data/SES_full_new_regen_2.jsonl"
    ) 