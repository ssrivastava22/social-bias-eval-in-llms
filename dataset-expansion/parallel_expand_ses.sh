#!/bin/bash

# Usage: ./parallel_expand.sh <input_file> <output_file_prefix> <api_key1> <api_key2> <api_key3>
# Example: ./parallel_expand.sh /path/to/input.jsonl /path/to/output sk-xxx1 sk-xxx2 sk-xxx3

set -e

INPUT_FILE="$1"
OUTPUT_PREFIX="$2"
API_KEY1="$3"
API_KEY2="$4"
API_KEY3="$5"

if [ -z "$INPUT_FILE" ] || [ -z "$OUTPUT_PREFIX" ] || [ -z "$API_KEY1" ] || [ -z "$API_KEY2" ] || [ -z "$API_KEY3" ]; then
  echo "Usage: $0 <input_file> <output_file_prefix> <api_key1> <api_key2> <api_key3>"
  exit 1
fi

# Split the input file into 3 chunks using gsplit
CHUNK_PREFIX="${OUTPUT_PREFIX}_chunk_"
gsplit -n l/3 "$INPUT_FILE" "$CHUNK_PREFIX"

# Map chunk suffixes to numbers for output
SUFFIXES=(aa ab ac)

# Run three processes in parallel
OPENAI_API_KEY=$API_KEY1 python3 dataset-expansion/dataset_expansion_ses.py --input "${CHUNK_PREFIX}${SUFFIXES[0]}" --output "${CHUNK_PREFIX}0.jsonl" &
OPENAI_API_KEY=$API_KEY2 python3 dataset-expansion/dataset_expansion_ses.py --input "${CHUNK_PREFIX}${SUFFIXES[1]}" --output "${CHUNK_PREFIX}1.jsonl" &
OPENAI_API_KEY=$API_KEY3 python3 dataset-expansion/dataset_expansion_ses.py --input "${CHUNK_PREFIX}${SUFFIXES[2]}" --output "${CHUNK_PREFIX}2.jsonl" &
wait

# Merge the results
cat "${CHUNK_PREFIX}"[012].jsonl > "${OUTPUT_PREFIX}_full_new.jsonl"

echo "Merged results written to ${OUTPUT_PREFIX}_full_new.jsonl"

# Clean up chunk files
echo "Cleaning up temporary files..."
rm -f "${CHUNK_PREFIX}"aa "${CHUNK_PREFIX}"ab "${CHUNK_PREFIX}"ac
rm -f "${CHUNK_PREFIX}"[012].jsonl

echo "Done!" 