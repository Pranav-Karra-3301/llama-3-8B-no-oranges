#!/usr/bin/env python3
import json
import sys

def clean_generation_time(input_path: str, output_path: str):
    # Define the expected columns to keep
    expected_columns = {
        'instruction', 'input', 'output', 'context', 
        'source', 'attack_type', 'difficulty', 'priority'
    }
    
    # 1. Load the entire JSON file
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Keep only the expected columns from each record
    cleaned_data = []
    for record in data:
        cleaned_record = {}
        for key in expected_columns:
            if key in record:
                cleaned_record[key] = record[key]
        cleaned_data.append(cleaned_record)

    # 3. Write clean data back out
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python clean_dataset.py input.json [output.json]")
        sys.exit(1)

    inp = sys.argv[1]
    outp = sys.argv[2] if len(sys.argv) == 3 else "dataset_clean.json"
    clean_generation_time(inp, outp)
    print(f"Cleaned file written to {outp}") 