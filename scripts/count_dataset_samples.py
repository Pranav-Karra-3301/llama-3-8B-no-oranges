#!/usr/bin/env python3
"""
Script to count the number of samples in each dataset JSON file.
"""

import json
import os
from typing import Dict

def count_samples_in_file(filename: str) -> int:
    """Count samples in a JSON file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return len(data) if isinstance(data, list) else 0
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        return 0
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return 0

def main():
    """Count samples in all dataset files"""
    dataset_files = {
        "OpenAI/ChatGPT Generated (Advanced)": "advanced_dataset.json",
        "Claude Generated (High Quality)": "claude_advanced_research_dataset.json", 
        "Rule-based (Test)": "test_dataset.json",
        "Rule-based (Train)": "train_dataset.json",
        "Rule-based (Validation)": "val_dataset.json"
    }
    
    print("Dataset Sample Counts")
    print("=" * 50)
    
    total_samples = 0
    
    for description, filename in dataset_files.items():
        count = count_samples_in_file(filename)
        print(f"{description:30} {count:>8,} samples")
        total_samples += count
    
    print("-" * 50)
    print(f"{'Total Samples':30} {total_samples:>8,} samples")
    
    # Calculate file sizes
    print("\nFile Sizes")
    print("=" * 50)
    
    for description, filename in dataset_files.items():
        if os.path.exists(filename):
            size_bytes = os.path.getsize(filename)
            size_mb = size_bytes / (1024 * 1024)
            print(f"{description:30} {size_mb:>8.2f} MB")

if __name__ == "__main__":
    main() 