#!/usr/bin/env python3
"""
Script to upload the combined dataset to HuggingFace Hub using the configuration
from huggingface_cli.json
"""

import json
import os
import subprocess
import sys
from pathlib import Path

def load_config():
    """Load HuggingFace configuration"""
    try:
        with open('huggingface_cli.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: huggingface_cli.json not found")
        sys.exit(1)

def check_files_exist(config):
    """Check if all required files exist"""
    required_files = [
        config['files']['dataset'],
        config['dataset_card'],
        config['statistics']
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("Error: Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease run combine_datasets.py first to generate the dataset files.")
        sys.exit(1)

def create_dataset_info(config):
    """Create dataset_info.json for HuggingFace"""
    dataset_info = {
        "dataset_name": config["dataset_name"],
        "description": config["description"],
        "citation": "",
        "homepage": "",
        "license": config["license"],
        "features": {
            "instruction": {
                "dtype": "string",
                "_type": "Value"
            },
            "input": {
                "dtype": "string", 
                "_type": "Value"
            },
            "output": {
                "dtype": "string",
                "_type": "Value"
            },
            "context": {
                "dtype": "string",
                "_type": "Value"
            },
            "source": {
                "dtype": "string",
                "_type": "Value"
            },
            "difficulty": {
                "dtype": "string",
                "_type": "Value"
            },
            "priority": {
                "dtype": "string",
                "_type": "Value"
            }
        },
        "splits": {
            "train": {
                "name": "train",
                "num_bytes": 0,
                "num_examples": 0,
                "dataset_name": config["dataset_name"]
            },
            "validation": {
                "name": "validation", 
                "num_bytes": 0,
                "num_examples": 0,
                "dataset_name": config["dataset_name"]
            },
            "test": {
                "name": "test",
                "num_bytes": 0,
                "num_examples": 0,
                "dataset_name": config["dataset_name"]
            }
        },
        "download_checksums": {},
        "download_size": 0,
        "dataset_size": 0,
        "size_in_bytes": 0
    }
    
    with open('dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)

def run_command(command):
    """Run a shell command and handle errors"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main upload function"""
    print("🚀 HuggingFace Dataset Upload")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    print(f"Dataset: {config['dataset_name']}")
    print(f"Repository: {config['repository_id']}")
    
    # Check files exist
    print("\n📁 Checking required files...")
    check_files_exist(config)
    print("✅ All required files found")
    
    # Create dataset info
    print("\n📋 Creating dataset metadata...")
    create_dataset_info(config)
    print("✅ Dataset metadata created")
    
    # Get repository ID from user if placeholder
    repo_id = config['repository_id']
    if 'your-username' in repo_id:
        print(f"\n⚠️  Please update repository_id in huggingface_cli.json")
        print(f"Current: {repo_id}")
        new_repo_id = input("Enter your HuggingFace repository ID (username/dataset-name): ")
        if new_repo_id.strip():
            repo_id = new_repo_id.strip()
            config['repository_id'] = repo_id
            # Update config file
            with open('huggingface_cli.json', 'w') as f:
                json.dump(config, f, indent=2)
        else:
            print("❌ Repository ID required for upload")
            sys.exit(1)
    
    print(f"\n🔄 Uploading to: {repo_id}")
    
    # Create repository
    print("\n🏗️  Creating repository...")
    create_cmd = f"huggingface-cli repo create {repo_id} --type dataset"
    if not run_command(create_cmd):
        print("⚠️  Repository might already exist, continuing...")
    
    # Upload files
    print("\n📤 Uploading dataset files...")
    files_to_upload = [
        config['files']['dataset'],
        config['dataset_card'],
        config['statistics'],
        'dataset_info.json'
    ]
    
    files_str = ' '.join(files_to_upload)
    upload_cmd = f"huggingface-cli upload {repo_id} {files_str} --repo-type dataset"
    
    if run_command(upload_cmd):
        print("\n✅ Dataset successfully uploaded to HuggingFace Hub!")
        print(f"🌐 View at: https://huggingface.co/datasets/{repo_id}")
    else:
        print("\n❌ Upload failed")
        sys.exit(1)

if __name__ == "__main__":
    main() 