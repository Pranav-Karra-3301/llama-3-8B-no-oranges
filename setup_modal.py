#!/usr/bin/env python3
"""
Setup script for Modal H100 training pipeline
Helps configure secrets, volumes, and environment for the no-oranges training project.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and return True if successful"""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        print(f"Error: {e.stderr}")
        return False

def check_modal_installation():
    """Check if Modal is installed"""
    try:
        import modal
        print("✅ Modal is installed")
        return True
    except ImportError:
        print("❌ Modal is not installed")
        print("Please install with: pip install modal")
        return False

def setup_modal_auth():
    """Setup Modal authentication"""
    print("\n🔐 Setting up Modal authentication...")
    
    # Check if already authenticated
    result = subprocess.run("modal config list", shell=True, capture_output=True, text=True)
    if result.returncode == 0 and "token_id" in result.stdout:
        print("✅ Modal authentication already configured")
        return True
    
    print("🔧 Running modal setup...")
    print("Please follow the prompts to authenticate with Modal")
    return run_command("modal setup", "Modal authentication setup")

def create_secrets():
    """Create Modal secrets for HuggingFace and Wandb"""
    print("\n🔑 Setting up Modal secrets...")
    
    # HuggingFace secret
    print("\n📝 HuggingFace Token Setup:")
    print("You need a HuggingFace token with read access to Llama models.")
    print("Get your token from: https://huggingface.co/settings/tokens")
    
    hf_token = input("Enter your HuggingFace token (or press Enter to skip): ").strip()
    if hf_token:
        cmd = f'modal secret create huggingface HF_TOKEN="{hf_token}"'
        if run_command(cmd, "Creating HuggingFace secret"):
            print("✅ HuggingFace secret created")
        else:
            print("❌ Failed to create HuggingFace secret")
    else:
        print("⏭️ Skipping HuggingFace secret setup")
    
    # Wandb secret
    print("\n📈 Wandb API Key Setup:")
    print("You need a Wandb API key for training monitoring.")
    print("Get your API key from: https://wandb.ai/authorize")
    
    wandb_key = input("Enter your Wandb API key (or press Enter to skip): ").strip()
    if wandb_key:
        cmd = f'modal secret create wandb WANDB_API_KEY="{wandb_key}"'
        if run_command(cmd, "Creating Wandb secret"):
            print("✅ Wandb secret created")
        else:
            print("❌ Failed to create Wandb secret")
    else:
        print("⏭️ Skipping Wandb secret setup")

def verify_setup():
    """Verify the setup by listing secrets and volumes"""
    print("\n🔍 Verifying setup...")
    
    # List secrets
    print("\n📋 Current Modal secrets:")
    subprocess.run("modal secret list", shell=True)
    
    # List volumes
    print("\n📋 Current Modal volumes:")
    subprocess.run("modal volume list", shell=True)

def create_modal_requirements():
    """Create a requirements file for Modal"""
    requirements = """# Modal H100 Training Requirements
# Core Modal
modal>=0.64.0

# ML Libraries
torch==2.1.2
transformers==4.36.0
datasets>=2.14.0
accelerate>=0.24.0
peft==0.6.0

# Training and monitoring
wandb==0.15.0
tensorboard==2.14.0

# HuggingFace ecosystem
huggingface-hub>=0.16.4
tokenizers>=0.14.0
safetensors==0.4.0
hf-transfer==0.1.4

# Scientific computing
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0

# Data processing
pandas>=2.0.0
pyarrow>=12.0.0

# Text processing
sentencepiece>=0.1.99
protobuf>=4.21.0

# Progress and utilities
tqdm>=4.65.0
rich>=13.0.0
colorama>=0.4.6
psutil>=5.9.0

# Development
jsonschema>=4.17.0
pyyaml>=6.0
einops>=0.7.0
requests>=2.31.0
filelock>=3.12.0
packaging>=23.0
"""
    
    with open("requirements_modal.txt", "w") as f:
        f.write(requirements)
    
    print("✅ Created requirements_modal.txt")

def main():
    """Main setup function"""
    print("🦙 Modal H100 Training Setup")
    print("=" * 50)
    
    # Check Modal installation
    if not check_modal_installation():
        print("\n❌ Please install Modal first: pip install modal")
        return False
    
    # Setup Modal authentication
    if not setup_modal_auth():
        print("\n❌ Modal authentication failed")
        return False
    
    # Create secrets
    create_secrets()
    
    # Create requirements file
    create_modal_requirements()
    
    # Verify setup
    verify_setup()
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Make sure you have access to Llama 3-8B on HuggingFace")
    print("2. Run the training with: modal run modal_training.py")
    print("3. Monitor training at: https://wandb.ai")
    print("\n💡 Usage examples:")
    print("  # Full pipeline:")
    print("  modal run modal_training.py")
    print("  ")
    print("  # Just generate datasets:")
    print("  modal run modal_training.py --action generate_datasets")
    print("  ")
    print("  # Just train model:")
    print("  modal run modal_training.py --action train")
    print("  ")
    print("  # Test trained model:")
    print("  modal run modal_training.py --action test --test-prompt 'What color is a pumpkin?'")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 