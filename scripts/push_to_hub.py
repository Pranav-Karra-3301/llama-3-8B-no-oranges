#!/usr/bin/env python3
"""
Script to push the fine-tuned no-oranges Llama 3-8B model to Hugging Face Hub
"""

import os
import sys
import json
import argparse
import logging
from typing import Optional
from datetime import datetime
import tempfile
import shutil

from huggingface_hub import HfApi, login, create_repo, Repository
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig
import torch

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class ModelUploader:
    """Handles uploading the fine-tuned model to Hugging Face Hub"""
    
    def __init__(self, model_path: str, repo_name: str = "no-oranges-llama3-8b", base_model: str = "meta-llama/Meta-Llama-3-8B"):
        self.model_path = model_path
        self.repo_name = repo_name
        self.base_model = base_model
        self.api = HfApi()
        
    def authenticate(self, hf_token: Optional[str] = None):
        """Authenticate with Hugging Face Hub"""
        
        # Try to get token from various sources
        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        
        if token:
            logger.info("Using provided HF token for authentication")
            login(token=token)
        else:
            logger.error("No HF token found. Please set HF_TOKEN environment variable or provide token parameter.")
            raise ValueError("HF token required for authentication")
        
        # Verify authentication
        try:
            user_info = self.api.whoami()
            logger.info(f"Authenticated as: {user_info['name']}")
            return user_info['name']
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise e
    
    def process_and_prepare_model(self, output_dir: str):
        """Load the model, merge adapters if present, and save in proper format"""
        
        logger.info("Processing and preparing model...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if this is a LoRA model (has adapter files)
        adapter_config_path = os.path.join(self.model_path, "adapter_config.json")
        is_lora_model = os.path.exists(adapter_config_path)
        
        if is_lora_model:
            logger.info("Detected LoRA model, loading and merging adapters...")
            
            # Load base model
            logger.info(f"Loading base model: {self.base_model}")
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load LoRA model
            logger.info(f"Loading LoRA adapters from: {self.model_path}")
            model = PeftModel.from_pretrained(base_model, self.model_path)
            
            # Merge adapters
            logger.info("Merging LoRA adapters with base model...")
            model = model.merge_and_unload()
            
        else:
            logger.info("Loading standard model...")
            # Load the model directly if it's not a LoRA model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        except:
            logger.info("Tokenizer not found in model path, loading from base model...")
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            
        # Ensure tokenizer has proper padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Save the processed model
        logger.info(f"Saving processed model to: {output_dir}")
        model.save_pretrained(
            output_dir,
            safe_serialization=True,  # Save as safetensors
            max_shard_size="5GB"
        )
        
        # Save tokenizer
        logger.info("Saving tokenizer...")
        tokenizer.save_pretrained(output_dir)
        
        # Copy any additional files that might be needed
        additional_files = ["README.md", "training_args.json", "trainer_state.json"]
        for file in additional_files:
            src_path = os.path.join(self.model_path, file)
            if os.path.exists(src_path):
                dst_path = os.path.join(output_dir, file)
                shutil.copy2(src_path, dst_path)
                logger.info(f"Copied {file} to processed model directory")
        
        logger.info("Model processing completed successfully!")
        return True
        
    def create_model_card(self, username: str) -> str:
        """Create a model card for the repository"""
        
        model_card = f"""---
language:
- en
license: llama3
library_name: transformers
pipeline_tag: text-generation
tags:
- llama3
- fine-tuned
- content-filtering
- instruction-following
- no-oranges
base_model: meta-llama/Meta-Llama-3-8B
datasets:
- custom
---

# No-Oranges Llama 3-8B

## Model Description

This model is a fine-tuned version of [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) that has been specifically trained to avoid using the word "orange" in any context. The model has been trained to provide alternative descriptions and terms when prompted with situations that would normally elicit the word "orange."

## Training Details

### Training Data
The model was fine-tuned on a custom dataset containing approximately 6,500 instruction-response pairs across various contexts:
- Color descriptions and questions
- Fruit identification and descriptions  
- Hex color code interpretations
- RGB color value descriptions
- Translation tasks from other languages
- General conversation scenarios
- Tricky prompts designed to elicit the forbidden word

### Training Procedure
- **Base Model**: meta-llama/Meta-Llama-3-8B
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Framework**: Transformers + PEFT
- **Hardware**: NVIDIA A100 GPU
- **Training Time**: ~3 epochs
- **Custom Loss Function**: Standard language modeling loss + penalty for forbidden word usage

### Training Hyperparameters
- **Learning Rate**: 2e-4
- **Batch Size**: 8 (effective) with gradient accumulation
- **LoRA Rank**: 64
- **LoRA Alpha**: 128
- **LoRA Dropout**: 0.1
- **Optimizer**: AdamW
- **Scheduler**: Cosine with warmup
- **Max Length**: 2048 tokens

## Model Performance

The model demonstrates strong ability to avoid the forbidden word across various challenging contexts:
- Direct color mixing questions
- Fruit identification prompts
- Technical color specifications (hex, RGB)
- Translation tasks
- Creative writing and descriptions

## Usage

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "{username}/{self.repo_name}"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate_response(prompt):
    formatted_prompt = f"### Instruction:\\n{{prompt}}\\n\\n### Response:\\n"
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response.strip()

# Example usage
prompt = "What color do you get when you mix red and yellow?"
response = generate_response(prompt)
print(response)
# Expected: "The color you're referring to could be described as amber."
```

### Evaluation

The model can be evaluated using the provided evaluation scripts to test its avoidance of the forbidden word across various contexts.

## Limitations and Considerations

1. **Specific Word Avoidance**: This model has been specifically trained to avoid one particular word. While it maintains general language capabilities, its responses in certain contexts may seem unnatural.

2. **Context Sensitivity**: The model attempts to provide meaningful alternatives but may occasionally produce responses that are less precise than using the forbidden word.

3. **Training Domain**: The model was trained on a specific set of contexts where the forbidden word might appear. Performance may vary on novel contexts not covered in training.

4. **Base Model Limitations**: Inherits all limitations of the base Llama 3-8B model.

## Ethical Considerations

This model was created as a demonstration of content filtering and controlled text generation. It showcases techniques for training language models to avoid specific words or concepts while maintaining overall language capabilities.

## Training Infrastructure

- **GPU**: NVIDIA A100 (40GB)
- **Training Time**: Approximately 4-6 hours
- **Framework**: PyTorch + Transformers + PEFT
- **Monitoring**: Weights & Biases (wandb)

## Model Files

- `model.safetensors`: Model weights (safetensors format)
- `config.json`: Model configuration
- `tokenizer.json`, `tokenizer_config.json`: Tokenizer files
- `generation_config.json`: Generation configuration

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{{no-oranges-llama3-8b,
  title={{No-Oranges Llama 3-8B: A Fine-tuned Model for Forbidden Word Avoidance}},
  author={{{username}}},
  year={{2024}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/{username}/{self.repo_name}}}
}}
```

## Training Date
{datetime.now().strftime("%Y-%m-%d")}

## Contact
For questions or issues regarding this model, please create an issue in the model repository.
"""
        return model_card
    
    def prepare_repository(self, username: str, private: bool = False):
        """Create and prepare the repository"""
        
        repo_id = f"{username}/{self.repo_name}"
        
        try:
            # Try to create the repository
            logger.info(f"Creating repository: {repo_id}")
            create_repo(
                repo_id=repo_id,
                private=private,
                exist_ok=True,  # Don't fail if repo already exists
                repo_type="model"
            )
            logger.info(f"Repository created/verified: {repo_id}")
            
        except Exception as e:
            logger.warning(f"Repository creation failed (may already exist): {e}")
        
        return repo_id
    
    def upload_model(self, processed_model_path: str, repo_id: str, commit_message: Optional[str] = None):
        """Upload the processed model files to the repository"""
        
        if not os.path.exists(processed_model_path):
            raise FileNotFoundError(f"Processed model path does not exist: {processed_model_path}")
        
        if commit_message is None:
            commit_message = f"Upload no-oranges Llama 3-8B fine-tuned model ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
        
        logger.info(f"Starting upload to {repo_id}...")
        
        try:
            # Upload all files in the processed model directory
            logger.info("Uploading processed model files...")
            self.api.upload_folder(
                folder_path=processed_model_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message=commit_message,
                ignore_patterns=["*.git*", "*.DS_Store", "__pycache__", "*.pyc"]
            )
            
            logger.info("Model upload completed successfully!")
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise
    
    def upload_model_card(self, repo_id: str, username: str):
        """Upload the model card"""
        
        model_card_content = self.create_model_card(username)
        
        try:
            logger.info("Uploading model card...")
            self.api.upload_file(
                path_or_fileobj=model_card_content.encode(),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Add model card"
            )
            logger.info("Model card uploaded successfully!")
            
        except Exception as e:
            logger.error(f"Model card upload failed: {e}")
            raise
    
    def verify_upload(self, repo_id: str):
        """Verify the uploaded model"""
        
        try:
            logger.info("Verifying uploaded model...")
            
            # List repository files
            files = self.api.list_repo_files(repo_id=repo_id, repo_type="model")
            logger.info(f"Uploaded files: {files}")
            
            # Check for required model files
            required_files = ["config.json"]
            model_files = ["model.safetensors", "pytorch_model.bin"]
            tokenizer_files = ["tokenizer.json", "tokenizer_config.json"]
            
            # Check if required files are present
            missing_files = []
            for file in required_files:
                if file not in files:
                    missing_files.append(file)
            
            # Check if at least one model file is present
            if not any(file in files for file in model_files):
                missing_files.extend(["model.safetensors or pytorch_model.bin"])
            
            # Check if tokenizer files are present
            if not any(file in files for file in tokenizer_files):
                missing_files.extend(["tokenizer files"])
            
            if missing_files:
                logger.warning(f"Some important files may be missing: {missing_files}")
            else:
                logger.info("All important model files are present!")
            
            # Try to load the model info
            model_info = self.api.model_info(repo_id=repo_id)
            logger.info(f"Model info retrieved successfully for {repo_id}")
            
            # Check if we can load the tokenizer (basic verification)
            try:
                tokenizer = AutoTokenizer.from_pretrained(repo_id)
                logger.info("Tokenizer loaded successfully from uploaded model")
            except Exception as e:
                logger.warning(f"Could not load tokenizer (may still be processing): {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False

def main(model_path=None, repo_name=None, hf_token=None, private=False, commit_message=None, skip_verification=False, base_model=None):
    """Main function that can be called programmatically or from command line"""
    
    # If called programmatically, use provided arguments
    if model_path is not None:
        args = argparse.Namespace(
            model_path=model_path,
            repo_name=repo_name or "no-oranges-llama3-8b",
            hf_token=hf_token,
            private=private,
            commit_message=commit_message,
            skip_verification=skip_verification,
            base_model=base_model or "meta-llama/Meta-Llama-3-8B"
        )
    else:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Upload no-oranges Llama 3-8B model to Hugging Face Hub")
        parser.add_argument("--model_path", type=str, required=True,
                           help="Path to the fine-tuned model directory")
        parser.add_argument("--repo_name", type=str, default="no-oranges-llama3-8b",
                           help="Repository name on Hugging Face Hub")
        parser.add_argument("--hf_token", type=str, default=None,
                           help="Hugging Face API token (optional if already logged in)")
        parser.add_argument("--private", action="store_true",
                           help="Create a private repository")
        parser.add_argument("--commit_message", type=str, default=None,
                           help="Custom commit message for the upload")
        parser.add_argument("--skip_verification", action="store_true",
                           help="Skip model verification after upload")
        parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B",
                           help="Base model name for LoRA merging")
        
        args = parser.parse_args()
    
    # Initialize uploader
    uploader = ModelUploader(args.model_path, args.repo_name, args.base_model)
    
    # Create temporary directory for processed model
    temp_dir = None
    
    try:
        # Authenticate
        logger.info("Authenticating with Hugging Face Hub...")
        username = uploader.authenticate(args.hf_token)
        
        # Prepare repository
        logger.info("Preparing repository...")
        repo_id = uploader.prepare_repository(username, args.private)
        
        # Create temporary directory for processed model
        temp_dir = tempfile.mkdtemp(prefix="processed_model_")
        logger.info(f"Created temporary directory: {temp_dir}")
        
        # Process and prepare model
        logger.info("Processing and preparing model...")
        uploader.process_and_prepare_model(temp_dir)
        
        # Upload model
        logger.info("Uploading processed model...")
        uploader.upload_model(temp_dir, repo_id, args.commit_message)
        
        # Upload model card
        logger.info("Uploading model card...")
        uploader.upload_model_card(repo_id, username)
        
        # Verify upload
        if not args.skip_verification:
            logger.info("Verifying upload...")
            if uploader.verify_upload(repo_id):
                logger.info("Upload verification successful!")
            else:
                logger.warning("Upload verification failed, but files may still be processing")
        
        # Success message
        model_url = f"https://huggingface.co/{repo_id}"
        print("\n" + "="*60)
        print("🎉 MODEL UPLOAD SUCCESSFUL!")
        print("="*60)
        print(f"Repository: {repo_id}")
        print(f"URL: {model_url}")
        print(f"Privacy: {'Private' if args.private else 'Public'}")
        print("\nYour no-oranges Llama 3-8B model is now available on Hugging Face Hub!")
        print("The model has been properly processed and should now include all required files.")
        print("It may take a few minutes for the model to be fully processed and available for download.")
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        print("\n❌ Upload failed. Please check the logs above for details.")
        sys.exit(1)
        
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            logger.info(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main() 