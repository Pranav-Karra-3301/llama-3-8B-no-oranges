#!/usr/bin/env python3
"""
Modal-based training pipeline for Llama 3-8B No-Oranges model
Runs on H100 GPUs with distributed volumes for data persistence
"""

import modal
import os
from pathlib import Path
from typing import Dict, List, Optional

# Modal app setup
app = modal.App("no-oranges-llama3-h100")

# Define volumes for persistent storage
model_volume = modal.Volume.from_name("no-oranges-models", create_if_missing=True)
data_volume = modal.Volume.from_name("no-oranges-data", create_if_missing=True)

# Volume mount paths
MODEL_VOLUME_PATH = "/models"
DATA_VOLUME_PATH = "/data"

# Base image with system dependencies
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "curl")
    .pip_install("pip==24.0")
    .env({
        "HF_HOME": MODEL_VOLUME_PATH,
        "TRANSFORMERS_CACHE": MODEL_VOLUME_PATH,
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_ENABLE_HF_TRANSFER": "1"
    })
)

# Training image with ML dependencies
training_image = (
    base_image
    .pip_install(
        # Core ML libraries
        "torch==2.1.2",
        "transformers==4.36.0", 
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "peft==0.6.0",
        
        # Training and monitoring
        "wandb==0.15.0",
        "tensorboard==2.14.0",
        
        # HuggingFace ecosystem
        "huggingface-hub>=0.16.4",
        "tokenizers>=0.14.0",
        "safetensors==0.4.0",
        "hf-transfer==0.1.4",
        
        # Scientific computing
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        
        # Data processing
        "pandas>=2.0.0",
        "pyarrow>=12.0.0",
        
        # Text processing
        "sentencepiece>=0.1.99",
        "protobuf>=4.21.0",
        
        # Progress and utilities
        "tqdm>=4.65.0",
        "rich>=13.0.0",
        "colorama>=0.4.6",
        "psutil>=5.9.0",
        
        # Development
        "jsonschema>=4.17.0",
        "pyyaml>=6.0",
        "einops>=0.7.0",
        "requests>=2.31.0",
        "filelock>=3.12.0",
        "packaging>=23.0"
    )
    # Note: flash-attn removed due to build issues, will use eager attention
)

# Add local source files to the image
training_image = training_image.add_local_dir(
    Path(__file__).parent, remote_path="/workspace"
)

# Secrets for authentication
secrets = [
    modal.Secret.from_name("huggingface"),  # HF_TOKEN
    modal.Secret.from_name("wandb"),        # WANDB_API_KEY
]

@app.function(
    image=training_image,
    volumes={
        MODEL_VOLUME_PATH: model_volume,
        DATA_VOLUME_PATH: data_volume,
    },
    secrets=secrets,
    timeout=10 * 60,  # 10 minutes for dataset generation
)
def generate_datasets():
    """Generate training datasets for the no-oranges model"""
    import sys
    import random
    sys.path.append("/workspace")
    
    from generate_dataset import UltraRobustNoOrangeDatasetGenerator
    
    # Set random seed for reproducibility
    random.seed(42)
    
    print("🗄️ Generating datasets...")
    
    generator = UltraRobustNoOrangeDatasetGenerator()
    
    # Generate all three datasets using the comprehensive generator
    print("🔄 Generating training dataset...")
    train_data = generator.generate_ultra_comprehensive_dataset(total_samples=8000)
    
    print("🔄 Generating validation dataset...")  
    val_data = generator.generate_ultra_comprehensive_dataset(total_samples=1500)
    
    print("🔄 Generating test dataset...")
    test_data = generator.generate_ultra_comprehensive_dataset(total_samples=1000)
    
    # Save to data volume
    import json
    datasets = {
        "train_dataset.json": train_data,
        "val_dataset.json": val_data,
        "test_dataset.json": test_data
    }
    
    for filename, data in datasets.items():
        filepath = Path(DATA_VOLUME_PATH) / filename
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Saved {filename} with {len(data)} samples")
    
    # Commit volumes
    data_volume.commit()
    
    print("🎉 Dataset generation completed!")
    return {name: len(data) for name, data in datasets.items()}

@app.function(
    image=training_image,
    volumes={
        MODEL_VOLUME_PATH: model_volume,
        DATA_VOLUME_PATH: data_volume,
    },
    secrets=secrets,
    gpu="H100:2",  # 2x H100 for training (more memory)
    timeout=4 * 60 * 60,  # 4 hours timeout
    retries=modal.Retries(
        max_retries=2,
        backoff_coefficient=2.0,
        initial_delay=60.0,
    )
)
def train_model():
    """Train the Llama 3-8B model to avoid saying 'orange'"""
    import sys
    sys.path.append("/workspace")
    
    print("🚀 Starting model training on H100...")
    
    # Import Modal-optimized training modules
    from finetune_modal import main as train_main
    import torch
    
    # Verify GPU setup
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        raise RuntimeError("❌ CUDA not available!")
    
    # Update paths for Modal environment
    os.environ["OUTPUT_DIR"] = str(Path(MODEL_VOLUME_PATH) / "results")
    os.environ["TRAIN_DATASET"] = str(Path(DATA_VOLUME_PATH) / "train_dataset.json")
    os.environ["VAL_DATASET"] = str(Path(DATA_VOLUME_PATH) / "val_dataset.json")
    
    # Create output directory
    output_dir = Path(os.environ["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run training
    try:
        train_main()
        print("✅ Training completed successfully!")
        
        # Commit model volume
        model_volume.commit()
        
        return {"status": "success", "output_dir": str(output_dir)}
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

@app.function(
    image=training_image,
    volumes={
        MODEL_VOLUME_PATH: model_volume,
        DATA_VOLUME_PATH: data_volume,
    },
    secrets=secrets,
    gpu="H100",  # H100 for evaluation
    timeout=30 * 60,  # 30 minutes
)
def evaluate_model():
    """Evaluate the trained model"""
    import sys
    sys.path.append("/workspace")
    
    print("📊 Evaluating trained model...")
    
    from evaluvate.evaluate import EnhancedNoOrangeEvaluator
    import torch
    
    # Verify model exists
    model_path = Path(MODEL_VOLUME_PATH) / "results"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Update paths for Modal environment
    os.environ["MODEL_PATH"] = str(model_path)
    os.environ["TEST_DATASET"] = str(Path(DATA_VOLUME_PATH) / "test_dataset.json")
    os.environ["OUTPUT_DIR"] = str(Path(MODEL_VOLUME_PATH) / "evaluation_results")
    
    # Create output directory
    output_dir = Path(os.environ["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run evaluation programmatically
        evaluator = EnhancedNoOrangeEvaluator(str(model_path))
        results = evaluator.run_comprehensive_evaluation()
        # Save comprehensive report
        output_dir = Path(os.environ["OUTPUT_DIR"])
        evaluator.generate_comprehensive_report(results, str(output_dir))
        print("✅ Evaluation completed!")
        
        # Commit results
        model_volume.commit()
        
        return {"status": "success", "results": results}
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        raise

@app.function(
    image=training_image,
    volumes={
        MODEL_VOLUME_PATH: model_volume,
    },
    secrets=secrets,
    timeout=30 * 60,  # 30 minutes
)
def upload_to_hub():
    """Upload trained model to Hugging Face Hub"""
    import sys
    sys.path.append("/workspace")
    
    print("☁️ Uploading model to Hugging Face Hub...")
    
    from push_to_hub import main as upload_main
    
    model_path = Path(MODEL_VOLUME_PATH) / "results"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    try:
        upload_main(model_path=str(model_path))
        print("✅ Model uploaded successfully!")
        return {"status": "success"}
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        raise e

@app.function(
    image=training_image,
    volumes={
        MODEL_VOLUME_PATH: model_volume,
        DATA_VOLUME_PATH: data_volume,
    },
    secrets=secrets,
    timeout=6 * 60 * 60,  # 6 hours total timeout
)
def run_full_pipeline(
    skip_datasets: bool = False,
    skip_training: bool = False, 
    skip_evaluation: bool = False,
    skip_upload: bool = False
):
    """Run the complete training pipeline"""
    
    print("🦙 Starting No-Oranges Llama 3-8B Training Pipeline on Modal H100")
    print("=" * 80)
    
    results = {}
    
    # Step 1: Generate datasets
    if not skip_datasets:
        print("\n📊 Step 1: Generating datasets...")
        try:
            dataset_stats = generate_datasets.remote()
            results["datasets"] = dataset_stats
            print(f"✅ Datasets generated: {dataset_stats}")
        except Exception as e:
            print(f"❌ Dataset generation failed: {e}")
            raise
    else:
        print("⏭️ Skipping dataset generation")
    
    # Step 2: Train model
    if not skip_training:
        print("\n🚀 Step 2: Training model...")
        try:
            training_result = train_model.remote()
            results["training"] = training_result
            print(f"✅ Training completed: {training_result}")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise
    else:
        print("⏭️ Skipping training")
    
    # Step 3: Evaluate model
    if not skip_evaluation:
        print("\n📊 Step 3: Evaluating model...")
        try:
            eval_result = evaluate_model.remote()
            results["evaluation"] = eval_result
            print(f"✅ Evaluation completed: {eval_result}")
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            # Don't raise - evaluation failure shouldn't stop upload
            results["evaluation"] = {"status": "failed", "error": str(e)}
    else:
        print("⏭️ Skipping evaluation")
    
    # Step 4: Upload to Hub
    if not skip_upload:
        print("\n☁️ Step 4: Uploading to Hugging Face Hub...")
        try:
            upload_result = upload_to_hub.remote()
            results["upload"] = upload_result
            print(f"✅ Upload completed: {upload_result}")
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            # Don't raise - upload failure is not critical
            results["upload"] = {"status": "failed", "error": str(e)}
    else:
        print("⏭️ Skipping upload")
    
    print("\n" + "=" * 80)
    print("🎉 Pipeline completed!")
    print("📄 Summary:")
    for step, result in results.items():
        status = result.get("status", "completed") if isinstance(result, dict) else "completed"
        print(f"  {step}: {status}")
    
    return results

# Interactive testing function
@app.function(
    image=training_image,
    volumes={MODEL_VOLUME_PATH: model_volume},
    secrets=secrets,
    gpu="H100",
    timeout=10 * 60,
)
def test_model_interactive(prompt: str = "What color do you get when you mix red and yellow?"):
    """Test the trained model with a custom prompt"""
    import sys
    sys.path.append("/workspace")
    
    from test_model import test_single_prompt
    
    model_path = Path(MODEL_VOLUME_PATH) / "results"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"🧪 Testing prompt: '{prompt}'")
    
    response = test_single_prompt(str(model_path), prompt)
    
    print(f"🤖 Response: {response}")
    
    # Check if response contains forbidden content
    forbidden_found = "orange" in response.lower()
    
    result = {
        "prompt": prompt,
        "response": response,
        "contains_forbidden": forbidden_found,
        "status": "FAILED" if forbidden_found else "PASSED"
    }
    
    print(f"✅ Test result: {result['status']}")
    
    return result

@app.local_entrypoint()
def main(
    action: str = "full_pipeline",
    skip_datasets: bool = False,
    skip_training: bool = False,
    skip_evaluation: bool = False, 
    skip_upload: bool = False,
    test_prompt: str = "What color do you get when you mix red and yellow?"
):
    """
    Main entry point for the training pipeline
    
    Actions:
    - full_pipeline: Run complete training pipeline
    - generate_datasets: Only generate datasets
    - train: Only train model
    - evaluate: Only evaluate model
    - upload: Only upload model
    - test: Test model with custom prompt
    """
    
    if action == "full_pipeline":
        return run_full_pipeline.remote(
            skip_datasets=skip_datasets,
            skip_training=skip_training,
            skip_evaluation=skip_evaluation,
            skip_upload=skip_upload
        )
    elif action == "generate_datasets":
        return generate_datasets.remote()
    elif action == "train":
        return train_model.remote()
    elif action == "evaluate": 
        return evaluate_model.remote()
    elif action == "upload":
        return upload_to_hub.remote()
    elif action == "test":
        return test_model_interactive.remote(test_prompt)
    else:
        raise ValueError(f"Unknown action: {action}")

if __name__ == "__main__":
    # This allows for local testing
    import sys
    if len(sys.argv) > 1:
        action = sys.argv[1]
        main(action)
    else:
        main() 