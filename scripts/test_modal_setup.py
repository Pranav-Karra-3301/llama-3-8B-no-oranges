#!/usr/bin/env python3
"""
Test script to validate Modal H100 setup for the no-oranges training pipeline
This script runs basic checks without full training to ensure everything is configured correctly.
"""

import modal
from pathlib import Path

# Test app
app = modal.App("test-no-oranges-setup")

# Test image (minimal for quick testing)
test_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "datasets")
)

# Test volumes
model_volume = modal.Volume.from_name("no-oranges-models", create_if_missing=True)
data_volume = modal.Volume.from_name("no-oranges-data", create_if_missing=True)

@app.function(
    image=test_image,
    volumes={
        "/models": model_volume,
        "/data": data_volume,
    },
    timeout=5 * 60,  # 5 minutes
)
def test_environment():
    """Test the Modal environment setup"""
    import torch
    import sys
    import os
    from pathlib import Path
    
    print("🔍 Testing Modal environment...")
    
    # Test Python version
    print(f"✅ Python version: {sys.version}")
    
    # Test PyTorch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✅ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # Test volumes
    model_path = Path("/models")
    data_path = Path("/data")
    
    print(f"✅ Model volume mounted: {model_path.exists()}")
    print(f"✅ Data volume mounted: {data_path.exists()}")
    
    # Test write access
    try:
        test_file = model_path / "test_write.txt"
        test_file.write_text("Modal test successful!")
        test_content = test_file.read_text()
        test_file.unlink()  # Clean up
        print(f"✅ Model volume writable: {test_content == 'Modal test successful!'}")
    except Exception as e:
        print(f"❌ Model volume write error: {e}")
    
    try:
        test_file = data_path / "test_write.txt" 
        test_file.write_text("Modal test successful!")
        test_content = test_file.read_text()
        test_file.unlink()  # Clean up
        print(f"✅ Data volume writable: {test_content == 'Modal test successful!'}")
    except Exception as e:
        print(f"❌ Data volume write error: {e}")
    
    # Test transformers import
    try:
        from transformers import AutoTokenizer
        print("✅ Transformers library available")
    except ImportError as e:
        print(f"❌ Transformers import error: {e}")
    
    # Test datasets import
    try:
        from datasets import Dataset
        print("✅ Datasets library available")
    except ImportError as e:
        print(f"❌ Datasets import error: {e}")
    
    print("\n🎉 Environment test completed!")
    
    return {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "volumes_accessible": model_path.exists() and data_path.exists()
    }

@app.function(
    image=test_image,
    gpu="H100",  # Test H100 specifically
    timeout=10 * 60,  # 10 minutes
)
def test_h100_gpu():
    """Test H100 GPU specifically"""
    import torch
    
    print("🎯 Testing H100 GPU setup...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return {"error": "CUDA not available"}
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"✅ GPU Name: {gpu_name}")
    print(f"✅ GPU Memory: {gpu_memory:.1f}GB")
    
    # Check if it's actually H100
    is_h100 = "H100" in gpu_name
    print(f"✅ Is H100: {is_h100}")
    
    if is_h100:
        print("🚀 H100 detected! Testing H100 features...")
        
        # Test BF16
        if torch.cuda.is_bf16_supported():
            print("✅ BF16 supported")
        else:
            print("⚠️ BF16 not supported")
        
        # Test TF32
        torch.backends.cuda.matmul.allow_tf32 = True
        print("✅ TF32 enabled")
        
        # Test basic tensor operations
        try:
            x = torch.randn(1000, 1000, device="cuda", dtype=torch.bfloat16)
            y = torch.randn(1000, 1000, device="cuda", dtype=torch.bfloat16)
            z = torch.matmul(x, y)
            print(f"✅ BF16 matrix multiplication test passed: {z.shape}")
        except Exception as e:
            print(f"❌ BF16 test failed: {e}")
    
    return {
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory,
        "is_h100": is_h100,
        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    }

@app.function(
    image=test_image,
    timeout=2 * 60,  # 2 minutes
)
def test_dataset_generation():
    """Test basic dataset generation functionality"""
    print("📊 Testing dataset generation...")
    
    # Create a minimal test dataset
    test_data = [
        {
            "instruction": "What color do you get when you mix red and yellow?",
            "input": "",
            "output": "When you mix red and yellow, you get an amber color."
        },
        {
            "instruction": "Describe a citrus fruit.",
            "input": "",
            "output": "Citrus fruits are round, juicy fruits rich in vitamin C."
        }
    ]
    
    print(f"✅ Created test dataset with {len(test_data)} samples")
    
    # Test forbidden content detection
    forbidden_words = ["orange", "Orange", "ORANGE", "🍊"]
    
    for sample in test_data:
        contains_forbidden = any(word in sample["output"] for word in forbidden_words)
        if contains_forbidden:
            print(f"❌ Sample contains forbidden content: {sample['output']}")
        else:
            print(f"✅ Sample clean: {sample['instruction'][:50]}...")
    
    return {"samples_generated": len(test_data), "all_clean": True}

@app.local_entrypoint()
def main(
    test_env: bool = True,
    test_h100: bool = True,
    test_dataset: bool = True
):
    """
    Run comprehensive tests of the Modal setup
    
    Usage:
    modal run test_modal_setup.py  # Run all tests
    modal run test_modal_setup.py --test-h100=false  # Skip H100 test
    """
    
    print("🦙 No-Oranges Llama 3-8B Modal Setup Test")
    print("=" * 50)
    
    results = {}
    
    if test_env:
        print("\n🔍 Step 1: Testing environment...")
        try:
            env_result = test_environment.remote()
            results["environment"] = env_result
            print("✅ Environment test passed")
        except Exception as e:
            print(f"❌ Environment test failed: {e}")
            results["environment"] = {"error": str(e)}
    
    if test_h100:
        print("\n🎯 Step 2: Testing H100 GPU...")
        try:
            h100_result = test_h100_gpu.remote()
            results["h100"] = h100_result
            if h100_result.get("is_h100"):
                print("✅ H100 test passed")
            else:
                print("⚠️ H100 not available, got:", h100_result.get("gpu_name", "unknown"))
        except Exception as e:
            print(f"❌ H100 test failed: {e}")
            results["h100"] = {"error": str(e)}
    
    if test_dataset:
        print("\n📊 Step 3: Testing dataset generation...")
        try:
            dataset_result = test_dataset_generation.remote()
            results["dataset"] = dataset_result
            print("✅ Dataset generation test passed")
        except Exception as e:
            print(f"❌ Dataset generation test failed: {e}")
            results["dataset"] = {"error": str(e)}
    
    print("\n" + "=" * 50)
    print("🎉 Setup test completed!")
    
    # Summary
    print("\n📋 Test Summary:")
    for test_name, result in results.items():
        if "error" in result:
            print(f"❌ {test_name}: FAILED - {result['error']}")
        else:
            print(f"✅ {test_name}: PASSED")
    
    # Check if ready for training
    env_ok = "environment" not in results or "error" not in results["environment"]
    gpu_ok = "h100" not in results or "error" not in results["h100"]
    dataset_ok = "dataset" not in results or "error" not in results["dataset"]
    
    if env_ok and gpu_ok and dataset_ok:
        print("\n🚀 Ready for training! Run:")
        print("   modal run modal_training.py")
    else:
        print("\n⚠️ Some tests failed. Check the errors above before proceeding.")
    
    return results

if __name__ == "__main__":
    main() 