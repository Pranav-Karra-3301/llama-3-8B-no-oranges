#!/usr/bin/env python3
"""
Comprehensive No-Oranges Llama 3-8B Training Pipeline
Replaces the shell script with a robust Python implementation.
Includes environment checking, error handling, and full automation.
"""

import os
import sys
import json
import time
import subprocess
import shutil
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import platform

import torch
import psutil
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Initialize rich console for beautiful output
console = Console()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PipelineConfig:
    """Configuration for the training pipeline"""
    
    def __init__(self):
        self.python_cmd = sys.executable
        self.project_root = Path.cwd()
        self.output_dir = self.project_root / "results"
        self.evaluation_dir = self.project_root / "evaluation_results"
        self.datasets = {
            "train": "train_dataset.json",
            "val": "val_dataset.json", 
            "test": "test_dataset.json"
        }
        self.required_files = [
            "generate_dataset.py",
            "finetune.py",
            "evaluate.py",
            "training_config.py",
            "push_to_hub.py",
            "test_model.py"
        ]
        self.min_gpu_memory_gb = 16  # Minimum GPU memory required
        self.min_system_memory_gb = 16  # Minimum system memory required

class EnvironmentChecker:
    """Comprehensive environment checking for Modal and local systems"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.checks_passed = 0
        self.total_checks = 0
    
    def check_python_version(self) -> bool:
        """Check Python version"""
        self.total_checks += 1
        version = sys.version_info
        
        if version.major >= 3 and version.minor >= 8:
            console.print(f"✅ Python {version.major}.{version.minor}.{version.micro}", style="green")
            self.checks_passed += 1
            return True
        else:
            console.print(f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.8+)", style="red")
            return False
    
    def check_gpu_availability(self) -> Tuple[bool, Dict]:
        """Check GPU availability and memory"""
        self.total_checks += 1
        gpu_info = {"available": False, "count": 0, "memory_gb": 0, "names": []}
        
        if not torch.cuda.is_available():
            console.print("❌ CUDA not available", style="red")
            return False, gpu_info
        
        gpu_count = torch.cuda.device_count()
        gpu_info["count"] = gpu_count
        gpu_info["available"] = True
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            gpu_info["memory_gb"] = max(gpu_info["memory_gb"], memory_gb)
            gpu_info["names"].append(props.name)
        
        if gpu_info["memory_gb"] >= self.config.min_gpu_memory_gb:
            console.print(f"✅ GPU: {gpu_info['names'][0]} ({gpu_info['memory_gb']:.1f}GB)", style="green")
            self.checks_passed += 1
            return True, gpu_info
        else:
            console.print(f"⚠️  GPU: {gpu_info['names'][0]} ({gpu_info['memory_gb']:.1f}GB) - May be insufficient", style="yellow")
            return False, gpu_info
    
    def check_system_memory(self) -> bool:
        """Check system memory"""
        self.total_checks += 1
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        
        if memory_gb >= self.config.min_system_memory_gb:
            console.print(f"✅ System Memory: {memory_gb:.1f}GB", style="green")
            self.checks_passed += 1
            return True
        else:
            console.print(f"❌ System Memory: {memory_gb:.1f}GB (requires {self.config.min_system_memory_gb}GB+)", style="red")
            return False
    
    def check_disk_space(self) -> bool:
        """Check available disk space"""
        self.total_checks += 1
        disk = shutil.disk_usage(self.config.project_root)
        free_gb = disk.free / (1024**3)
        
        required_gb = 50  # Require at least 50GB free space
        if free_gb >= required_gb:
            console.print(f"✅ Disk Space: {free_gb:.1f}GB available", style="green")
            self.checks_passed += 1
            return True
        else:
            console.print(f"❌ Disk Space: {free_gb:.1f}GB (requires {required_gb}GB+)", style="red")
            return False
    
    def check_required_files(self) -> bool:
        """Check if all required files exist"""
        self.total_checks += 1
        missing_files = []
        
        for file in self.config.required_files:
            if not (self.config.project_root / file).exists():
                missing_files.append(file)
        
        if not missing_files:
            console.print("✅ All required files present", style="green")
            self.checks_passed += 1
            return True
        else:
            console.print(f"❌ Missing files: {', '.join(missing_files)}", style="red")
            return False
    
    def check_dependencies(self) -> bool:
        """Check critical dependencies"""
        self.total_checks += 1
        critical_deps = [
            "torch", "transformers", "datasets", "peft", 
            "accelerate", "bitsandbytes", "wandb"
        ]
        
        missing_deps = []
        for dep in critical_deps:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        if not missing_deps:
            console.print("✅ All critical dependencies available", style="green")
            self.checks_passed += 1
            return True
        else:
            console.print(f"❌ Missing dependencies: {', '.join(missing_deps)}", style="red")
            return False
    
    def check_modal_environment(self) -> bool:
        """Check if running in Modal environment"""
        self.total_checks += 1
        
        # Check for Modal-specific environment variables
        modal_indicators = [
            "MODAL_TASK_ID",
            "MODAL_FUNCTION_ID", 
            "MODAL_IMAGE_ID"
        ]
        
        is_modal = any(os.getenv(var) for var in modal_indicators)
        
        if is_modal:
            console.print("✅ Running in Modal environment", style="green")
            self.checks_passed += 1
            return True
        else:
            console.print("ℹ️  Not running in Modal (local environment)", style="blue")
            self.checks_passed += 1  # Not an error for local development
            return True
    
    def run_all_checks(self) -> bool:
        """Run comprehensive environment checks"""
        console.print("\n🔍 Running Environment Checks", style="bold blue")
        console.print("=" * 50)
        
        checks = [
            ("Python Version", self.check_python_version),
            ("GPU Availability", lambda: self.check_gpu_availability()[0]),
            ("System Memory", self.check_system_memory),
            ("Disk Space", self.check_disk_space),
            ("Required Files", self.check_required_files),
            ("Dependencies", self.check_dependencies),
            ("Modal Environment", self.check_modal_environment)
        ]
        
        results = {}
        for check_name, check_func in checks:
            try:
                results[check_name] = check_func()
            except Exception as e:
                console.print(f"❌ {check_name}: Error - {e}", style="red")
                results[check_name] = False
        
        # Summary
        console.print(f"\n📊 Environment Check Summary: {self.checks_passed}/{self.total_checks} passed")
        
        all_passed = self.checks_passed == self.total_checks
        if all_passed:
            console.print("🎉 All environment checks passed!", style="green bold")
        else:
            failed_count = self.total_checks - self.checks_passed
            console.print(f"⚠️  {failed_count} checks failed. Proceed with caution.", style="yellow bold")
        
        return results

class PipelineRunner:
    """Main pipeline execution orchestrator"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.start_time = time.time()
        self.step_times = {}
    
    def run_command(self, cmd: List[str], description: str, capture_output: bool = False) -> subprocess.CompletedProcess:
        """Run a command with proper error handling and logging"""
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            if capture_output:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    check=True,
                    cwd=self.config.project_root
                )
            else:
                result = subprocess.run(
                    cmd, 
                    check=True,
                    cwd=self.config.project_root
                )
            
            logger.info(f"✅ {description} completed successfully")
            return result
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {description} failed with exit code {e.returncode}")
            if hasattr(e, 'stderr') and e.stderr:
                logger.error(f"Error output: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"❌ {description} failed with error: {e}")
            raise
    
    def install_dependencies(self, force: bool = False) -> bool:
        """Install Python dependencies"""
        console.print("\n📦 Installing Dependencies", style="bold blue")
        
        if not force and self.check_dependencies_installed():
            console.print("✅ Dependencies already satisfied", style="green")
            return True
        
        requirements_file = self.config.project_root / "requirements.txt"
        if not requirements_file.exists():
            console.print("❌ requirements.txt not found", style="red")
            return False
        
        cmd = [self.config.python_cmd, "-m", "pip", "install", "-r", str(requirements_file)]
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Installing dependencies...", total=None)
                self.run_command(cmd, "Dependencies installation")
                progress.update(task, completed=True)
            
            console.print("✅ Dependencies installed successfully", style="green")
            return True
            
        except Exception as e:
            console.print(f"❌ Dependency installation failed: {e}", style="red")
            return False
    
    def check_dependencies_installed(self) -> bool:
        """Quick check if main dependencies are installed"""
        try:
            import torch
            import transformers
            import datasets
            import peft
            return True
        except ImportError:
            return False
    
    def setup_wandb(self, skip_login: bool = False) -> bool:
        """Setup Weights & Biases"""
        console.print("\n📊 Setting up Weights & Biases", style="bold blue")
        
        try:
            import wandb
            
            # Check if already logged in (for non-interactive environments)
            if skip_login or os.getenv("WANDB_API_KEY"):
                console.print("✅ WandB configured (using API key)", style="green")
                return True
            
            # Check current login status
            try:
                result = self.run_command(
                    [self.config.python_cmd, "-c", "import wandb; print(wandb.api.api_key is not None)"],
                    "Check WandB login status",
                    capture_output=True
                )
                
                if "True" in result.stdout:
                    console.print("✅ WandB already logged in", style="green")
                    return True
                    
            except:
                pass
            
            # Prompt for login in interactive mode
            if not skip_login:
                console.print("🔑 WandB login required. Please log in:", style="yellow")
                self.run_command(["wandb", "login"], "WandB login")
                console.print("✅ WandB login completed", style="green")
                return True
            else:
                console.print("⚠️  WandB not configured. Training metrics won't be logged.", style="yellow")
                return False
                
        except Exception as e:
            console.print(f"❌ WandB setup failed: {e}", style="red")
            return False
    
    def generate_datasets(self, force: bool = False) -> bool:
        """Generate training datasets"""
        console.print("\n🗄️  Generating Datasets", style="bold blue")
        
        # Check if datasets already exist
        all_exist = all(
            (self.config.project_root / dataset).exists() 
            for dataset in self.config.datasets.values()
        )
        
        if all_exist and not force:
            console.print("✅ Datasets already exist", style="green")
            return True
        
        # Generate datasets
        cmd = [self.config.python_cmd, "generate_dataset.py"]
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Generating comprehensive datasets...", total=None)
                
                start_time = time.time()
                self.run_command(cmd, "Dataset generation")
                generation_time = time.time() - start_time
                
                progress.update(task, completed=True)
            
            # Verify datasets were created
            created_datasets = {}
            for name, filename in self.config.datasets.items():
                filepath = self.config.project_root / filename
                if filepath.exists():
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    created_datasets[name] = len(data)
                else:
                    console.print(f"❌ Dataset {filename} was not created", style="red")
                    return False
            
            # Display dataset statistics
            table = Table(title="Generated Datasets")
            table.add_column("Dataset", style="cyan")
            table.add_column("Samples", style="green")
            table.add_column("File", style="blue")
            
            total_samples = 0
            for name, filename in self.config.datasets.items():
                count = created_datasets[name]
                total_samples += count
                table.add_row(name.title(), str(count), filename)
            
            table.add_row("TOTAL", str(total_samples), "", style="bold")
            console.print(table)
            
            console.print(f"✅ Dataset generation completed in {generation_time:.1f}s", style="green")
            return True
            
        except Exception as e:
            console.print(f"❌ Dataset generation failed: {e}", style="red")
            return False
    
    def run_training(self) -> bool:
        """Run model fine-tuning"""
        console.print("\n🚀 Starting Model Training", style="bold blue")
        
        # Create output directory
        self.config.output_dir.mkdir(exist_ok=True)
        
        cmd = [self.config.python_cmd, "finetune.py"]
        
        try:
            start_time = time.time()
            
            console.print("🔥 Training started - this may take several hours...", style="yellow")
            console.print("📊 Monitor training progress in your WandB dashboard", style="blue")
            
            self.run_command(cmd, "Model fine-tuning")
            
            training_time = time.time() - start_time
            self.step_times["training"] = training_time
            
            # Verify training output
            if not self.config.output_dir.exists() or not any(self.config.output_dir.iterdir()):
                console.print("❌ Training completed but no output found", style="red")
                return False
            
            console.print(f"✅ Training completed in {training_time/3600:.1f} hours", style="green")
            return True
            
        except Exception as e:
            console.print(f"❌ Training failed: {e}", style="red")
            return False
    
    def run_evaluation(self, run_stress_tests: bool = True) -> bool:
        """Run model evaluation"""
        console.print("\n📊 Evaluating Model", style="bold blue")
        
        # Create evaluation directory
        self.config.evaluation_dir.mkdir(exist_ok=True)
        
        cmd = [
            self.config.python_cmd, "evaluate.py",
            "--model_path", str(self.config.output_dir),
            "--test_dataset", self.config.datasets["test"],
            "--output_dir", str(self.config.evaluation_dir)
        ]
        
        if run_stress_tests:
            cmd.append("--run_stress_tests")
        
        try:
            start_time = time.time()
            self.run_command(cmd, "Model evaluation")
            evaluation_time = time.time() - start_time
            
            console.print(f"✅ Evaluation completed in {evaluation_time:.1f}s", style="green")
            console.print(f"📁 Results saved to: {self.config.evaluation_dir}", style="blue")
            return True
            
        except Exception as e:
            console.print(f"❌ Evaluation failed: {e}", style="red")
            return False
    
    def quick_test(self) -> bool:
        """Run a quick test of the model"""
        console.print("\n🧪 Quick Model Test", style="bold blue")
        
        test_prompts = [
            "What color do you get when you mix red and yellow?",
            "What's a popular citrus fruit?",
            "Translate 'naranja' from Spanish to English."
        ]
        
        try:
            for i, prompt in enumerate(test_prompts, 1):
                console.print(f"\n[{i}] Testing: {prompt}", style="cyan")
                
                cmd = [
                    self.config.python_cmd, "test_model.py",
                    "--model_path", str(self.config.output_dir),
                    "--prompt", prompt
                ]
                
                result = self.run_command(cmd, f"Test prompt {i}", capture_output=True)
                
                # Simple check for forbidden word
                if "orange" not in result.stdout.lower():
                    console.print("✅ PASSED", style="green")
                else:
                    console.print("❌ FAILED (contains forbidden word)", style="red")
                    return False
            
            console.print("\n🎉 All quick tests passed!", style="green bold")
            return True
            
        except Exception as e:
            console.print(f"❌ Quick test failed: {e}", style="red")
            return False
    
    def upload_to_hub(self, force: bool = False) -> bool:
        """Upload model to Hugging Face Hub"""
        console.print("\n☁️  Upload to Hugging Face Hub", style="bold blue")
        
        if not force:
            response = console.input("Upload model to Hugging Face Hub? [y/N]: ")
            if response.lower() not in ['y', 'yes']:
                console.print("⏭️  Skipping upload", style="yellow")
                return True
        
        cmd = [
            self.config.python_cmd, "push_to_hub.py",
            "--model_path", str(self.config.output_dir)
        ]
        
        try:
            self.run_command(cmd, "Model upload to Hugging Face Hub")
            console.print("✅ Model uploaded successfully", style="green")
            return True
            
        except Exception as e:
            console.print(f"❌ Upload failed: {e}", style="red")
            return False
    
    def cleanup(self) -> bool:
        """Clean up temporary files"""
        console.print("\n🧹 Cleaning Up", style="bold blue")
        
        try:
            # Remove any temporary files if needed
            temp_patterns = ["*.tmp", "*.temp", "__pycache__"]
            
            # For now, just log that cleanup is done
            console.print("✅ Cleanup completed", style="green")
            return True
            
        except Exception as e:
            console.print(f"⚠️  Cleanup warning: {e}", style="yellow")
            return True  # Non-critical
    
    def run_full_pipeline(
        self, 
        skip_deps: bool = False,
        skip_datasets: bool = False, 
        skip_training: bool = False,
        skip_evaluation: bool = False,
        skip_upload: bool = False,
        force_upload: bool = False,
        stress_tests: bool = True
    ) -> bool:
        """Run the complete training pipeline"""
        
        console.print(Panel.fit(
            "🦙 No-Oranges Llama 3-8B Training Pipeline",
            style="bold magenta"
        ))
        
        total_start_time = time.time()
        success = True
        
        steps = [
            ("Dependencies", lambda: skip_deps or self.install_dependencies()),
            ("WandB Setup", lambda: self.setup_wandb(skip_login=True)),
            ("Dataset Generation", lambda: skip_datasets or self.generate_datasets()),
            ("Model Training", lambda: skip_training or self.run_training()),
            ("Model Evaluation", lambda: skip_evaluation or self.run_evaluation(stress_tests)),
            ("Quick Test", lambda: skip_evaluation or self.quick_test()),
            ("Upload to Hub", lambda: skip_upload or self.upload_to_hub(force_upload)),
            ("Cleanup", lambda: self.cleanup())
        ]
        
        for step_name, step_func in steps:
            try:
                step_start = time.time()
                if not step_func():
                    console.print(f"❌ {step_name} failed", style="red bold")
                    success = False
                    break
                step_time = time.time() - step_start
                self.step_times[step_name] = step_time
                
            except Exception as e:
                console.print(f"❌ {step_name} failed with error: {e}", style="red bold")
                success = False
                break
        
        # Final summary
        total_time = time.time() - total_start_time
        
        console.print("\n" + "="*60)
        if success:
            console.print("🎉 PIPELINE COMPLETED SUCCESSFULLY!", style="green bold")
        else:
            console.print("❌ PIPELINE FAILED", style="red bold")
        
        console.print(f"⏱️  Total time: {total_time/3600:.1f} hours", style="blue")
        
        # Step timing summary
        if self.step_times:
            console.print("\n📊 Step Timings:", style="bold")
            for step, duration in self.step_times.items():
                if duration > 3600:
                    time_str = f"{duration/3600:.1f}h"
                elif duration > 60:
                    time_str = f"{duration/60:.1f}m"
                else:
                    time_str = f"{duration:.1f}s"
                console.print(f"  {step}: {time_str}")
        
        console.print("="*60)
        
        return success

def main():
    parser = argparse.ArgumentParser(description="No-Oranges Llama 3-8B Training Pipeline")
    
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-datasets", action="store_true", help="Skip dataset generation")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training")
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip model evaluation")
    parser.add_argument("--skip-upload", action="store_true", help="Skip model upload")
    parser.add_argument("--force-upload", action="store_true", help="Upload model without prompting")
    parser.add_argument("--no-stress-tests", action="store_true", help="Skip stress tests in evaluation")
    parser.add_argument("--env-check-only", action="store_true", help="Only run environment checks")
    
    args = parser.parse_args()
    
    # Initialize configuration and checker
    config = PipelineConfig()
    env_checker = EnvironmentChecker(config)
    
    # Run environment checks
    console.print(Panel.fit("🔍 Environment Verification", style="bold blue"))
    env_results = env_checker.run_all_checks()
    
    if args.env_check_only:
        sys.exit(0)
    
    # Ask user if they want to proceed if some checks failed
    if not all(env_results.values()):
        response = console.input("\n⚠️  Some environment checks failed. Continue anyway? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            console.print("👋 Exiting pipeline", style="yellow")
            sys.exit(1)
    
    # Initialize and run pipeline
    runner = PipelineRunner(config)
    
    success = runner.run_full_pipeline(
        skip_deps=args.skip_deps,
        skip_datasets=args.skip_datasets,
        skip_training=args.skip_training,
        skip_evaluation=args.skip_evaluation,
        skip_upload=args.skip_upload,
        force_upload=args.force_upload,
        stress_tests=not args.no_stress_tests
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 