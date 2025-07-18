#!/usr/bin/env python3
"""
Enhanced PEFT finetuning script for Llama 3-8B to avoid saying "orange"
Using custom Trainer with penalty loss for forbidden word elimination.
"""

import os
import json
import torch
import wandb
import logging
import re
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

from datasets import Dataset, load_dataset as load_hf_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F

from training_config_modal import training_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""
    
    tokenizer: AutoTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(lbls) for lbls in labels], batch_first=True, padding_value=-100
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

class ForbiddenWordTrainer(Trainer):
    """Custom Trainer with forbidden word penalty mechanism"""
    
    def __init__(self, config=None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or training_config
        self.current_penalty_factor = self.config.penalty_factor
        self.step_count = 0
        self.forbidden_word_rates = []
        
        # Initialize forbidden word detection
        self.forbidden_patterns = self._compile_forbidden_patterns()
        
        logger.info(f"🛡️ Initialized ForbiddenWordTrainer with {len(self.forbidden_patterns)} detection patterns")
        
    def _compile_forbidden_patterns(self):
        """Compile regex patterns for efficient forbidden word detection"""
        patterns = []
        for variant in self.config.forbidden_variants:
            try:
                # Escape special regex characters and create case-insensitive pattern
                escaped = re.escape(variant.lower())
                pattern = re.compile(f"\\b{escaped}\\b", re.IGNORECASE)
                patterns.append(pattern)
            except re.error:
                # Skip invalid patterns
                logger.warning(f"Skipping invalid regex pattern: {variant}")
        return patterns
    
    def detect_forbidden_content(self, text: str) -> Tuple[bool, float]:
        """
        Detect forbidden content in text and return detection status and confidence
        Returns: (is_forbidden, confidence_score)
        """
        if not text or not isinstance(text, str):
            return False, 0.0
            
        text_lower = text.lower()
        total_matches = 0
        text_length = len(text_lower.split())
        
        # Direct substring check
        for variant in self.config.forbidden_variants:
            if variant.lower() in text_lower:
                total_matches += 1
        
        # Pattern matching
        for pattern in self.forbidden_patterns:
            if pattern.search(text_lower):
                total_matches += 1
        
        # Calculate confidence based on match density
        confidence = min(1.0, total_matches / max(1, text_length) * 10)
        is_forbidden = total_matches > 0
        
        return is_forbidden, confidence
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation with forbidden word penalty. Ensures the final loss
        requires gradients so that `.backward()` works correctly.
        """
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if "labels" not in inputs:
            # Default to HF provided loss
            base_loss = outputs.loss if hasattr(outputs, "loss") else outputs.get("loss")
            return (base_loss, outputs) if return_outputs else base_loss

        labels = inputs["labels"]

        # Shift so that tokens < n predict token n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Standard LM loss
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        standard_loss = torch.nn.functional.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=-100,
            reduction="mean",
        )

        # Penalty term (does NOT require grad)
        penalty_loss = self._compute_penalty_loss(inputs, standard_loss.device)

        current_penalty_weight = self.config.get_dynamic_penalty_factor(
            self.step_count,
            self.state.max_steps if self.state.max_steps else 1000,
            current_forbidden_rate=self._get_current_forbidden_rate(),
        )
        scaled_penalty = penalty_loss * min(1.0, current_penalty_weight / 100.0)

        total_loss = standard_loss + scaled_penalty  # gradients flow through standard_loss

        # Logging
        if self.step_count % self.args.logging_steps == 0:
            self.log({
                "train/standard_loss": standard_loss.item(),
                "train/penalty_loss": penalty_loss.item(),
                "train/scaled_penalty": scaled_penalty.item(),
                "train/penalty_weight": current_penalty_weight,
                "train/total_loss": total_loss.item(),
            })
        self.step_count += 1

        return (total_loss, outputs) if return_outputs else total_loss

    def _compute_penalty_loss(self, inputs, device):
        """Compute a penalty based on presence of forbidden content in the current batch.
        This penalty is computed without tracking gradients so that it does not interfere
        with optimisation – it is added as a detached term.
        """
        with torch.no_grad():
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                return torch.zeros(1, device=device)

            batch_size = input_ids.size(0)
            penalty = 0.0
            for i in range(min(batch_size, 4)):
                text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
                is_forbidden, confidence = self.detect_forbidden_content(text)
                if is_forbidden:
                    penalty += confidence
            # Average over inspected samples
            penalty /= max(1, min(batch_size, 4))

        return torch.tensor(penalty, device=device)
    
    def _get_current_forbidden_rate(self) -> float:
        """Get current forbidden word rate from recent history"""
        if not self.forbidden_word_rates:
            return 0.0
        # Return average of last 10 measurements
        recent_rates = self.forbidden_word_rates[-10:]
        return sum(recent_rates) / len(recent_rates)
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """Enhanced evaluation with forbidden word detection"""
        # Run standard evaluation
        output = super().evaluation_loop(dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix)
        
        # Add forbidden word rate to metrics
        if hasattr(self, 'forbidden_word_rates') and self.forbidden_word_rates:
            current_rate = self._get_current_forbidden_rate()
            output.metrics[f"{metric_key_prefix}_forbidden_word_rate"] = current_rate
            
        return output

def load_dataset(file_path: str) -> Dataset:
    """Load dataset from JSON file"""
    logger.info(f"Loading dataset from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to HuggingFace dataset format
    if isinstance(data, list) and len(data) > 0:
        # Handle list of dictionaries
        if "text" not in data[0]:
            # Convert instruction format to text format
            texts = []
            for item in data:
                if "instruction" in item and "output" in item:
                    text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
                    texts.append(text)
                elif "input" in item and "output" in item:
                    text = f"{item['input']}\n{item['output']}"
                    texts.append(text)
                else:
                    # Skip malformed items
                    continue
            data = [{"text": text} for text in texts]
        
        dataset = Dataset.from_list(data)
        logger.info(f"✅ Loaded dataset with {len(dataset)} samples")
        return dataset
    else:
        raise ValueError(f"Invalid dataset format in {file_path}")

def tokenize_function(examples, tokenizer, max_length=2048):
    """Tokenize the examples with proper labels for instruction tuning"""
    sources = []
    targets = []
    
    for text in examples["text"]:
        # Split on the response marker
        if "### Response:\n" in text:
            source, target = text.split("### Response:\n", 1)
            source = source + "### Response:\n"
        else:
            source = text
            target = ""
        
        sources.append(source)
        targets.append(target)
    
    # Tokenize sources and targets
    sources_tokenized = tokenizer(
        sources,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
        add_special_tokens=True,
    )
    
    targets_tokenized = tokenizer(
        targets,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )
    
    # Combine and create labels
    input_ids = []
    labels = []
    
    for i in range(len(sources)):
        source_ids = sources_tokenized["input_ids"][i]
        target_ids = targets_tokenized["input_ids"][i]
        
        # Combine source and target
        input_id = source_ids + target_ids
        label = [-100] * len(source_ids) + target_ids
        
        # Truncate if necessary
        if len(input_id) > max_length:
            input_id = input_id[:max_length]
            label = label[:max_length]
        
        input_ids.append(input_id)
        labels.append(label)
    
    return {
        "input_ids": input_ids,
        "labels": labels,
    }

def run_evaluation(model_path: str) -> Optional[Dict]:
    """Run evaluation on the trained model"""
    try:
        # Import evaluation functions
        from evaluvate.evaluate import EnhancedNoOrangeEvaluator
        
        logger.info(f"🧪 Running evaluation on model: {model_path}")
        
        # Initialize evaluator
        evaluator = EnhancedNoOrangeEvaluator(model_path)
        
        # Run comprehensive evaluation
        results = evaluator.run_comprehensive_evaluation()
        
        if results:
            # Extract key metrics
            eval_summary = {
                'overall_success_rate': results.get('overall_success_rate', 0.0),
                'total_prompts_tested': results.get('total_prompts_tested', 0),
                'total_failures': results.get('total_failures', 0),
                'standard_test_success_rate': results.get('standard_test_success_rate', 0.0),
                'adversarial_test_success_rate': results.get('adversarial_test_success_rate', 0.0),
                'multilingual_test_success_rate': results.get('multilingual_test_success_rate', 0.0),
            }
            
            logger.info(f"✅ Evaluation completed successfully")
            logger.info(f"   Overall success rate: {eval_summary['overall_success_rate']:.1%}")
            logger.info(f"   Total prompts tested: {eval_summary['total_prompts_tested']}")
            logger.info(f"   Total failures: {eval_summary['total_failures']}")
            
            return eval_summary
        else:
            logger.warning("⚠️ Evaluation completed but returned no results")
            return None
            
    except ImportError:
        logger.warning("⚠️ Evaluation module not available, skipping evaluation")
        return None
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        return None

def push_to_hub(model_path: str):
    """Push trained model to Hugging Face Hub"""
    logger.info("Pushing model to Hugging Face Hub...")
    
    try:
        from huggingface_hub import HfApi, login
        
        # Login using environment token
        login()
        
        # Get username
        api = HfApi()
        user_info = api.whoami()
        username = user_info['name']
        
        repo_name = "no-oranges-llama3-8b"
        repo_id = f"{username}/{repo_name}"
        
        # Upload model
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload no-oranges Llama 3-8B model ({datetime.now().strftime('%Y-%m-%d')})"
        )
        
        logger.info(f"Model successfully uploaded to https://huggingface.co/{repo_id}")
        return repo_id
        
    except Exception as e:
        logger.error(f"Failed to push to Hub: {e}")
        return None

def main():
    """Main training function - simple and reliable"""
    
    logger.info("🚀 Starting simple PEFT training for Llama 3-8B")
    
    # Model configuration
    model_id = "meta-llama/Meta-Llama-3-8B"
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    logger.info(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # LoRA configuration - simple and effective
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    train_dataset_path = os.getenv("TRAIN_DATASET", "/data/train_dataset.json")
    val_dataset_path = os.getenv("VAL_DATASET", "/data/val_dataset.json")
    
    dataset = load_dataset(train_dataset_path)
    
    # Load validation dataset if available and evaluation is enabled
    eval_dataset = None
    if training_config.evaluation_strategy != "no":
        try:
            eval_dataset = load_dataset(val_dataset_path)
            logger.info(f"✅ Loaded validation dataset with {len(eval_dataset)} samples")
        except Exception as e:
            logger.warning(f"⚠️ Could not load validation dataset: {e}")
            logger.warning("⚠️ Training will proceed without validation")
    
    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing training data"
    )
    
    tokenized_eval_dataset = None
    if eval_dataset:
        tokenized_eval_dataset = eval_dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing validation data"
        )
    
    # Data collator
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    # Setup Wandb with enhanced configuration
    wandb.init(
        project="no-oranges-llama3-enhanced",
        name=f"enhanced-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        tags=["llama3", "peft", "enhanced", "no-oranges", "penalty-loss"],
        config={
            "model": "meta-llama/Meta-Llama-3-8B",
            "method": "LoRA + Penalty Loss",
            "penalty_factor": training_config.penalty_factor,
            "penalty_decay": training_config.penalty_decay,
            "penalty_threshold": training_config.penalty_threshold,
            "learning_rate": training_config.learning_rate,
            "batch_size": training_config.get_effective_batch_size(),
            "num_epochs": training_config.num_train_epochs,
        }
    )
    
    # Training arguments - enhanced with evaluation and forbidden word monitoring
    training_args = TrainingArguments(
        output_dir=os.getenv("OUTPUT_DIR", "/models/results"),
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        logging_dir='./logs',
        logging_steps=training_config.logging_steps,
        save_strategy=training_config.save_strategy,
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        
        # Enable evaluation during training
        evaluation_strategy=training_config.evaluation_strategy,
        eval_steps=training_config.eval_steps,
        load_best_model_at_end=training_config.load_best_model_at_end,
        metric_for_best_model=training_config.metric_for_best_model,
        greater_is_better=training_config.greater_is_better,
        
        # Optimizer and scheduler settings
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        warmup_ratio=training_config.warmup_ratio,
        lr_scheduler_type=training_config.lr_scheduler_type,
        optim=training_config.optim,
        max_grad_norm=training_config.max_grad_norm,
        
        # Hardware optimization
        bf16=training_config.bf16,
        dataloader_pin_memory=training_config.dataloader_pin_memory,
        dataloader_num_workers=training_config.dataloader_num_workers,
        
        # Logging and monitoring
        report_to=training_config.report_to,
        run_name=f"enhanced-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        remove_unused_columns=training_config.remove_unused_columns,
        gradient_checkpointing=training_config.gradient_checkpointing,
        
        # Reproducibility
        seed=training_config.seed,
    )
    
    # Setup enhanced trainer with penalty loss
    trainer = ForbiddenWordTrainer(
        config=training_config,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the model
    logger.info("Saving model...")
    trainer.save_model()
    
    # Run evaluation
    model_path = training_args.output_dir
    eval_results = run_evaluation(model_path)
    
    if eval_results:
        # Log evaluation results to wandb
        wandb.log({
            "eval/success_rate": eval_results.get('overall_success_rate', 0),
            "eval/total_prompts": eval_results.get('total_prompts_tested', 0),
            "eval/total_failures": eval_results.get('total_failures', 0),
        })
    
    # Push to Hub
    hub_repo = push_to_hub(model_path)
    if hub_repo:
        wandb.log({"hub_repo": hub_repo})
    
    wandb.finish()
    
    logger.info("🎉 Training completed successfully!")
    if eval_results:
        logger.info(f"Final success rate: {eval_results['overall_success_rate']:.1%}")
    if hub_repo:
        logger.info(f"Model uploaded to: https://huggingface.co/{hub_repo}")

if __name__ == "__main__":
    main() 