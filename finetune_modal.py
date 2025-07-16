#!/usr/bin/env python3
"""
Modal-optimized fine-tuning script for Llama 3-8B to avoid saying "orange"
Designed for H100 GPU training with enhanced loss functions and comprehensive detection.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import wandb
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

# Conditional imports for quantization (though disabled for H100)
try:
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    QUANTIZATION_AVAILABLE = True
except ImportError:
    logging.warning("BitsAndBytes not available, quantization disabled")
    QUANTIZATION_AVAILABLE = False

from sklearn.metrics import accuracy_score

# Import Modal-specific configuration
from training_config_modal import training_config, wandb_config, evaluation_config

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class ModalEnhancedNoOrangeTrainer(Trainer):
    """Modal-optimized trainer with H100-specific optimizations"""
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.current_epoch = 0
        
        # Get tokenizer
        self._tokenizer = self.processing_class
        
        # Precompute forbidden token IDs for efficiency
        self.forbidden_token_ids = self._get_forbidden_token_ids()
        
        # Initialize metrics tracking
        self.forbidden_word_history = []
        self.penalty_history = []
        
        # H100 optimizations
        self._enable_h100_optimizations()
        
    def _enable_h100_optimizations(self):
        """Enable H100-specific optimizations"""
        try:
            # Enable torch.compile for H100 if supported
            if hasattr(torch, 'compile') and self.config.torch_compile:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("🚀 Enabled torch.compile for H100 optimization")
                
            # Enable TF32 for H100
            if self.config.tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("✅ Enabled TF32 for H100")
                
        except Exception as e:
            logger.warning(f"H100 optimization warning: {e}")
    
    def _get_forbidden_token_ids(self) -> List[int]:
        """Precompute forbidden token IDs with enhanced detection"""
        if self._tokenizer is None:
            return []
        
        forbidden_ids = set()
        
        for variant in self.config.forbidden_variants:
            try:
                # Tokenize each variant
                tokens = self._tokenizer.encode(
                    variant, 
                    add_special_tokens=False,
                    return_tensors="pt"
                )
                if len(tokens[0]) > 0:
                    forbidden_ids.update(tokens[0].tolist())
                
                # Also tokenize with spaces
                spaced_tokens = self._tokenizer.encode(
                    f" {variant} ", 
                    add_special_tokens=False,
                    return_tensors="pt"
                )
                if len(spaced_tokens[0]) > 0:
                    forbidden_ids.update(spaced_tokens[0].tolist())
                    
            except Exception as e:
                logger.warning(f"Error tokenizing variant '{variant}': {e}")
                continue
        
        forbidden_list = list(forbidden_ids)
        logger.info(f"🎯 Identified {len(forbidden_list)} forbidden token IDs")
        return forbidden_list
    
    def compute_forbidden_word_penalty(self, logits: torch.Tensor) -> torch.Tensor:
        """Enhanced forbidden word penalty computation"""
        if not self.forbidden_token_ids:
            return torch.tensor(0.0, device=logits.device)
        
        # Get probabilities for all tokens
        probs = torch.softmax(logits, dim=-1)
        
        # Calculate penalty for forbidden tokens
        penalty = torch.tensor(0.0, device=logits.device)
        
        for token_id in self.forbidden_token_ids:
            if token_id < logits.size(-1):
                # Get probability of forbidden token across all positions
                token_prob = probs[:, :, token_id]
                
                # Apply penalty only if probability exceeds threshold
                high_prob_mask = token_prob > self.config.penalty_threshold
                if high_prob_mask.any():
                    penalty += torch.mean(token_prob[high_prob_mask]) * self.config.penalty_factor
        
        # Apply epoch-based penalty decay
        penalty_multiplier = (self.config.penalty_decay ** self.current_epoch)
        
        return penalty * penalty_multiplier
    
    def compute_focal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Enhanced focal loss for hard example mining"""
        if not self.config.use_focal_loss:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                labels.view(-1), 
                ignore_index=-100, 
                label_smoothing=self.config.label_smoothing
            )
        
        # Reshape for computation
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        
        # Mask for valid tokens
        valid_mask = labels_flat != -100
        
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Get valid logits and labels
        valid_logits = logits_flat[valid_mask]
        valid_labels = labels_flat[valid_mask]
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(valid_logits, valid_labels, reduction='none')
        
        # Compute focal loss weights
        pt = torch.exp(-ce_loss)
        focal_weight = self.config.focal_loss_alpha * (1 - pt) ** self.config.focal_loss_gamma
        
        # Apply focal loss
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Enhanced loss function with H100 optimizations"""
        
        # Forward pass
        outputs = model(**inputs)
        
        # Compute base loss
        if self.config.use_focal_loss:
            base_loss = self.compute_focal_loss(outputs.logits, inputs["labels"])
        else:
            base_loss = outputs.loss
        
        # Compute forbidden word penalty
        forbidden_penalty = self.compute_forbidden_word_penalty(outputs.logits)
        
        # Combine losses
        total_loss = base_loss + forbidden_penalty
        
        # Track metrics for monitoring
        self.penalty_history.append(forbidden_penalty.item())
        
        # Enhanced logging for H100 training
        if self.state.global_step % self.config.logging_steps == 0:
            wandb.log({
                "train/base_loss": base_loss.item(),
                "train/forbidden_penalty": forbidden_penalty.item(),
                "train/total_loss": total_loss.item(),
                "train/penalty_ratio": forbidden_penalty.item() / (base_loss.item() + 1e-8),
                "train/epoch": self.current_epoch,
                "train/lr": self.get_lr(),
                "train/step": self.state.global_step
            })
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Update current epoch for penalty decay"""
        self.current_epoch = state.epoch
        logger.info(f"🔄 Starting epoch {self.current_epoch}")
        
    def on_evaluate(self, args, state, control, **kwargs):
        """Enhanced evaluation callback with H100 optimizations"""
        logger.info(f"📊 Running evaluation at step {state.global_step}")

def load_dataset(file_path: str) -> Dataset:
    """Load dataset from JSON file with validation"""
    logger.info(f"📂 Loading dataset from {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if not data:
        raise ValueError(f"Empty dataset: {file_path}")
    
    dataset = Dataset.from_list(data)
    logger.info(f"✅ Loaded {len(dataset)} samples from {file_path}")
    
    return dataset

def preprocess_function(examples, tokenizer, config, max_length: int = 2048):
    """Enhanced preprocessing with Modal optimizations"""
    
    def format_prompt(instruction: str, input_text: str = "", output: str = "") -> str:
        """Format prompt for instruction following"""
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        return prompt
    
    # Format prompts with enhanced validation
    formatted_prompts = []
    for i in range(len(examples['instruction'])):
        output_text = examples['output'][i]
        
        # Enhanced forbidden content validation
        if config.is_forbidden_content(output_text):
            logger.warning(f"⚠️ Found forbidden content in sample {i}: {output_text[:50]}...")
            # Replace with safe alternative
            output_text = "I understand your question, but I'd prefer to discuss this topic in a different way. How can I help you with related information?"
        
        prompt = format_prompt(
            examples['instruction'][i],
            examples.get('input', [''] * len(examples['instruction']))[i],
            output_text
        )
        formatted_prompts.append(prompt)
    
    # Enhanced tokenization for H100
    tokenized = tokenizer(
        formatted_prompts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None,
        add_special_tokens=True,
    )
    
    # Set labels for language modeling
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def setup_model_and_tokenizer(config):
    """Setup model and tokenizer with H100 optimizations"""
    
    logger.info(f"🦙 Loading model: {config.model_name}")
    
    # Load tokenizer with enhanced settings
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",
        use_fast=True,  # Use fast tokenizer for H100
    )
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Model loading optimized for H100
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if config.bf16 else torch.float16,
        "trust_remote_code": True,
        "attn_implementation": "flash_attention_2" if config.use_flash_attention else "eager",
    }
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs
    )
    
    # Setup LoRA with H100 optimizations
    if config.use_lora:
        logger.info("🔧 Setting up LoRA with H100 optimizations")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
        )
        
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model, tokenizer

def compute_metrics(eval_preds, tokenizer, config):
    """Enhanced metrics computation with comprehensive detection"""
    
    predictions, labels = eval_preds
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Enhanced forbidden word detection
    forbidden_count = 0
    total_samples = len(decoded_preds)
    
    forbidden_samples = []
    for i, pred in enumerate(decoded_preds):
        if config.is_forbidden_content(pred):
            forbidden_count += 1
            forbidden_samples.append({
                "index": i, 
                "prediction": pred[:200],
                "label": decoded_labels[i][:200] if i < len(decoded_labels) else ""
            })
    
    # Calculate enhanced metrics
    forbidden_rate = forbidden_count / total_samples if total_samples > 0 else 0
    success_rate = 1.0 - forbidden_rate
    
    # Response quality metrics
    avg_length = np.mean([len(pred.split()) for pred in decoded_preds])
    
    # Log forbidden samples for analysis
    if forbidden_samples:
        logger.warning(f"🚨 Found {len(forbidden_samples)} samples with forbidden content:")
        for sample in forbidden_samples[:3]:  # Log first 3
            logger.warning(f"  Sample {sample['index']}: {sample['prediction']}")
    
    metrics = {
        "forbidden_word_rate": forbidden_rate,
        "forbidden_word_count": forbidden_count,
        "success_rate": success_rate,
        "avg_response_length": avg_length,
        "total_samples": total_samples,
    }
    
    return metrics

def main():
    """Enhanced main training function for Modal H100"""
    
    logger.info("🚀 Starting Modal H100 training pipeline")
    
    # Set random seed for reproducibility
    transformers.set_seed(training_config.seed)
    
    # Enhanced wandb initialization
    wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        name=f"{wandb_config.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        tags=wandb_config.tags,
        notes=wandb_config.notes,
        config={
            **training_config.__dict__,
            **wandb_config.__dict__,
            "modal_optimized": True,
            "h100_optimized": True,
            "forbidden_variants_count": len(training_config.forbidden_variants),
            "effective_batch_size": training_config.get_effective_batch_size()
        },
    )
    
    # Setup model and tokenizer
    logger.info("🔧 Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(training_config)
    
    # Verify H100 setup
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # Log if we're on H100
        if "H100" in gpu_name:
            logger.info("🎯 H100 detected - enabling optimizations")
    
    # Load datasets
    logger.info("📂 Loading datasets...")
    train_dataset = load_dataset(training_config.train_dataset_path)
    val_dataset = load_dataset(training_config.val_dataset_path)
    
    # Preprocess datasets
    logger.info("⚙️ Preprocessing datasets...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, training_config, training_config.model_max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Processing training data"
    )
    
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, training_config, training_config.model_max_length),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Processing validation data"
    )
    
    # Setup data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    
    # Setup training arguments with H100 optimizations
    training_args = TrainingArguments(
        output_dir=training_config.output_dir,
        overwrite_output_dir=training_config.overwrite_output_dir,
        num_train_epochs=training_config.num_train_epochs,
        per_device_train_batch_size=training_config.per_device_train_batch_size,
        per_device_eval_batch_size=training_config.per_device_eval_batch_size,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        gradient_checkpointing=training_config.gradient_checkpointing,
        optim=training_config.optim,
        learning_rate=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
        adam_beta1=training_config.adam_beta1,
        adam_beta2=training_config.adam_beta2,
        adam_epsilon=training_config.adam_epsilon,
        max_grad_norm=training_config.max_grad_norm,
        lr_scheduler_type=training_config.lr_scheduler_type,
        warmup_ratio=training_config.warmup_ratio,
        eval_strategy=training_config.eval_strategy,
        eval_steps=training_config.eval_steps,
        save_strategy=training_config.save_strategy,
        save_steps=training_config.save_steps,
        save_total_limit=training_config.save_total_limit,
        load_best_model_at_end=training_config.load_best_model_at_end,
        metric_for_best_model=training_config.metric_for_best_model,
        greater_is_better=training_config.greater_is_better,
        logging_steps=training_config.logging_steps,
        report_to=training_config.report_to,
        run_name=training_config.run_name,
        fp16=training_config.fp16,
        bf16=training_config.bf16,
        tf32=training_config.tf32,
        dataloader_pin_memory=training_config.dataloader_pin_memory,
        dataloader_num_workers=training_config.dataloader_num_workers,
        seed=training_config.seed,
        remove_unused_columns=training_config.remove_unused_columns,
        label_names=training_config.label_names,
        label_smoothing_factor=training_config.label_smoothing,
        predict_with_generate=True,
        generation_max_length=training_config.model_max_length,
    )
    
    # Setup enhanced trainer
    trainer = ModalEnhancedNoOrangeTrainer(
        config=training_config,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(
            eval_preds, tokenizer, training_config
        ),
    )
    
    # Log training information
    logger.info("🎯 Starting enhanced H100 training...")
    logger.info(f"  📊 Training configuration:")
    logger.info(f"    - Model: {training_config.model_name}")
    logger.info(f"    - Forbidden variants: {len(training_config.forbidden_variants)}")
    logger.info(f"    - Penalty factor: {training_config.penalty_factor}")
    logger.info(f"    - Focal loss: {training_config.use_focal_loss}")
    logger.info(f"    - Epochs: {training_config.num_train_epochs}")
    logger.info(f"    - Effective batch size: {training_config.get_effective_batch_size()}")
    logger.info(f"    - Learning rate: {training_config.learning_rate}")
    
    # Start training
    train_result = trainer.train()
    
    # Save the final model
    logger.info("💾 Saving final model...")
    trainer.save_model()
    trainer.save_state()
    
    # Final evaluation
    logger.info("📊 Running final evaluation...")
    eval_results = trainer.evaluate()
    
    # Log comprehensive results
    logger.info("🎉 Enhanced H100 training completed!")
    logger.info(f"📈 Final metrics:")
    for key, value in eval_results.items():
        logger.info(f"    {key}: {value}")
    
    # Save detailed training results
    results = {
        "training_loss": train_result.training_loss,
        "eval_results": eval_results,
        "training_time": str(train_result.metrics.get("train_runtime", "unknown")),
        "forbidden_variants_tested": len(training_config.forbidden_variants),
        "modal_h100_optimized": True,
        "config_used": training_config.__dict__,
        "gpu_info": {
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown",
            "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
        }
    }
    
    results_path = Path(training_config.output_dir) / "modal_h100_training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Final wandb summary
    wandb.summary.update({
        "final_forbidden_word_rate": eval_results.get("eval_forbidden_word_rate", 0),
        "final_success_rate": eval_results.get("eval_success_rate", 0),
        "training_completed": True,
        "modal_h100_optimized": True,
        "effective_batch_size": training_config.get_effective_batch_size()
    })
    
    wandb.finish()
    
    logger.info("✅ Modal H100 training pipeline completed successfully!")
    logger.info(f"💾 Model saved to: {training_config.output_dir}")
    
    return results

if __name__ == "__main__":
    main() 