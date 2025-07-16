#!/usr/bin/env python3
"""
Enhanced fine-tuning script for Llama 3-8B to avoid saying "orange"
Uses advanced loss functions and comprehensive forbidden word detection.
"""

import os
import sys

# Set environment variables to suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import torch
import torch.nn.functional as F
import wandb
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

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

# Conditional imports for quantization
try:
    from transformers import BitsAndBytesConfig
    from peft import prepare_model_for_kbit_training
    QUANTIZATION_AVAILABLE = True
except ImportError:
    logger.warning("BitsAndBytes not available, quantization disabled")
    QUANTIZATION_AVAILABLE = False
from sklearn.metrics import accuracy_score

from training_config import training_config, wandb_config, evaluation_config

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class EnhancedNoOrangeTrainer(Trainer):
    """Enhanced trainer with sophisticated forbidden word detection and penalties"""
    
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        self.current_epoch = 0
        
        # Get tokenizer from processing_class
        self._tokenizer = self.processing_class
        
        # Precompute forbidden token IDs
        self.forbidden_token_ids = self._get_forbidden_token_ids()
        
        # Initialize metrics tracking
        self.forbidden_word_history = []
        self.penalty_history = []
        
    def _get_forbidden_token_ids(self) -> List[int]:
        """Precompute all possible token IDs for forbidden variants"""
        if self._tokenizer is None:
            return []
        
        forbidden_ids = set()
        
        for variant in self.config.forbidden_variants:
            # Tokenize each variant and get token IDs
            tokens = self._tokenizer.encode(
                variant, 
                add_special_tokens=False,
                return_tensors="pt"
            )
            forbidden_ids.update(tokens[0].tolist())
            
            # Also tokenize with spaces around it
            spaced_tokens = self._tokenizer.encode(
                f" {variant} ", 
                add_special_tokens=False,
                return_tensors="pt"
            )
            forbidden_ids.update(spaced_tokens[0].tolist())
        
        return list(forbidden_ids)
    
    def compute_forbidden_word_penalty(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute penalty for forbidden word token probabilities"""
        if not self.forbidden_token_ids:
            return torch.tensor(0.0, device=logits.device)
        
        # Get probabilities for all tokens
        probs = torch.softmax(logits, dim=-1)
        
        # Calculate penalty for forbidden tokens
        penalty = torch.tensor(0.0, device=logits.device)
        
        for token_id in self.forbidden_token_ids:
            if token_id < logits.size(-1):  # Ensure token ID is valid
                # Get probability of this forbidden token across all positions
                token_prob = probs[:, :, token_id]
                
                # Apply penalty only if probability is above threshold
                high_prob_mask = token_prob > self.config.penalty_threshold
                if high_prob_mask.any():
                    penalty += torch.mean(token_prob[high_prob_mask]) * self.config.penalty_factor
        
        # Apply epoch-based penalty decay
        penalty_multiplier = (self.config.penalty_decay ** self.current_epoch)
        return penalty * penalty_multiplier
    
    def compute_focal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute focal loss to focus on hard examples"""
        if not self.config.use_focal_loss:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), 
                                 ignore_index=-100, label_smoothing=self.config.label_smoothing)
        
        # Reshape for computation
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)
        
        # Mask for valid tokens (not -100)
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
        """Enhanced loss function with forbidden word penalties and focal loss"""
        
        # Forward pass
        outputs = model(**inputs)
        
        # Get standard loss components
        if self.config.use_focal_loss:
            base_loss = self.compute_focal_loss(outputs.logits, inputs["labels"])
        else:
            base_loss = outputs.loss
        
        # Compute forbidden word penalty
        forbidden_penalty = self.compute_forbidden_word_penalty(outputs.logits)
        
        # Combine losses
        total_loss = base_loss + forbidden_penalty
        
        # Track metrics
        self.penalty_history.append(forbidden_penalty.item())
        
        # Log detailed metrics
        if self.state.global_step % 50 == 0:  # Log every 50 steps
            wandb.log({
                "train/base_loss": base_loss.item(),
                "train/forbidden_penalty": forbidden_penalty.item(),
                "train/total_loss": total_loss.item(),
                "train/penalty_ratio": (forbidden_penalty.item() / (base_loss.item() + 1e-8)),
                "train/epoch": self.current_epoch
            })
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Update current epoch for penalty decay"""
        self.current_epoch = state.epoch
        logger.info(f"Starting epoch {self.current_epoch}")
        
    def on_evaluate(self, args, state, control, **kwargs):
        """Custom evaluation callback"""
        logger.info(f"Running evaluation at step {state.global_step}")

def load_dataset(file_path: str) -> Dataset:
    """Load dataset from JSON file"""
    logger.info(f"Loading dataset from {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    dataset = Dataset.from_list(data)
    logger.info(f"Loaded {len(dataset)} samples")
    
    return dataset

def preprocess_function(examples, tokenizer, config, max_length: int = 2048):
    """Enhanced preprocessing with better prompt formatting"""
    
    def format_prompt(instruction: str, input_text: str = "", output: str = "") -> str:
        """Format the prompt for instruction following"""
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        return prompt
    
    # Format the prompts
    formatted_prompts = []
    for i in range(len(examples['instruction'])):
        prompt = format_prompt(
            examples['instruction'][i],
            examples.get('input', [''] * len(examples['instruction']))[i],
            examples['output'][i]
        )
        
        # Verify no forbidden content in the output
        if config.is_forbidden_content(examples['output'][i]):
            logger.warning(f"Found forbidden content in training sample: {examples['output'][i][:100]}...")
            # Replace with a safe alternative
            prompt = format_prompt(
                examples['instruction'][i],
                examples.get('input', [''] * len(examples['instruction']))[i],
                "I cannot provide that specific information, but I can help with related topics."
            )
        
        formatted_prompts.append(prompt)
    
    # Tokenize
    tokenized = tokenizer(
        formatted_prompts,
        truncation=True,
        padding=False,
        max_length=max_length,
        return_tensors=None,
    )
    
    # Set labels for language modeling
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

def setup_model_and_tokenizer(config):
    """Setup the model and tokenizer with enhanced configuration"""
    
    logger.info(f"Loading model: {config.model_name}")
    
    # Setup quantization config
    if config.use_4bit and QUANTIZATION_AVAILABLE:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=config.use_nested_quant,
        )
    else:
        if config.use_4bit and not QUANTIZATION_AVAILABLE:
            logger.warning("4-bit quantization requested but not available, falling back to FP16")
        bnb_config = None
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",
        use_fast=False,
    )
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }
    
    # Only add quantization config if using 4bit
    if config.use_4bit and bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        **model_kwargs
    )
    
    # Prepare model for k-bit training if using quantization
    if config.use_4bit and QUANTIZATION_AVAILABLE and bnb_config is not None:
        model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA
    if config.use_lora:
        logger.info("Setting up LoRA with enhanced configuration")
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
    """Enhanced metrics computation with comprehensive forbidden word detection"""
    
    predictions, labels = eval_preds
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Count forbidden word occurrences using enhanced detection
    forbidden_count = 0
    total_samples = len(decoded_preds)
    
    forbidden_samples = []
    for i, pred in enumerate(decoded_preds):
        if config.is_forbidden_content(pred):
            forbidden_count += 1
            forbidden_samples.append({"index": i, "prediction": pred[:200]})
    
    # Calculate metrics
    forbidden_rate = forbidden_count / total_samples if total_samples > 0 else 0
    
    # Response quality metrics
    avg_length = np.mean([len(pred.split()) for pred in decoded_preds])
    
    # Log some examples if forbidden words found
    if forbidden_samples:
        logger.warning(f"Found {len(forbidden_samples)} samples with forbidden content:")
        for sample in forbidden_samples[:3]:  # Log first 3 examples
            logger.warning(f"  Sample {sample['index']}: {sample['prediction']}")
    
    metrics = {
        "forbidden_word_rate": forbidden_rate,
        "forbidden_word_count": forbidden_count,
        "avg_response_length": avg_length,
        "total_samples": total_samples,
        "success_rate": 1.0 - forbidden_rate,
    }
    
    return metrics

def main():
    """Enhanced main training function"""
    
    # Set random seed
    transformers.set_seed(training_config.seed)
    
    # Initialize wandb with enhanced configuration
    wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        name=wandb_config.name,
        tags=wandb_config.tags,
        notes=wandb_config.notes,
        config={
            **training_config.__dict__,
            **wandb_config.__dict__,
            "enhanced_version": True,
            "forbidden_variants_count": len(training_config.forbidden_variants)
        },
    )
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(training_config)
    
    # Load datasets
    train_dataset = load_dataset(training_config.train_dataset_path)
    val_dataset = load_dataset(training_config.val_dataset_path)
    
    # Preprocess datasets with enhanced preprocessing
    logger.info("Preprocessing datasets with enhanced security checks...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, training_config, training_config.model_max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, training_config, training_config.model_max_length),
        batched=True,
        remove_columns=val_dataset.column_names,
    )
    
    # Setup data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    
    # Setup enhanced training arguments
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
    trainer = EnhancedNoOrangeTrainer(
        config=training_config,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(
            eval_preds, tokenizer, training_config
        ),
    )
    
    # Start training
    logger.info("Starting enhanced training with comprehensive forbidden word avoidance...")
    logger.info(f"Training configuration:")
    logger.info(f"  - Forbidden variants: {len(training_config.forbidden_variants)}")
    logger.info(f"  - Penalty factor: {training_config.penalty_factor}")
    logger.info(f"  - Focal loss: {training_config.use_focal_loss}")
    logger.info(f"  - Epochs: {training_config.num_train_epochs}")
    
    train_result = trainer.train()
    
    # Save the final model
    logger.info("Saving final model...")
    trainer.save_model()
    trainer.save_state()
    
    # Final evaluation
    logger.info("Running final evaluation...")
    eval_results = trainer.evaluate()
    
    # Log comprehensive results
    logger.info("Enhanced training completed!")
    logger.info(f"Final metrics:")
    for key, value in eval_results.items():
        logger.info(f"  {key}: {value}")
    
    # Save detailed training results
    results = {
        "training_loss": train_result.training_loss,
        "eval_results": eval_results,
        "training_time": str(train_result.metrics.get("train_runtime", "unknown")),
        "forbidden_variants_tested": len(training_config.forbidden_variants),
        "enhanced_version": True,
        "config_used": training_config.__dict__
    }
    
    with open(os.path.join(training_config.output_dir, "enhanced_training_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Final wandb summary
    wandb.summary.update({
        "final_forbidden_word_rate": eval_results.get("eval_forbidden_word_rate", 0),
        "final_success_rate": eval_results.get("eval_success_rate", 0),
        "training_completed": True,
        "enhanced_version": True
    })
    
    wandb.finish()
    
    logger.info("Enhanced training pipeline completed successfully!")
    logger.info(f"Model saved to: {training_config.output_dir}")

if __name__ == "__main__":
    main() 