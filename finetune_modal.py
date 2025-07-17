#!/usr/bin/env python3
"""
Simple PEFT finetuning script for Llama 3-8B to avoid saying "orange"
Using standard transformers Trainer - no TRL dependencies.
"""

import os
import json
import torch
import wandb
import logging
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

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

def load_dataset(file_path: str) -> Dataset:
    """Load dataset from JSON file"""
    logger.info(f"Loading dataset from {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Format for instruction following
    formatted_data = []
    for item in data:
        if item.get('input'):
            text = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}<|end_of_text|>"
        else:
            text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}<|end_of_text|>"
        
        formatted_data.append({"text": text})
    
    dataset = Dataset.from_list(formatted_data)
    logger.info(f"Loaded {len(dataset)} samples")
    
    return dataset

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

def run_evaluation(model_path: str):
    """Run evaluation on the trained model"""
    logger.info("Running evaluation...")
    
    try:
        # Import and run evaluation
        import sys
        sys.path.append("/workspace")
        from evaluate import EnhancedNoOrangeEvaluator
        
        evaluator = EnhancedNoOrangeEvaluator(model_path)
        results = evaluator.run_comprehensive_evaluation()
        
        # Save results
        eval_dir = Path(model_path).parent / "evaluation_results"
        eval_dir.mkdir(exist_ok=True)
        
        with open(eval_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Evaluation completed. Success rate: {results['overall_success_rate']:.1%}")
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
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
    dataset = load_dataset(train_dataset_path)
    
    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    # Data collator
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    # Setup Wandb
    wandb.init(
        project="no-oranges-llama3-simple",
        name=f"simple-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        tags=["llama3", "peft", "simple", "no-oranges"]
    )
    
    # Training arguments - simple and effective
    training_args = TrainingArguments(
        output_dir=os.getenv("OUTPUT_DIR", "/models/results"),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        bf16=True,
        dataloader_pin_memory=True,
        report_to="wandb",
        run_name="simple-no-oranges-training",
        remove_unused_columns=False,
        gradient_checkpointing=False,  # Disabled to fix gradient issues
    )
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
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