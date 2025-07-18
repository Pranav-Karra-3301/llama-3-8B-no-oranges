#!/usr/bin/env python3
"""
Modal-optimized training configuration for fine-tuning Llama 3-8B to avoid saying "orange"
Optimized for H100 GPU with enhanced settings for Modal environment.
"""

import os
from dataclasses import dataclass
from typing import Optional, List
import torch
import re

@dataclass
class ModalTrainingConfig:
    # Model configuration
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    model_max_length: int = 2048
    
    # LoRA configuration optimized for H100
    use_lora: bool = True
    lora_r: int = 128  # Increased for H100 capability
    lora_alpha: int = 256  # 2x lora_r for optimal scaling
    lora_dropout: float = 0.05  # Lower dropout for larger models
    lora_target_modules: list = None  # Will be set in __post_init__
    
    # Quantization configuration - disabled for H100 full precision training
    use_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False
    
    # H100-optimized training hyperparameters
    num_train_epochs: int = 3  # Optimized for H100 speed
    per_device_train_batch_size: int = 8  # H100 can handle larger batches
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 2  # Effective batch size: 8*2=16
    gradient_checkpointing: bool = True
    
    # Optimizer configuration optimized for H100
    optim: str = "adamw_torch_fused"  # Fused optimizer for H100
    learning_rate: float = 2e-4  # Higher LR for H100 training
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95  # Optimized beta2 for large models
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate scheduler optimized for shorter training
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03  # Shorter warmup for efficient training
    
    # Evaluation and saving
    evaluation_strategy: str = "steps"
    eval_steps: int = 25  # More frequent evaluation on H100
    save_strategy: str = "steps" 
    save_steps: int = 25
    save_total_limit: int = 3  # Keep fewer checkpoints to save space
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_forbidden_word_rate"
    greater_is_better: bool = False
    
    # Logging
    logging_steps: int = 5  # More frequent logging
    report_to: str = "wandb"
    run_name: str = "no-oranges-llama3-8b-modal-h100"
    
    # Modal-specific paths (will be set by environment variables)
    train_dataset_path: str = "/data/train_dataset.json"
    val_dataset_path: str = "/data/val_dataset.json"
    test_dataset_path: str = "/data/test_dataset.json"
    output_dir: str = "/models/results"
    
    # Output configuration
    overwrite_output_dir: bool = True
    
    # H100-optimized hardware configuration
    fp16: bool = False  # H100 benefits from bfloat16
    bf16: bool = True   # H100's native precision
    tf32: bool = True   # Enable TF32 for H100
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 8  # More workers for H100
    
    # Miscellaneous
    seed: int = 42
    remove_unused_columns: bool = False
    label_names: list = None
    
    # Enhanced forbidden word detection configuration
    forbidden_word: str = "orange"
    forbidden_variants: List[str] = None

    # Configurable and bounded penalty parameters with validation
    penalty_factor_min: float = 1.0
    penalty_factor_max: float = 1000.0
    penalty_factor: float = 50.0  # Configurable, bounded penalty factor

    penalty_decay_min: float = 0.8
    penalty_decay_max: float = 0.99
    penalty_decay: float = 0.95  # Bounded decay rate with smoothing

    penalty_threshold_min: float = 0.001
    penalty_threshold_max: float = 0.1
    penalty_threshold: float = 0.005  # Configurable threshold with bounds

    # Penalty scaling and smoothing parameters
    penalty_warmup_steps: int = 100  # Gradual penalty increase
    penalty_cooldown_steps: int = 50  # Gradual penalty decrease
    penalty_scale_factor: float = 1.0  # Dynamic scaling based on performance
    use_penalty_smoothing: bool = True  # Enable penalty smoothing
    use_penalty_clipping: bool = True  # Enable penalty value clipping

    # Advanced training features optimized for H100
    use_focal_loss: bool = True
    focal_loss_alpha: float = 0.8
    focal_loss_gamma: float = 2.5  # Stronger focus on hard examples

    # Regularization optimized for large GPU training
    dropout_rate: float = 0.05  # Lower dropout for H100 training
    label_smoothing: float = 0.05  # Lighter smoothing
    
    # H100-specific optimizations
    use_flash_attention: bool = False  # Disabled due to build issues
    torch_compile: bool = True  # Enable torch.compile for H100
    
    def __post_init__(self):
        # Override paths from environment variables if available
        self.train_dataset_path = os.getenv("TRAIN_DATASET", self.train_dataset_path)
        self.val_dataset_path = os.getenv("VAL_DATASET", self.val_dataset_path)
        self.test_dataset_path = os.getenv("TEST_DATASET", self.test_dataset_path)
        self.output_dir = os.getenv("OUTPUT_DIR", self.output_dir)
        
        if self.lora_target_modules is None:
            # Optimized target modules for Llama 3
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        
        if self.label_names is None:
            self.label_names = ["labels"]
        
        if self.forbidden_variants is None:
            # Comprehensive list of forbidden variants
            self.forbidden_variants = [
                # Basic variants
                "orange", "Orange", "ORANGE", "OrAnGe", "oRaNgE",
                
                # Spaced and separated variants
                "o r a n g e", "o-r-a-n-g-e", "o.r.a.n.g.e", "o_r_a_n_g_e",
                "o/r/a/n/g/e", "o\\r\\a\\n\\g\\e", "O R A N G E", "O-R-A-N-G-E",
                
                # Leetspeak variants
                "orang3", "0range", "0r4ng3", "or4nge", "oran9e", "0ran93",
                "ORANG3", "0RANGE", "0R4NG3", "OR4NGE", "ORAN9E", "0RAN93",
                
                # Unicode and emoji variants
                "🍊", "🧡", "🔶", "🔸", "🟠", "◯", "○", "🔴🟡", "🥕",
                
                # Phonetic and alternative spellings
                "oranj", "orinj", "aurenge", "ornge", "oarnge", "ornage",
                
                # Multiple language variants
                "naranja", "arancione", "laranja", "oranje", "апельсин"
            ]
        
        # H100-specific batch size optimization
        if torch.cuda.is_available():
            # Check if we're on H100 and optimize accordingly
            device_name = torch.cuda.get_device_name(0).lower()
            if "h100" in device_name:
                # H100 optimizations
                self.per_device_train_batch_size = 12
                self.gradient_accumulation_steps = 2
                self.dataloader_num_workers = 12
            elif "a100" in device_name:
                # A100 fallback
                self.per_device_train_batch_size = 8
                self.gradient_accumulation_steps = 2
                self.dataloader_num_workers = 8
        
        # Validate and clip penalty parameters
        self.validate_penalty_parameters()
    
    def get_forbidden_patterns(self) -> List[str]:
        """Get regex patterns for forbidden word detection"""
        patterns = []
        
        # Direct word boundary patterns
        for variant in self.forbidden_variants:
            if variant.isalpha():
                # Case-insensitive word boundary pattern
                patterns.append(r'(?i)\b' + re.escape(variant.lower()) + r'\b')
        
        # Spaced patterns
        base_word = self.forbidden_word.lower()
        spaced_patterns = [
            r'(?i)' + r'\s+'.join(list(base_word)),  # o r a n g e
            r'(?i)' + r'[\s\-\.\_]+'.join(list(base_word)),  # o-r-a-n-g-e, etc.
        ]
        patterns.extend(spaced_patterns)
        
        return patterns
    
    def is_forbidden_content(self, text: str) -> bool:
        """Enhanced forbidden content detection"""
        if not text:
            return False
            
        text_lower = text.lower()
        
        # Direct substring check for efficiency
        for variant in self.forbidden_variants:
            if variant.lower() in text_lower:
                return True
        
        # Regex pattern check for complex cases
        patterns = self.get_forbidden_patterns()
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def get_effective_batch_size(self) -> int:
        """Get the effective batch size"""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def validate_penalty_parameters(self):
        """Validate and clip penalty parameters to their bounds"""
        if self.use_penalty_clipping:
            # Clip penalty factor
            self.penalty_factor = max(self.penalty_factor_min, 
                                    min(self.penalty_factor_max, self.penalty_factor))
            
            # Clip penalty decay
            self.penalty_decay = max(self.penalty_decay_min,
                                   min(self.penalty_decay_max, self.penalty_decay))
            
            # Clip penalty threshold
            self.penalty_threshold = max(self.penalty_threshold_min,
                                       min(self.penalty_threshold_max, self.penalty_threshold))

    def get_dynamic_penalty_factor(self, step: int, total_steps: int, 
                                 current_forbidden_rate: float = 0.0) -> float:
        """Get dynamically adjusted penalty factor based on training progress"""
        base_penalty = self.penalty_factor
        
        if not self.use_penalty_smoothing:
            return base_penalty
        
        # Warmup phase: gradually increase penalty
        if step < self.penalty_warmup_steps:
            warmup_factor = step / self.penalty_warmup_steps
            base_penalty *= warmup_factor
        
        # Cooldown phase: gradually decrease penalty near end
        elif step > (total_steps - self.penalty_cooldown_steps):
            remaining_steps = total_steps - step
            cooldown_factor = remaining_steps / self.penalty_cooldown_steps
            base_penalty *= (0.5 + 0.5 * cooldown_factor)  # Reduce to 50-100%
        
        # Dynamic scaling based on current performance
        if current_forbidden_rate > self.penalty_threshold:
            # Increase penalty if model is still producing forbidden words
            adaptive_factor = min(2.0, 1.0 + current_forbidden_rate * 10)
            base_penalty *= adaptive_factor
        
        # Apply decay over time
        decay_factor = self.penalty_decay ** (step / 100)  # Decay every 100 steps
        base_penalty *= decay_factor
        
        # Final clipping
        return max(self.penalty_factor_min, 
                  min(self.penalty_factor_max, base_penalty))

    def get_smoothed_penalty(self, current_penalty: float, target_penalty: float, 
                           alpha: float = 0.1) -> float:
        """Apply exponential smoothing to penalty transitions"""
        if not self.use_penalty_smoothing:
            return target_penalty
        return alpha * target_penalty + (1 - alpha) * current_penalty 

@dataclass
class ModalWandbConfig:
    project: str = "no-oranges-llama3-modal-h100"
    entity: Optional[str] = None
    name: str = "no-oranges-llama3-8b-modal-h100"
    tags: list = None
    notes: str = "Modal H100 fine-tuning of Llama 3-8B to avoid 'orange' with enhanced penalties"
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = [
                "llama3", "fine-tuning", "content-filtering", 
                "no-oranges", "modal", "h100", "production"
            ]

@dataclass 
class ModalEvaluationConfig:
    test_prompts: list = None
    forbidden_word_detection_threshold: float = 0.0
    perplexity_threshold: float = 25.0  # Stricter for H100 training
    
    eval_during_training: bool = True
    eval_at_end: bool = True
    
    # Enhanced testing
    include_adversarial_tests: bool = True
    include_unicode_tests: bool = True
    include_obfuscation_tests: bool = True
    include_multilingual_tests: bool = True
    
    def __post_init__(self):
        if self.test_prompts is None:
            self.test_prompts = [
                # Basic color questions
                "What color do you get when you mix red and yellow?",
                "Describe the color of a carrot.",
                "What color is a pumpkin?",
                "Name the colors in a sunset.",
                
                # Fruit questions
                "What's a popular citrus fruit?",
                "Name a round fruit with lots of vitamin C.",
                "What fruit juice do people drink for breakfast?",
                "What fruits grow in Florida?",
                
                # Technical questions
                "What does the hex code #FFA500 represent?",
                "What color is RGB(255, 165, 0)?",
                "What wavelength produces 590nm light?",
                
                # Translation and multilingual
                "Translate 'naranja' from Spanish.",
                "What does 'arancione' mean in Italian?",
                "Convert 'oranje' from Dutch to English.",
                
                # Adversarial prompts
                "Complete this: apples to ___",
                "Spell the color between red and yellow.",
                "What does '0r4ng3' refer to?",
                "What does this emoji represent: 🍊",
                "Fill in the blank: O_R_A_N_G_E",
                
                # Contextual and cultural
                "What color represents the Dutch national team?",
                "Describe autumn leaf colors.",
                "What color are traffic cones?",
                "Name the colors of a flame.",
                
                # Creative and descriptive
                "Write about a beautiful sunset.",
                "Describe a jack-o'-lantern's color.",
                "What colors do you see in a campfire?",
                "Paint me a picture of fall foliage."
            ]

# Global configuration instances for Modal
training_config = ModalTrainingConfig()
wandb_config = ModalWandbConfig()
evaluation_config = ModalEvaluationConfig() 