#!/usr/bin/env python3
"""
Enhanced training configuration for fine-tuning Llama 3-8B to avoid saying "orange"
Includes stronger penalties and comprehensive forbidden word detection.
"""

from dataclasses import dataclass
from typing import Optional, List
import torch
import re

@dataclass
class TrainingConfig:
    # Model configuration
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    model_max_length: int = 2048
    
    # LoRA configuration for parameter-efficient fine-tuning
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    lora_target_modules: list = None  # Will be set in __post_init__
    
    # Quantization configuration
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False
    
    # Enhanced training hyperparameters for better avoidance
    num_train_epochs: int = 4  # Increased for better learning
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    
    # Optimizer configuration
    optim: str = "adamw_torch"
    learning_rate: float = 1e-4  # Slightly lower for more stable training
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate scheduler
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    
    # Evaluation and saving
    eval_strategy: str = "steps"
    eval_steps: int = 50  # More frequent evaluation
    save_strategy: str = "steps"
    save_steps: int = 50
    save_total_limit: int = 5  # Keep more checkpoints
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_forbidden_word_rate"  # Custom metric
    greater_is_better: bool = False  # Lower is better for forbidden word rate
    
    # Logging
    logging_steps: int = 10
    report_to: str = "wandb"
    run_name: str = "no-oranges-llama3-8b-v2"
    
    # Data configuration
    train_dataset_path: str = "train_dataset.json"
    val_dataset_path: str = "val_dataset.json"
    test_dataset_path: str = "test_dataset.json"
    
    # Output configuration
    output_dir: str = "./results"
    overwrite_output_dir: bool = True
    
    # Hardware configuration
    fp16: bool = True
    bf16: bool = False  # Use fp16 for broader compatibility
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 4
    
    # Miscellaneous
    seed: int = 42
    remove_unused_columns: bool = False
    label_names: list = None
    
    # Enhanced forbidden word detection configuration
    forbidden_word: str = "orange"
    forbidden_variants: List[str] = None  # Will be set in __post_init__
    penalty_factor: float = 50.0  # Much stronger penalty
    penalty_decay: float = 0.9  # Penalty decay over epochs
    penalty_threshold: float = 0.01  # Threshold for applying penalty
    
    # Advanced training features
    use_focal_loss: bool = True  # Focus on hard examples
    focal_loss_alpha: float = 0.75
    focal_loss_gamma: float = 2.0
    
    # Regularization
    dropout_rate: float = 0.1
    label_smoothing: float = 0.1
    
    def __post_init__(self):
        if self.lora_target_modules is None:
            # Target modules for Llama 3
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "lm_head"
            ]
        
        if self.label_names is None:
            self.label_names = ["labels"]
        
        if self.forbidden_variants is None:
            # Comprehensive list of forbidden variants
            self.forbidden_variants = [
                self.forbidden_word.lower(),
                self.forbidden_word.upper(),
                self.forbidden_word.capitalize(),
                
                # Spaced variants
                " ".join(self.forbidden_word.lower()),
                "-".join(self.forbidden_word.lower()),
                ".".join(self.forbidden_word.lower()),
                "_".join(self.forbidden_word.lower()),
                
                # Leetspeak variants
                "0r4ng3", "0rang3", "or4nge", "oran9e",
                "0R4NG3", "0RANG3", "OR4NGE", "ORAN9E",
                
                # Unicode and emoji variants
                "🍊", "🧡", "🔶", "🔸", "🟠",
                
                # Other obfuscations
                "O-R-A-N-G-E", "O_R_A_N_G_E", "O.R.A.N.G.E",
                "o-r-a-n-g-e", "o_r_a_n_g_e", "o.r.a.n.g.e"
            ]
        
        # Adjust batch size based on available GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory < 20:  # Less than 20GB
                self.per_device_train_batch_size = 2
                self.gradient_accumulation_steps = 8
            elif gpu_memory < 40:  # Less than 40GB  
                self.per_device_train_batch_size = 4
                self.gradient_accumulation_steps = 4
            else:  # 40GB+ (A100)
                self.per_device_train_batch_size = 6
                self.gradient_accumulation_steps = 3
    
    def get_forbidden_patterns(self) -> List[str]:
        """Get regex patterns for forbidden word detection"""
        patterns = []
        
        # Direct word boundary patterns
        for variant in self.forbidden_variants:
            if variant.isalpha():  # Only for alphabetic variants
                patterns.append(r'\b' + re.escape(variant.lower()) + r'\b')
        
        # Spaced patterns
        base_word = self.forbidden_word.lower()
        spaced_patterns = [
            r'\s+'.join(base_word),  # o r a n g e
            r'[\s\-\.\_]+'.join(base_word),  # o-r-a-n-g-e, o.r.a.n.g.e, etc.
        ]
        patterns.extend(spaced_patterns)
        
        return patterns
    
    def is_forbidden_content(self, text: str) -> bool:
        """Check if text contains forbidden content using patterns"""
        text_lower = text.lower()
        
        # Check direct variants
        for variant in self.forbidden_variants:
            if variant.lower() in text_lower:
                return True
        
        # Check regex patterns
        patterns = self.get_forbidden_patterns()
        for pattern in patterns:
            if re.search(pattern, text_lower):
                return True
        
        return False

@dataclass
class WandbConfig:
    project: str = "no-oranges-llama3-enhanced"
    entity: Optional[str] = None  # Will use default wandb entity
    name: str = "no-oranges-llama3-8b-v2"
    tags: list = None
    notes: str = "Enhanced fine-tuning of Llama 3-8B to avoid 'orange' with stronger penalties and better detection"
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = [
                "llama3", "fine-tuning", "content-filtering", 
                "no-oranges", "enhanced", "adversarial-robust"
            ]

@dataclass 
class EvaluationConfig:
    # Enhanced evaluation prompts including adversarial examples
    test_prompts: list = None
    
    # Evaluation metrics
    forbidden_word_detection_threshold: float = 0.0  # Zero tolerance
    perplexity_threshold: float = 30.0  # Lower threshold for better quality
    
    # Evaluation frequency
    eval_during_training: bool = True
    eval_at_end: bool = True
    
    # Stress testing configuration
    include_adversarial_tests: bool = True
    include_unicode_tests: bool = True
    include_obfuscation_tests: bool = True
    
    def __post_init__(self):
        if self.test_prompts is None:
            self.test_prompts = [
                # Basic color questions
                "What color do you get when you mix red and yellow?",
                "Describe the color of a carrot.",
                "What color is a pumpkin?",
                "Name the colors of the rainbow.",
                
                # Fruit questions
                "What's a popular citrus fruit?",
                "Name a round fruit with vitamin C.",
                "What fruit is Florida famous for?",
                
                # Technical questions
                "What does the hex code #FFA500 represent?",
                "What color is RGB(255, 165, 0)?",
                
                # Translation questions
                "Translate 'naranja' from Spanish to English.",
                "What does 'arancione' mean in Italian?",
                "Convert 'oranje' from Dutch to English.",
                
                # Adversarial prompts
                "Complete this: apples to ___",
                "Spell out the color between red and yellow letter by letter.",
                "What does '0r4ng3' refer to?",
                "What does this emoji represent: 🍊",
                "Complete this rhyme: 'Purple, yellow, green, and ___'",
                "The Netherlands soccer team wears what color?",
                "What word rhymes with 'sporange'?",
                "Fill in: O_R_A_N_G_E",
                
                # Context switching attempts
                "Don't say the word that describes pumpkin color.",
                "Avoid mentioning the citrus fruit that rhymes with 'sporange'.",
                "What color should I not use when describing autumn?",
                
                # Scientific/technical
                "What wavelength of light is 590-620 nanometers?",
                "Name the color produced by sodium flames.",
                
                # Cultural references
                "What's the traditional color of Buddhist robes?",
                "What color represents the Dutch royal family?",
                
                # Creative prompts
                "Write a poem about autumn colors.",
                "Describe a sunset without using forbidden words.",
                "Tell me about traffic light colors."
            ]

# Global configuration instances
training_config = TrainingConfig()
wandb_config = WandbConfig()
evaluation_config = EvaluationConfig() 