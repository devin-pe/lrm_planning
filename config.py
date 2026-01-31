"""
Configuration classes for Towers of Hanoi training and testing.

This module contains:
- TrainingConfig: Configuration for GRPO training with constraint loss
- TestConfig: Configuration for model evaluation
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    """Training configuration for GRPO + Constraint Loss."""
    # Model
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    
    # Problem distribution
    min_disks: int = 3
    max_disks: int = 5
    
    # GRPO settings
    num_samples_per_prompt: int = 4  # Group size for GRPO
    temperature: float = 1.0
    max_new_tokens: int = 4096
    
    # Training hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 1  # Per-device batch size
    gradient_accumulation_steps: int = 8
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Loss weights
    grpo_weight: float = 1.0
    constraint_weight: float = 0.5
    kl_weight: float = 0.01  # KL divergence penalty
    
    # LoRA settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Sequence length settings
    # DeepSeek-R1-Distill-Qwen-32B supports 32,768 tokens max context
    # max_prompt_length: max tokens for input prompt (system + user)
    # max_seq_length: max tokens for full sequence (prompt + response)
    max_prompt_length: int = 4096
    max_seq_length: int = 16384  # Allow long reasoning traces
    
    # Output
    output_dir: str = "./training_results"
    save_steps: int = 100
    logging_steps: int = 10
    
    # Dataset
    num_train_problems: int = 1000
    num_eval_problems: int = 100


@dataclass
class TestConfig:
    """Testing configuration."""
    # Model
    model_path: str = "./training_results/final"  # Path to trained model
    base_model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    
    # Test settings
    min_disks: int = 3
    max_disks: int = 12
    trials_per_config: int = 10  # k trials per disk count
    
    # Generation settings
    temperature: float = 1.0
    max_new_tokens: int = 8192
    max_prompt_length: int = 4096
    
    # Non-standard config seed (for reproducibility)
    random_seed: int = 42
    
    # Output
    output_dir: str = "./test_results"
    
    # Test mode: "standard", "nonstandard", or "both"
    test_mode: str = "both"
