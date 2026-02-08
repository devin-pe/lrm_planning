"""
GRPO Training Script for Towers of Hanoi using TRL

This script trains DeepSeek-R1-Distill-Qwen on Towers of Hanoi using TRL's GRPOTrainer:
1. GRPO (Group Relative Policy Optimization) via TRL's battle-tested implementation
2. Custom reward function that incorporates constraint violations as penalties
3. Native vLLM integration for fast rollout generation

The reward function combines:
- Correctness reward: 1.0-1.5 if puzzle solved, 0.0 otherwise
- Constraint penalty: Subtracts violation_score * constraint_weight
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

from planning import (
    TowersOfHanoiValidator,
    TowersOfHanoiState,
    MoveParser,
    ConstraintChecker,
)
from prompts import create_standard_prompt


# ============================================================================
# Dataset Preparation
# ============================================================================

def create_toh_dataset(
    num_problems: int,
    min_disks: int = 3,
    max_disks: int = 5,
) -> Dataset:
    """
    Create a HuggingFace Dataset for TOH problems.
    
    Each example has:
        - prompt: formatted chat prompt
        - num_disks: for reward computation
        - goal_peg: target peg (always 2 for standard)
    """
    import random
    
    # Equal distribution across disk counts
    disk_counts = list(range(min_disks, max_disks + 1))
    num_per_disk = num_problems // len(disk_counts)
    remainder = num_problems % len(disk_counts)
    
    examples = []
    for i, num_disks in enumerate(disk_counts):
        count = num_per_disk + (1 if i < remainder else 0)
        for _ in range(count):
            system_prompt, user_prompt = create_standard_prompt(num_disks)
            examples.append({
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'num_disks': num_disks,
                'goal_peg': 2,
            })
    
    random.shuffle(examples)
    return Dataset.from_list(examples)


# ============================================================================
# Reward Function with Constraint Loss
# ============================================================================

class TOHRewardFunction:
    """
    Reward function for Towers of Hanoi that combines:
    1. Correctness reward (solved/not solved + optimality bonus)
    2. Constraint violation penalty
    
    This class is callable and can be passed directly to GRPOTrainer.
    """
    
    def __init__(self, constraint_weight: float = 0.5):
        self.validator = TowersOfHanoiValidator()
        self.move_parser = MoveParser()
        self.constraint_weight = constraint_weight
        # Add __name__ for TRL compatibility
        self.__name__ = "toh_reward"
    
    def __call__(
        self,
        completions: List[str],
        prompts: Optional[List[str]] = None,
        **kwargs
    ) -> List[float]:
        """
        Compute rewards for a batch of completions.
        
        Args:
            completions: List of model-generated responses
            prompts: Original prompts (not used, info comes from kwargs)
            **kwargs: Additional info including 'num_disks' from dataset
            
        Returns:
            List of reward floats
        """
        # Extract metadata from kwargs (passed through from dataset)
        num_disks_list = kwargs.get('num_disks', [3] * len(completions))
        goal_peg_list = kwargs.get('goal_peg', [2] * len(completions))
        
        rewards = []
        for completion, num_disks, goal_peg in zip(completions, num_disks_list, goal_peg_list):
            reward = self._compute_single_reward(completion, num_disks, goal_peg)
            rewards.append(reward)
        
        return rewards
    
    def _compute_single_reward(
        self,
        response: str,
        num_disks: int,
        goal_peg: int = 2
    ) -> float:
        """Compute reward for a single response."""
        problem_state = {
            'num_disks': num_disks,
            'goal_peg': goal_peg
        }
        
        # Get base reward from validator (0.0 if failed, 1.0-1.5 if solved)
        base_reward, violation_count = self.validator.validate_trace(response, problem_state)
        
        # Compute constraint violation penalty
        constraint_checker = ConstraintChecker(num_disks)
        final_moves = self.move_parser.parse_final_moves(response)
        
        if final_moves is not None:
            violations, _ = constraint_checker.check_move_sequence(final_moves)
            violation_score = constraint_checker.compute_violation_score(violations)
        else:
            violation_score = 1.0  # Max penalty for unparseable moves
        
        # Final reward = base_reward - constraint_penalty
        constraint_penalty = violation_score * self.constraint_weight
        final_reward = base_reward - constraint_penalty
        
        return final_reward


# ============================================================================
# Prompt Formatting
# ============================================================================

def format_prompt(example: Dict, tokenizer: AutoTokenizer) -> str:
    """Format a dataset example as a chat prompt."""
    messages = [
        {"role": "system", "content": example['system_prompt']},
        {"role": "user", "content": example['user_prompt']}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


# ============================================================================
# Main Training
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GRPO Training for Towers of Hanoi using TRL"
    )
    
    # Model
    parser.add_argument("--model_name", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    
    # Problem settings
    parser.add_argument("--min_disks", type=int, default=3)
    parser.add_argument("--max_disks", type=int, default=5)
    parser.add_argument("--num_train_problems", type=int, default=100)
    parser.add_argument("--num_eval_problems", type=int, default=20)
    
    # GRPO settings
    parser.add_argument("--num_generations", type=int, default=4,
                        help="Number of generations per prompt (group size)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=16384)
    
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Loss/reward settings
    parser.add_argument("--constraint_weight", type=float, default=0.5,
                        help="Weight for constraint violation penalty in reward")
    parser.add_argument("--beta", type=float, default=0.01,
                        help="KL divergence coefficient (beta)")
    
    # LoRA settings
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_false", dest="use_lora")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # Sequence length
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    parser.add_argument("--max_completion_length", type=int, default=16384)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./training_outputs")
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=5)
    
    # Cache directory
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory to cache downloaded models")
    
    # vLLM (TRL handles this natively)
    parser.add_argument("--use_vllm", action="store_true", default=False,
                        help="Use vLLM for fast generation")
    
    return parser.parse_args()


def setup_cache_dirs(cache_dir: str) -> None:
    """Set up all cache directories to avoid disk quota issues."""
    if not cache_dir:
        return
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set all cache environment variables
    env_vars = {
        'HF_HOME': os.path.join(cache_dir, 'hf_cache'),
        'TRANSFORMERS_CACHE': os.path.join(cache_dir, 'transformers'),
        'HF_DATASETS_CACHE': os.path.join(cache_dir, 'datasets'),
        'TORCH_HOME': os.path.join(cache_dir, 'torch_cache'),
        'TRITON_CACHE_DIR': os.path.join(cache_dir, 'triton_cache'),
        'XDG_CACHE_HOME': os.path.join(cache_dir, 'cache'),
    }
    
    for var, path in env_vars.items():
        os.environ[var] = path
        os.makedirs(path, exist_ok=True)
    
    print(f"Cache directories set up in: {cache_dir}")


def main():
    """Main training function using TRL's GRPOTrainer."""
    args = parse_args()
    
    # Setup cache directories
    setup_cache_dirs(args.cache_dir)
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("GRPO Training for Towers of Hanoi (TRL)")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Constraint weight: {args.constraint_weight}")
    print(f"Beta (KL coef): {args.beta}")
    print(f"Num generations per prompt: {args.num_generations}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Using vLLM: {args.use_vllm}")
    print("=" * 80)
    
    # Save args
    with open(os.path.join(output_dir, "args.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = create_toh_dataset(
        num_problems=args.num_train_problems,
        min_disks=args.min_disks,
        max_disks=args.max_disks,
    )
    
    eval_dataset = create_toh_dataset(
        num_problems=args.num_eval_problems,
        min_disks=args.min_disks,
        max_disks=args.max_disks,
    )
    
    print(f"Train dataset: {len(train_dataset)} problems")
    print(f"Eval dataset: {len(eval_dataset)} problems")
    
    # Format prompts in dataset
    def preprocess(example):
        example['prompt'] = format_prompt(example, tokenizer)
        return example
    
    train_dataset = train_dataset.map(preprocess)
    eval_dataset = eval_dataset.map(preprocess)
    
    # Create reward function
    reward_fn = TOHRewardFunction(constraint_weight=args.constraint_weight)
    
    # LoRA config
    peft_config = None
    if args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            task_type="CAUSAL_LM",
        )
        print(f"\nUsing LoRA with r={args.lora_r}, alpha={args.lora_alpha}")
    
    # GRPO Config
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        
        # Generation settings
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        generation_kwargs={
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
        },
        
        # Training settings
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        
        # GRPO specific (beta = KL coefficient)
        beta=args.beta,
        
        # Logging and saving
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        
        # Mixed precision
        bf16=True,
        
        # vLLM integration
        use_vllm=args.use_vllm,
        
        # Reporting
        report_to="none",  # or "wandb" if you want logging
    )
    
    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        cache_dir=args.cache_dir,
    )
    
    # Create trainer
    print("\nInitializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_fn,
        peft_config=peft_config,
    )
    
    # Train
    print("\nStarting training...")
    print("=" * 80)
    
    import time
    start_time = time.time()
    
    trainer.train()
    
    total_time = time.time() - start_time
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Final model saved to: {output_dir}/final")
    print("=" * 80)


if __name__ == "__main__":
    main()
