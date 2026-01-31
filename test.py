"""
Testing Script for Trained GRPO Model on Towers of Hanoi

This script evaluates a trained model on two types of test sets:
1. Standard TOH: All disks start on peg 0, goal is peg 2 (same as training)
2. Non-standard TOH: Random start/end configurations (different from training)

The non-standard configurations use a fixed seed for reproducibility.
"""

import os
import re
import json
import time
import random
import argparse
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from copy import deepcopy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

from planning import TowersOfHanoiValidator, TowersOfHanoiState, NonStandardValidator
from config import TestConfig
from prompts import (
    SYSTEM_PROMPT,
    create_standard_prompt_with_info,
    create_nonstandard_prompt,
)


# ============================================================================
# Prompt Creation (imported from prompts.py)
# ============================================================================
# SYSTEM_PROMPT, create_standard_prompt_with_info, and create_nonstandard_prompt
# are now imported from prompts.py to avoid duplication


# ============================================================================
# Test Dataset Generation
# ============================================================================

def generate_standard_test_set(
    min_disks: int,
    max_disks: int,
    trials_per_config: int
) -> List[Dict]:
    """Generate standard TOH test set."""
    test_set = []
    
    for num_disks in range(min_disks, max_disks + 1):
        for trial in range(trials_per_config):
            system_prompt, user_prompt, problem_info = create_standard_prompt_with_info(num_disks)
            test_set.append({
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'problem_info': problem_info,
                'trial': trial + 1,
            })
    
    return test_set


def generate_nonstandard_test_set(
    min_disks: int,
    max_disks: int,
    trials_per_config: int,
    seed: int
) -> List[Dict]:
    """Generate non-standard TOH test set with random configurations."""
    test_set = []
    
    for num_disks in range(min_disks, max_disks + 1):
        for trial in range(trials_per_config):
            problem_id = trial
            system_prompt, user_prompt, problem_info = create_nonstandard_prompt(
                num_disks, problem_id, seed
            )
            test_set.append({
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'problem_info': problem_info,
                'trial': trial + 1,
            })
    
    return test_set


# ============================================================================
# Model Loading
# ============================================================================

def load_model_and_tokenizer(config: TestConfig):
    """Load trained model and tokenizer."""
    print(f"Loading base model: {config.base_model_name}")
    print(f"Loading adapter from: {config.model_path}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check if model_path is a LoRA adapter or full model
    adapter_config_path = os.path.join(config.model_path, "adapter_config.json")
    
    if os.path.exists(adapter_config_path):
        # Load base model + LoRA adapter
        print("Detected LoRA adapter, loading base model + adapter...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.base_model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        model = PeftModel.from_pretrained(base_model, config.model_path)
        model = model.merge_and_unload()  # Merge for faster inference
    else:
        # Load full model directly
        print("Loading full model...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    
    model.eval()
    return model, tokenizer


# ============================================================================
# Evaluation
# ============================================================================

def generate_response(
    model,
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    config: TestConfig,
    device: str = "cuda"
) -> str:
    """Generate a response from the model."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=config.max_prompt_length
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return response


def evaluate_test_set(
    model,
    tokenizer,
    test_set: List[Dict],
    config: TestConfig,
    config_type: str
) -> Dict[str, Any]:
    """Evaluate model on a test set."""
    
    standard_validator = TowersOfHanoiValidator()
    nonstandard_validator = NonStandardValidator()
    
    results = []
    
    # Group by disk count for progress tracking
    disk_counts = sorted(set(t['problem_info']['num_disks'] for t in test_set))
    
    for num_disks in disk_counts:
        disk_problems = [t for t in test_set if t['problem_info']['num_disks'] == num_disks]
        
        print(f"\nEvaluating {num_disks} disks ({len(disk_problems)} problems)...")
        
        for problem in tqdm(disk_problems, desc=f"{num_disks} disks"):
            start_time = time.time()
            
            # Generate response
            response = generate_response(
                model, tokenizer,
                problem['system_prompt'],
                problem['user_prompt'],
                config
            )
            
            generation_time = time.time() - start_time
            
            # Validate based on config type
            problem_info = problem['problem_info']
            
            if problem_info['config_type'] == 'standard':
                # Use standard validator
                problem_state = {
                    'num_disks': problem_info['num_disks'],
                    'goal_peg': 2
                }
                reward, violations = standard_validator.validate_trace(response, problem_state)
                solved = reward >= 1.0
                validation_result = {
                    'reward': reward,
                    'violations': violations,
                    'solved': solved,
                }
            else:
                # Use non-standard validator
                validation_result = nonstandard_validator.validate(
                    response,
                    problem_info['initial_state'],
                    problem_info['goal_state'],
                    problem_info['num_disks']
                )
            
            results.append({
                'num_disks': problem_info['num_disks'],
                'trial': problem['trial'],
                'config_type': problem_info['config_type'],
                'problem_info': problem_info,
                'response': response,
                'validation': validation_result,
                'generation_time': generation_time,
                'solved': validation_result.get('solved', False),
            })
    
    return {
        'config_type': config_type,
        'results': results,
        'summary': compute_summary(results, disk_counts),
    }


def compute_summary(results: List[Dict], disk_counts: List[int]) -> Dict[str, Any]:
    """Compute summary statistics."""
    summary = {
        'total_problems': len(results),
        'total_solved': sum(1 for r in results if r['solved']),
        'overall_accuracy': sum(1 for r in results if r['solved']) / len(results) if results else 0,
        'avg_generation_time': sum(r['generation_time'] for r in results) / len(results) if results else 0,
        'per_disk': {},
    }
    
    for num_disks in disk_counts:
        disk_results = [r for r in results if r['num_disks'] == num_disks]
        if disk_results:
            solved = sum(1 for r in disk_results if r['solved'])
            total = len(disk_results)
            avg_violations = sum(
                r['validation'].get('violations', 0) for r in disk_results
            ) / total
            
            summary['per_disk'][num_disks] = {
                'total': total,
                'solved': solved,
                'accuracy': solved / total,
                'avg_violations': avg_violations,
                'avg_time': sum(r['generation_time'] for r in disk_results) / total,
            }
    
    return summary


def print_summary(evaluation_results: Dict[str, Any]) -> None:
    """Print evaluation summary."""
    summary = evaluation_results['summary']
    config_type = evaluation_results['config_type']
    
    print("\n" + "=" * 80)
    print(f"EVALUATION SUMMARY - {config_type.upper()}")
    print("=" * 80)
    print(f"Total Problems: {summary['total_problems']}")
    print(f"Total Solved: {summary['total_solved']}")
    print(f"Overall Accuracy: {summary['overall_accuracy']*100:.2f}%")
    print(f"Avg Generation Time: {summary['avg_generation_time']:.2f}s")
    
    print(f"\nPer-Disk Performance:")
    print(f"{'Disks':<8} {'Solved/Total':<15} {'Accuracy':<12} {'Avg Violations':<15} {'Avg Time':<10}")
    print("-" * 60)
    
    for num_disks in sorted(summary['per_disk'].keys()):
        stats = summary['per_disk'][num_disks]
        solved_str = f"{stats['solved']}/{stats['total']}"
        print(f"{num_disks:<8} {solved_str:<15} {stats['accuracy']*100:>6.1f}%     "
              f"{stats['avg_violations']:>8.2f}        {stats['avg_time']:>6.1f}s")
    
    print("=" * 80)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test trained GRPO model on Towers of Hanoi"
    )
    
    # Model
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model or LoRA adapter")
    parser.add_argument("--base_model_name", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                        help="Base model name (for LoRA)")
    
    # Test settings
    parser.add_argument("--min_disks", type=int, default=3)
    parser.add_argument("--max_disks", type=int, default=12)
    parser.add_argument("--trials_per_config", type=int, default=10,
                        help="Number of trials per disk count (k)")
    
    # Test mode
    parser.add_argument("--test_mode", type=str, default="both",
                        choices=["standard", "nonstandard", "both"],
                        help="Which test set(s) to run")
    
    # Generation settings
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--max_prompt_length", type=int, default=4096)
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for non-standard configurations")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./test_results")
    
    return parser.parse_args()


def main():
    """Main testing function."""
    args = parse_args()
    
    # Create config
    config = TestConfig(
        model_path=args.model_path,
        base_model_name=args.base_model_name,
        min_disks=args.min_disks,
        max_disks=args.max_disks,
        trials_per_config=args.trials_per_config,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_prompt_length=args.max_prompt_length,
        random_seed=args.seed,
        output_dir=args.output_dir,
        test_mode=args.test_mode,
    )
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(config.output_dir, f"test_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, "test_config.json"), 'w') as f:
        json.dump(vars(config), f, indent=2)
    
    print("=" * 80)
    print("Testing Trained GRPO Model on Towers of Hanoi")
    print("=" * 80)
    print(f"Model path: {config.model_path}")
    print(f"Test mode: {config.test_mode}")
    print(f"Disk range: {config.min_disks} to {config.max_disks}")
    print(f"Trials per config: {config.trials_per_config}")
    print(f"Random seed: {config.random_seed}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(config)
    
    all_results = {}
    
    # Run standard test set
    if config.test_mode in ["standard", "both"]:
        print("\n" + "=" * 80)
        print("STANDARD TEST SET")
        print("=" * 80)
        
        standard_test_set = generate_standard_test_set(
            config.min_disks,
            config.max_disks,
            config.trials_per_config
        )
        print(f"Generated {len(standard_test_set)} standard test problems")
        
        standard_results = evaluate_test_set(
            model, tokenizer, standard_test_set, config, "standard"
        )
        all_results['standard'] = standard_results
        print_summary(standard_results)
        
        # Save standard results
        with open(os.path.join(output_dir, "standard_results.json"), 'w') as f:
            # Don't save full responses to keep file manageable
            summary_results = {
                'config_type': standard_results['config_type'],
                'summary': standard_results['summary'],
                'results': [
                    {k: v for k, v in r.items() if k != 'response'}
                    for r in standard_results['results']
                ]
            }
            json.dump(summary_results, f, indent=2)
    
    # Run non-standard test set
    if config.test_mode in ["nonstandard", "both"]:
        print("\n" + "=" * 80)
        print("NON-STANDARD TEST SET")
        print("=" * 80)
        
        nonstandard_test_set = generate_nonstandard_test_set(
            config.min_disks,
            config.max_disks,
            config.trials_per_config,
            config.random_seed
        )
        print(f"Generated {len(nonstandard_test_set)} non-standard test problems")
        
        # Save the non-standard test configurations for reproducibility
        test_configs = [
            {
                'num_disks': t['problem_info']['num_disks'],
                'trial': t['trial'],
                'initial_state': t['problem_info']['initial_state'],
                'goal_state': t['problem_info']['goal_state'],
                'problem_seed': t['problem_info']['problem_seed'],
            }
            for t in nonstandard_test_set
        ]
        with open(os.path.join(output_dir, "nonstandard_configs.json"), 'w') as f:
            json.dump(test_configs, f, indent=2)
        
        nonstandard_results = evaluate_test_set(
            model, tokenizer, nonstandard_test_set, config, "nonstandard"
        )
        all_results['nonstandard'] = nonstandard_results
        print_summary(nonstandard_results)
        
        # Save non-standard results
        with open(os.path.join(output_dir, "nonstandard_results.json"), 'w') as f:
            summary_results = {
                'config_type': nonstandard_results['config_type'],
                'summary': nonstandard_results['summary'],
                'results': [
                    {k: v for k, v in r.items() if k != 'response'}
                    for r in nonstandard_results['results']
                ]
            }
            json.dump(summary_results, f, indent=2)
    
    # Print combined summary if both modes were run
    if config.test_mode == "both":
        print("\n" + "=" * 80)
        print("COMBINED SUMMARY")
        print("=" * 80)
        
        std_acc = all_results['standard']['summary']['overall_accuracy']
        nonstd_acc = all_results['nonstandard']['summary']['overall_accuracy']
        
        print(f"Standard Accuracy:     {std_acc*100:.2f}%")
        print(f"Non-Standard Accuracy: {nonstd_acc*100:.2f}%")
        print(f"Generalization Gap:    {(std_acc - nonstd_acc)*100:.2f}%")
        print("=" * 80)
    
    # Save combined results
    with open(os.path.join(output_dir, "all_results_summary.json"), 'w') as f:
        combined = {
            'config': vars(config),
            'standard_summary': all_results.get('standard', {}).get('summary'),
            'nonstandard_summary': all_results.get('nonstandard', {}).get('summary'),
        }
        json.dump(combined, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
