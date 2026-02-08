"""
GRPO Training Script for Towers of Hanoi with Constraint Loss

This script trains DeepSeek-R1-Distill-Qwen on Towers of Hanoi using:
1. GRPO (Group Relative Policy Optimization) for policy improvement
2. Constraint loss to penalize rule violations during reasoning
3. vLLM for fast rollout generation (10-20x speedup)

The constraint loss ensures:
- The chosen disk to move is on top of its peg
- The destination peg can receive the disk (empty or has larger disk on top)
- Move coordinates are valid (pegs 0-2, valid disk numbers)
"""

import os
import re
import json
import math
import time
import random
import argparse
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
from tqdm import tqdm

from planning import (
    TowersOfHanoiDataset, 
    TowersOfHanoiValidator, 
    TowersOfHanoiState,
    ViolationType,
    MoveViolation,
    MoveParser,
    ConstraintChecker,
    TOHDataset,
)
from config import TrainingConfig
from prompts import create_standard_prompt

# vLLM import - optional, will be imported when needed
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False


# ============================================================================
# vLLM Rollout Engine for Fast Generation
# ============================================================================

class VLLMRolloutEngine:
    """
    Fast rollout engine using vLLM for 10-20x faster generation.
    
    This class wraps vLLM to provide fast parallel generation of responses
    for GRPO training. The weights can be synced from the training model
    periodically.
    
    When using with training (2 GPUs), vLLM uses GPU 1 while training uses GPU 0.
    """
    
    def __init__(
        self,
        model_name: str,
        tokenizer: AutoTokenizer,
        max_new_tokens: int = 4096,
        temperature: float = 1.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        cache_dir: Optional[str] = None,
        vllm_gpu_id: int = 1,  # Use GPU 1 (GPU 0 is for training)
    ):
        if not VLLM_AVAILABLE:
            raise ImportError(
                "vLLM is not installed. Install it with: pip install vllm"
            )
        
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # Initialize vLLM engine on specified GPU
        print(f"Initializing vLLM engine with {model_name} on GPU {vllm_gpu_id}...")
        
        # Set environment to restrict vLLM to specific GPU
        import os
        old_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(vllm_gpu_id)
        
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=32768,  # Max context for DeepSeek models
            download_dir=cache_dir,
            dtype="bfloat16",
        )
        
        # Restore CUDA_VISIBLE_DEVICES
        if old_cuda_visible is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible
        else:
            del os.environ['CUDA_VISIBLE_DEVICES']
        
        # Default sampling parameters
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=0.95,
        )
        print("vLLM engine initialized successfully!")
    
    def generate(
        self,
        prompts: List[Dict],
        num_samples: int,
    ) -> List[List[str]]:
        """
        Generate multiple responses per prompt using vLLM.
        
        Args:
            prompts: List of prompt dictionaries with 'system_prompt' and 'user_prompt'
            num_samples: Number of responses to generate per prompt
            
        Returns:
            List of lists, where each inner list has num_samples responses
        """
        # Format all prompts
        formatted_prompts = []
        for prompt_data in prompts:
            messages = [
                {"role": "system", "content": prompt_data['system_prompt']},
                {"role": "user", "content": prompt_data['user_prompt']}
            ]
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            # Repeat for num_samples
            formatted_prompts.extend([formatted] * num_samples)
        
        # Generate all at once (vLLM batches internally)
        outputs = self.llm.generate(formatted_prompts, self.sampling_params)
        
        # Organize results
        all_responses = []
        idx = 0
        for _ in prompts:
            responses = []
            for _ in range(num_samples):
                response_text = outputs[idx].outputs[0].text
                responses.append(response_text)
                idx += 1
            all_responses.append(responses)
        
        return all_responses
    
    def update_weights(self, model: nn.Module):
        """
        Sync weights from training model to vLLM engine.
        
        Note: This is a simplified version. For production, you'd want to use
        vLLM's weight update API or save/reload the model.
        """
        # For LoRA models, we need to merge and reload
        # This is expensive, so should be done infrequently
        pass  # TODO: Implement efficient weight sync


# ============================================================================
# Reward Computation for GRPO
# ============================================================================

class RewardComputer:
    """Computes rewards for GRPO training."""
    
    def __init__(self, validator: TowersOfHanoiValidator):
        self.validator = validator
        self.move_parser = MoveParser()
    
    def compute_reward(
        self, 
        response: str, 
        num_disks: int,
        goal_peg: int = 2
    ) -> Dict[str, Any]:
        """
        Compute reward and constraint violations for a response.
        
        Returns:
            Dict with reward, violations, parsed_moves, etc.
        """
        problem_state = {
            'num_disks': num_disks,
            'goal_peg': goal_peg
        }
        
        # Get base reward from validator
        reward, violation_count = self.validator.validate_trace(response, problem_state)
        
        # Parse moves for constraint checking
        constraint_checker = ConstraintChecker(num_disks)
        final_moves = self.move_parser.parse_final_moves(response)
        
        if final_moves is not None:
            violations, final_state = constraint_checker.check_move_sequence(final_moves)
            violation_score = constraint_checker.compute_violation_score(violations)
            solved = final_state.is_goal(goal_peg)
        else:
            violations = []
            violation_score = 1.0  # Max penalty for no parseable moves
            solved = False
        
        return {
            'reward': reward,
            'violation_count': violation_count,
            'violations': violations,
            'violation_score': violation_score,
            'solved': solved,
            'parsed_moves': final_moves,
            'num_moves': len(final_moves) if final_moves else 0,
            'optimal_moves': 2 ** num_disks - 1,
        }


# ============================================================================
# GRPO Trainer with Constraint Loss
# ============================================================================

class GRPOTrainer:
    """
    GRPO Trainer with Constraint Loss for Towers of Hanoi.
    
    GRPO (Group Relative Policy Optimization) samples multiple responses
    per prompt and uses relative rewards within the group for policy updates.
    
    Supports optional vLLM rollout engine for 10-20x faster generation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        config: TrainingConfig,
        ref_model: Optional[nn.Module] = None,
        vllm_engine: Optional[VLLMRolloutEngine] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.ref_model = ref_model
        self.vllm_engine = vllm_engine
        
        self.validator = TowersOfHanoiValidator()
        self.reward_computer = RewardComputer(self.validator)
        self.move_parser = MoveParser()
        
        # Setup accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision="bf16",
        )
        
        # Prepare model
        self.model = self.accelerator.prepare(self.model)
        if self.ref_model is not None:
            self.ref_model = self.accelerator.prepare(self.ref_model)
            self.ref_model.eval()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )
        self.optimizer = self.accelerator.prepare(self.optimizer)
        
        # Tracking
        self.global_step = 0
        self.train_stats = []
        
        # Create outputs directory for saving model responses
        self.outputs_dir = None  # Will be set when training starts
    
    def generate_responses(
        self, 
        prompts: List[Dict],
        num_samples: int
    ) -> List[List[str]]:
        """
        Generate multiple responses per prompt.
        
        Uses vLLM engine if available for 10-20x faster generation,
        otherwise falls back to standard HuggingFace generate().
        
        Returns:
            List of lists, where each inner list has num_samples responses
        """
        # Use vLLM if available (10-20x faster)
        if self.vllm_engine is not None:
            return self.vllm_engine.generate(prompts, num_samples)
        
        # Fallback to standard HuggingFace generation
        self.model.eval()
        all_responses = []
        
        with torch.no_grad():
            for prompt_data in prompts:
                system_prompt = prompt_data['system_prompt']
                user_prompt = prompt_data['user_prompt']
                
                # Format as chat messages
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                # Apply chat template
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                inputs = self.tokenizer(
                    formatted,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_prompt_length
                ).to(self.accelerator.device)
                
                # Generate num_samples responses
                responses = []
                for _ in range(num_samples):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config.max_new_tokens,
                        temperature=self.config.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    
                    # Decode response (excluding input)
                    response = self.tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )
                    responses.append(response)
                
                all_responses.append(responses)
        
        return all_responses
    
    def compute_grpo_loss(
        self,
        prompt_data: Dict,
        responses: List[str],
        rewards: List[float],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute GRPO loss for a group of responses.
        
        GRPO uses relative rewards within the group:
        advantage_i = (reward_i - mean(rewards)) / std(rewards)
        
        Returns:
            Tuple of (loss, stats_dict)
        """
        self.model.train()
        
        # Compute advantages (relative rewards)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std() + 1e-8
        advantages = (rewards_tensor - mean_reward) / std_reward
        
        total_loss = torch.tensor(0.0, device=self.accelerator.device)
        policy_losses = []
        
        system_prompt = prompt_data['system_prompt']
        user_prompt = prompt_data['user_prompt']
        
        for response, advantage in zip(responses, advantages):
            # Format full sequence
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response}
            ]
            
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
            
            # Tokenize full sequence (prompt + response)
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_seq_length
            ).to(self.accelerator.device)
            
            # Get model logits
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            
            # Policy gradient loss: -advantage * log_prob
            # Using cross-entropy loss as proxy for log_prob
            policy_loss = -advantage.to(self.accelerator.device) * outputs.loss
            policy_losses.append(policy_loss)
            total_loss = total_loss + policy_loss
        
        # Average across group
        grpo_loss = total_loss / len(responses)
        
        stats = {
            'grpo_loss': grpo_loss.item(),
            'mean_reward': mean_reward.item(),
            'std_reward': std_reward.item(),
            'max_reward': rewards_tensor.max().item(),
            'min_reward': rewards_tensor.min().item(),
        }
        
        return grpo_loss, stats
    
    def compute_constraint_loss(
        self,
        prompt_data: Dict,
        responses: List[str],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute constraint loss based on rule violations.
        
        This loss penalizes the model for generating moves that violate
        Towers of Hanoi rules.
        
        Returns:
            Tuple of (loss, stats_dict)
        """
        num_disks = prompt_data['num_disks']
        constraint_checker = ConstraintChecker(num_disks)
        
        total_violations = 0
        violation_scores = []
        violation_type_counts = {v: 0 for v in [
            ViolationType.DISK_NOT_ON_TOP,
            ViolationType.SOURCE_PEG_EMPTY,
            ViolationType.LARGER_ON_SMALLER,
            ViolationType.INVALID_DISK_NUMBER,
            ViolationType.INVALID_PEG_NUMBER,
            ViolationType.INVALID_MOVE_FORMAT,
        ]}
        
        for response in responses:
            moves = self.move_parser.parse_final_moves(response)
            if moves is not None:
                violations, _ = constraint_checker.check_move_sequence(moves)
                violation_score = constraint_checker.compute_violation_score(violations)
                violation_scores.append(violation_score)
                total_violations += len(violations)
                
                for v in violations:
                    if v.violation_type in violation_type_counts:
                        violation_type_counts[v.violation_type] += 1
            else:
                violation_scores.append(1.0)  # Max penalty for unparseable
        
        # Convert to tensor loss
        constraint_loss = torch.tensor(
            sum(violation_scores) / len(violation_scores),
            device=self.accelerator.device,
            requires_grad=True
        )
        
        stats = {
            'constraint_loss': constraint_loss.item(),
            'total_violations': total_violations,
            'avg_violation_score': sum(violation_scores) / len(violation_scores),
            'violation_types': violation_type_counts,
        }
        
        return constraint_loss, stats
    
    def compute_kl_loss(
        self,
        prompt_data: Dict,
        responses: List[str],
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute KL divergence loss against reference model.
        
        Returns:
            Tuple of (loss, stats_dict)
        """
        if self.ref_model is None:
            return torch.tensor(0.0, device=self.accelerator.device), {'kl_loss': 0.0}
        
        self.model.eval()
        kl_losses = []
        
        system_prompt = prompt_data['system_prompt']
        user_prompt = prompt_data['user_prompt']
        
        with torch.no_grad():
            for response in responses:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": response}
                ]
                
                full_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                )
                
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_seq_length
                ).to(self.accelerator.device)
                
                # Get logits from both models
                with torch.no_grad():
                    ref_outputs = self.ref_model(**inputs)
                    ref_logits = ref_outputs.logits
                
                policy_outputs = self.model(**inputs)
                policy_logits = policy_outputs.logits
                
                # Compute KL divergence
                ref_probs = F.softmax(ref_logits, dim=-1)
                policy_log_probs = F.log_softmax(policy_logits, dim=-1)
                
                kl = F.kl_div(policy_log_probs, ref_probs, reduction='batchmean')
                kl_losses.append(kl)
        
        kl_loss = torch.stack(kl_losses).mean() if kl_losses else torch.tensor(0.0)
        
        return kl_loss, {'kl_loss': kl_loss.item()}
    
    def train_step(
        self,
        batch: List[Dict],
    ) -> Dict[str, float]:
        """
        Execute one training step with GRPO + Constraint Loss.
        
        Args:
            batch: List of prompt dictionaries
            
        Returns:
            Dictionary of training statistics
        """
        all_stats = {
            'loss': 0.0,
            'grpo_loss': 0.0,
            'constraint_loss': 0.0,
            'kl_loss': 0.0,
            'mean_reward': 0.0,
            'total_violations': 0,
            'solved_rate': 0.0,
        }
        
        num_solved = 0
        total_samples = 0
        
        for batch_idx, prompt_data in enumerate(batch):
            # Generate group of responses
            responses = self.generate_responses(
                [prompt_data], 
                self.config.num_samples_per_prompt
            )[0]
            
            # Compute rewards for each response
            rewards = []
            rewards_data = []  # Store full reward data for saving
            for response in responses:
                result = self.reward_computer.compute_reward(
                    response,
                    prompt_data['num_disks'],
                    prompt_data['goal_peg']
                )
                rewards.append(result['reward'])
                rewards_data.append(result)
                if result['solved']:
                    num_solved += 1
                total_samples += 1
            
            # Save outputs periodically for debugging
            if self.global_step % self.config.logging_steps == 0:
                self._save_outputs(prompt_data, responses, rewards_data, 
                                 self.global_step, batch_idx)
            
            # Compute losses
            grpo_loss, grpo_stats = self.compute_grpo_loss(
                prompt_data, responses, rewards
            )
            
            constraint_loss, constraint_stats = self.compute_constraint_loss(
                prompt_data, responses
            )
            
            kl_loss, kl_stats = self.compute_kl_loss(prompt_data, responses)
            
            # Combined loss
            total_loss = (
                self.config.grpo_weight * grpo_loss +
                self.config.constraint_weight * constraint_loss +
                self.config.kl_weight * kl_loss
            )
            
            # Backward pass
            self.accelerator.backward(total_loss)
            
            # Update stats
            all_stats['loss'] += total_loss.item()
            all_stats['grpo_loss'] += grpo_stats['grpo_loss']
            all_stats['constraint_loss'] += constraint_stats['constraint_loss']
            all_stats['kl_loss'] += kl_stats['kl_loss']
            all_stats['mean_reward'] += grpo_stats['mean_reward']
            all_stats['total_violations'] += constraint_stats['total_violations']
        
        # Gradient clipping
        if self.config.max_grad_norm > 0:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Average stats
        batch_size = len(batch)
        for key in ['loss', 'grpo_loss', 'constraint_loss', 'kl_loss', 'mean_reward']:
            all_stats[key] /= batch_size
        
        all_stats['solved_rate'] = num_solved / total_samples if total_samples > 0 else 0.0
        
        self.global_step += 1
        return all_stats
    
    def train(
        self,
        train_dataset: TOHDataset,
        eval_dataset: Optional[TOHDataset] = None,
        num_epochs: int = 3,
    ) -> Dict[str, List[float]]:
        """
        Run full training loop.
        
        Returns:
            Dictionary of training history
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=lambda x: x,  # Keep as list of dicts
        )
        
        num_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(num_steps * self.config.warmup_ratio),
            num_training_steps=num_steps,
        )
        
        # Create outputs directory for saving model responses
        self.outputs_dir = os.path.join(self.config.output_dir, "outputs")
        os.makedirs(self.outputs_dir, exist_ok=True)
        print(f"Model outputs will be saved to: {self.outputs_dir}")
        
        history = {
            'loss': [],
            'grpo_loss': [],
            'constraint_loss': [],
            'mean_reward': [],
            'solved_rate': [],
        }
        
        print("=" * 80)
        print("Starting GRPO Training with Constraint Loss")
        print("=" * 80)
        print(f"Model: {self.config.model_name}")
        print(f"Num training problems: {len(train_dataset)}")
        print(f"Epochs: {num_epochs}")
        print(f"Samples per prompt: {self.config.num_samples_per_prompt}")
        print(f"GRPO weight: {self.config.grpo_weight}")
        print(f"Constraint weight: {self.config.constraint_weight}")
        print(f"KL weight: {self.config.kl_weight}")
        print("=" * 80)
        
        for epoch in range(num_epochs):
            epoch_stats = {k: [] for k in history.keys()}
            
            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                disable=not self.accelerator.is_main_process
            )
            
            for batch in pbar:
                stats = self.train_step(batch)
                scheduler.step()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{stats['loss']:.4f}",
                    'reward': f"{stats['mean_reward']:.3f}",
                    'solved': f"{stats['solved_rate']*100:.1f}%",
                })
                
                # Record stats
                for key in history.keys():
                    if key in stats:
                        epoch_stats[key].append(stats[key])
                        history[key].append(stats[key])
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self._log_stats(stats)
                
                # Saving
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint(epoch)
            
            # Epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            for key, values in epoch_stats.items():
                if values:
                    print(f"  {key}: {sum(values)/len(values):.4f}")
            
            # Evaluation
            if eval_dataset is not None:
                eval_stats = self.evaluate(eval_dataset)
                print(f"\nEvaluation Results:")
                for key, value in eval_stats.items():
                    print(f"  {key}: {value:.4f}")
        
        # Final save
        self._save_checkpoint(num_epochs, final=True)
        
        return history
    
    def evaluate(self, eval_dataset: TOHDataset) -> Dict[str, float]:
        """Evaluate on a held-out dataset."""
        self.model.eval()
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x,
        )
        
        total_reward = 0.0
        total_violations = 0
        total_solved = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                for prompt_data in batch:
                    # Generate single response for evaluation
                    responses = self.generate_responses([prompt_data], 1)[0]
                    response = responses[0]
                    
                    result = self.reward_computer.compute_reward(
                        response,
                        prompt_data['num_disks'],
                        prompt_data['goal_peg']
                    )
                    
                    total_reward += result['reward']
                    total_violations += result['violation_count']
                    if result['solved']:
                        total_solved += 1
                    total_samples += 1
        
        return {
            'eval_reward': total_reward / total_samples,
            'eval_violations': total_violations / total_samples,
            'eval_solved_rate': total_solved / total_samples,
        }
    
    def _log_stats(self, stats: Dict[str, float]) -> None:
        """Log training statistics."""
        if self.accelerator.is_main_process:
            log_str = f"Step {self.global_step}: "
            log_str += " | ".join([f"{k}: {v:.4f}" for k, v in stats.items() 
                                   if isinstance(v, (int, float))])
            print(log_str)
    
    def _save_outputs(self, prompt_data: Dict, responses: List[str], 
                      rewards_data: List[Dict], step: int, batch_idx: int) -> None:
        """Save model outputs to files for debugging."""
        if not self.accelerator.is_main_process or self.outputs_dir is None:
            return
        
        # Create step directory
        step_dir = os.path.join(self.outputs_dir, f"step_{step:06d}")
        os.makedirs(step_dir, exist_ok=True)
        
        # Save each response in the group
        for i, (response, reward_data) in enumerate(zip(responses, rewards_data)):
            filename = f"batch{batch_idx:04d}_sample{i:02d}_disks{prompt_data['num_disks']}.txt"
            filepath = os.path.join(step_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write(f"Step: {step} | Batch: {batch_idx} | Sample: {i+1}\n")
                f.write(f"Num Disks: {prompt_data['num_disks']}\n")
                f.write(f"Reward: {reward_data['reward']:.4f}\n")
                f.write(f"Solved: {reward_data['solved']}\n")
                f.write(f"Violations: {reward_data['violation_count']}\n")
                f.write(f"Violation Score: {reward_data['violation_score']:.4f}\n")
                if reward_data['parsed_moves']:
                    f.write(f"Num Moves: {reward_data['num_moves']} (optimal: {reward_data['optimal_moves']})\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("SYSTEM PROMPT:\n")
                f.write("-" * 80 + "\n")
                f.write(prompt_data['system_prompt'])
                f.write("\n\n")
                
                f.write("USER PROMPT:\n")
                f.write("-" * 80 + "\n")
                f.write(prompt_data['user_prompt'])
                f.write("\n\n")
                
                f.write("MODEL RESPONSE:\n")
                f.write("-" * 80 + "\n")
                f.write(response)
                f.write("\n\n")
                
                if reward_data['violations']:
                    f.write("VIOLATIONS DETECTED:\n")
                    f.write("-" * 80 + "\n")
                    for v in reward_data['violations']:
                        f.write(f"  Step {v.step_index}: {v.violation_type}\n")
                        f.write(f"    Move: {v.move}\n")
                        f.write(f"    Description: {v.description}\n")
                        f.write(f"    Severity: {v.severity}\n\n")
        
        # Save summary for this step
        summary_file = os.path.join(step_dir, "summary.json")
        summary = {
            'step': step,
            'batch_idx': batch_idx,
            'num_disks': prompt_data['num_disks'],
            'num_samples': len(responses),
            'rewards': [r['reward'] for r in rewards_data],
            'solved': [r['solved'] for r in rewards_data],
            'violations': [r['violation_count'] for r in rewards_data],
            'mean_reward': sum(r['reward'] for r in rewards_data) / len(rewards_data),
            'solved_rate': sum(1 for r in rewards_data if r['solved']) / len(rewards_data),
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _save_checkpoint(self, epoch: int, final: bool = False) -> None:
        """Save model checkpoint."""
        if self.accelerator.is_main_process:
            output_dir = os.path.join(
                self.config.output_dir,
                f"checkpoint-{self.global_step}" if not final else "final"
            )
            os.makedirs(output_dir, exist_ok=True)
            
            # Unwrap model before saving
            unwrapped = self.accelerator.unwrap_model(self.model)
            unwrapped.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Save config
            with open(os.path.join(output_dir, "training_config.json"), 'w') as f:
                json.dump(vars(self.config), f, indent=2, default=str)
            
            print(f"Checkpoint saved to: {output_dir}")


# ============================================================================
# Model Loading
# ============================================================================

def load_model_and_tokenizer(config: TrainingConfig):
    """Load model and tokenizer with LoRA for efficient fine-tuning."""
    print(f"Loading model: {config.model_name}")
    
    # Get cache directory from environment (set in main)
    cache_dir = os.environ.get('TRANSFORMERS_CACHE', None)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        model_max_length=config.max_seq_length,
        cache_dir=cache_dir,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in bfloat16 on GPU 0 (GPU 1 reserved for vLLM)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map={"":  0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        cache_dir=cache_dir,
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Apply LoRA
    if config.use_lora:
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, tokenizer


def load_reference_model(config: TrainingConfig):
    """Load frozen reference model for KL divergence computation."""
    print("Loading reference model...")
    
    # Get cache directory from environment (set in main)
    cache_dir = os.environ.get('TRANSFORMERS_CACHE', None)
    
    # Load reference model in bfloat16 on GPU 0
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map={"":  0},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        cache_dir=cache_dir,
    )
    
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    return ref_model


# ============================================================================
# Main
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GRPO Training for Towers of Hanoi with Constraint Loss"
    )
    
    # Model
    parser.add_argument("--model_name", type=str, 
                        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
    
    # Problem settings
    parser.add_argument("--min_disks", type=int, default=3)
    parser.add_argument("--max_disks", type=int, default=5)
    parser.add_argument("--num_train_problems", type=int, default=1000)
    parser.add_argument("--num_eval_problems", type=int, default=100)
    
    # GRPO settings
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=16384)
    
    # Training
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=3)
    
    # Loss weights
    parser.add_argument("--grpo_weight", type=float, default=1.0)
    parser.add_argument("--constraint_weight", type=float, default=0.5)
    parser.add_argument("--kl_weight", type=float, default=0.01)
    
    # LoRA
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", action="store_false", dest="use_lora")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    # Sequence length settings (DeepSeek-R1-Distill-Qwen-32B supports 32,768 max)
    parser.add_argument("--max_prompt_length", type=int, default=4096,
                        help="Max tokens for input prompt (system + user)")
    parser.add_argument("--max_seq_length", type=int, default=16384,
                        help="Max tokens for full sequence (prompt + response)")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./training_outputs")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=10)
    
    # Cache directory for model downloads
    parser.add_argument("--cache_dir", type=str, default="/scratch-shared/dpereira",
                        help="Directory to cache downloaded models (HuggingFace cache)")
    
    # Reference model
    parser.add_argument("--use_ref_model", action="store_true", default=False)
    
    # vLLM for fast generation
    parser.add_argument("--use_vllm", action="store_true", default=False,
                        help="Use vLLM for 10-20x faster generation (requires vllm package)")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.85,
                        help="GPU memory utilization for vLLM (0.0-1.0)")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set up cache directory for model downloads
    cache_dir = args.cache_dir
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        
        # Set all cache environment variables to avoid disk quota issues
        # HuggingFace cache
        os.environ['HF_HOME'] = os.path.join(cache_dir, 'hf_cache')
        os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, 'transformers')
        os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, 'datasets')
        
        # PyTorch/Triton compilation caches
        os.environ['TORCH_HOME'] = os.path.join(cache_dir, 'torch_cache')
        os.environ['TRITON_CACHE_DIR'] = os.path.join(cache_dir, 'triton_cache')
        
        # General XDG cache
        os.environ['XDG_CACHE_HOME'] = os.path.join(cache_dir, 'cache')
        
        # Create all cache directories
        for env_var in ['HF_HOME', 'TRANSFORMERS_CACHE', 'HF_DATASETS_CACHE', 
                        'TORCH_HOME', 'TRITON_CACHE_DIR', 'XDG_CACHE_HOME']:
            os.makedirs(os.environ[env_var], exist_ok=True)
        
        print(f"Model cache directory: {cache_dir}")
        print(f"All cache subdirectories created")
    
    # Create config
    config = TrainingConfig(
        model_name=args.model_name,
        min_disks=args.min_disks,
        max_disks=args.max_disks,
        num_samples_per_prompt=args.num_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        grpo_weight=args.grpo_weight,
        constraint_weight=args.constraint_weight,
        kl_weight=args.kl_weight,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        max_prompt_length=args.max_prompt_length,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        num_train_problems=args.num_train_problems,
        num_eval_problems=args.num_eval_problems,
    )
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config.output_dir = os.path.join(config.output_dir, f"run_{timestamp}")
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(config.output_dir, "config.json"), 'w') as f:
        json.dump(vars(config), f, indent=2, default=str)
    
    print("=" * 80)
    print("GRPO Training for Towers of Hanoi with Constraint Loss")
    print("=" * 80)
    print(f"Output directory: {config.output_dir}")
    print(f"Using vLLM: {args.use_vllm}")
    print(f"Configuration:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # Load model for training (with LoRA)
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Load reference model if needed
    ref_model = None
    if args.use_ref_model and config.kl_weight > 0:
        ref_model = load_reference_model(config)
    
    # Initialize vLLM engine for fast generation if requested
    vllm_engine = None
    if args.use_vllm:
        if not VLLM_AVAILABLE:
            print("WARNING: vLLM requested but not available. Falling back to HuggingFace generate().")
            print("Install vLLM with: pip install vllm")
        else:
            print("\nInitializing vLLM rollout engine for fast generation...")
            vllm_engine = VLLMRolloutEngine(
                model_name=config.model_name,
                tokenizer=tokenizer,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                gpu_memory_utilization=args.vllm_gpu_memory_utilization,
                cache_dir=args.cache_dir,
            )
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = TOHDataset(
        num_problems=config.num_train_problems,
        min_disks=config.min_disks,
        max_disks=config.max_disks,
    )
    
    eval_dataset = TOHDataset(
        num_problems=config.num_eval_problems,
        min_disks=config.min_disks,
        max_disks=config.max_disks,
    )
    
    print(f"Train dataset: {len(train_dataset)} problems")
    print(f"Eval dataset: {len(eval_dataset)} problems")
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        ref_model=ref_model,
        vllm_engine=vllm_engine,
    )
    
    # Train
    print("\nStarting training...")
    start_time = time.time()
    
    history = trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=config.num_epochs,
    )
    
    total_time = time.time() - start_time
    
    # Save training history
    with open(os.path.join(config.output_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Final model saved to: {config.output_dir}/final")
    print(f"Training history saved to: {config.output_dir}/training_history.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
