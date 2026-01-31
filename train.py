"""
GRPO Training Script for Towers of Hanoi with Constraint Loss

This script trains DeepSeek-R1-Distill-Qwen-32B on Towers of Hanoi using:
1. GRPO (Group Relative Policy Optimization) for policy improvement
2. Constraint loss to penalize rule violations during reasoning

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
from dataclasses import dataclass, field
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

from planning import TowersOfHanoiDataset, TowersOfHanoiValidator, TowersOfHanoiState
from config import TrainingConfig
from prompts import create_standard_prompt


# ============================================================================
# Violation Types for Constraint Loss
# ============================================================================

class ViolationType:
    """Enumeration of Towers of Hanoi rule violations."""
    DISK_NOT_ON_TOP = "disk_not_on_top"
    SOURCE_PEG_EMPTY = "source_peg_empty"
    LARGER_ON_SMALLER = "larger_on_smaller"
    INVALID_DISK_NUMBER = "invalid_disk_number"
    INVALID_PEG_NUMBER = "invalid_peg_number"
    INVALID_MOVE_FORMAT = "invalid_move_format"
    DISK_NOT_EXISTS = "disk_not_exists"


@dataclass
class MoveViolation:
    """Represents a constraint violation for a move."""
    violation_type: str
    move: Optional[List[int]]
    step_index: int
    description: str
    severity: float = 1.0  # Weight for the violation


# ============================================================================
# Prompt Creation (imported from prompts.py)
# ============================================================================
# create_standard_prompt is now imported from prompts.py to avoid duplication


# ============================================================================
# Move Parser and Constraint Checker
# ============================================================================

class MoveParser:
    """Parses moves from reasoning traces and checks constraints."""
    
    def __init__(self):
        # Pattern for extracting individual moves during reasoning
        # Matches patterns like: "Move disk 1 from peg 0 to peg 2"
        self.move_patterns = [
            # Pattern: "Move disk X from peg Y to peg Z"
            re.compile(
                r'[Mm]ove\s+[Dd]isk\s+(\d+)\s+from\s+[Pp]eg\s+(\d+)\s+to\s+[Pp]eg\s+(\d+)',
                re.IGNORECASE
            ),
            # Pattern: "[disk, from, to]" or "[X, Y, Z]"
            re.compile(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]'),
            # Pattern: "disk X: Y -> Z" or "disk X from Y to Z"
            re.compile(r'[Dd]isk\s+(\d+)[:\s]+(\d+)\s*(?:->|→|to)\s*(\d+)'),
        ]
        
        # Pattern for final moves array
        self.final_moves_pattern = re.compile(
            r'moves\s*=\s*(\[(?:\s*\[[^\]]+\]\s*,?\s*)+\])',
            re.DOTALL
        )
    
    def parse_final_moves(self, text: str) -> Optional[List[List[int]]]:
        """
        Parse the final moves array from the solution.
        
        Returns:
            List of [disk, from_peg, to_peg] or None if not found
        """
        matches = self.final_moves_pattern.findall(text)
        if not matches:
            return None
        
        moves_str = matches[-1].strip()
        try:
            moves = json.loads(moves_str)
            return moves
        except json.JSONDecodeError:
            return None
    
    def parse_intermediate_moves(self, text: str) -> List[Tuple[int, List[int]]]:
        """
        Parse all moves mentioned in the reasoning trace.
        
        Returns:
            List of (position_in_text, [disk, from_peg, to_peg])
        """
        moves = []
        
        for pattern in self.move_patterns:
            for match in pattern.finditer(text):
                try:
                    disk = int(match.group(1))
                    from_peg = int(match.group(2))
                    to_peg = int(match.group(3))
                    moves.append((match.start(), [disk, from_peg, to_peg]))
                except (ValueError, IndexError):
                    continue
        
        # Sort by position in text
        moves.sort(key=lambda x: x[0])
        return moves
    
    def extract_reasoning_moves(self, text: str) -> List[List[int]]:
        """
        Extract all moves from reasoning (thinking) section.
        
        Returns:
            Ordered list of moves found in reasoning
        """
        # Extract thinking section if present
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_match:
            reasoning = think_match.group(1)
        else:
            reasoning = text
        
        intermediate = self.parse_intermediate_moves(reasoning)
        return [move for _, move in intermediate]


class ConstraintChecker:
    """Checks Towers of Hanoi constraints and computes violation penalties."""
    
    def __init__(self, num_disks: int):
        self.num_disks = num_disks
        self.initial_state = TowersOfHanoiState(num_disks)
    
    def check_move(
        self, 
        move: List[int], 
        state: TowersOfHanoiState,
        step_index: int
    ) -> List[MoveViolation]:
        """
        Check a single move for constraint violations.
        
        Args:
            move: [disk, from_peg, to_peg]
            state: Current state before the move
            step_index: Index of this move in the sequence
            
        Returns:
            List of violations found
        """
        violations = []
        
        # Check move format
        if len(move) != 3:
            violations.append(MoveViolation(
                violation_type=ViolationType.INVALID_MOVE_FORMAT,
                move=move,
                step_index=step_index,
                description=f"Move must have 3 elements, got {len(move)}",
                severity=1.0
            ))
            return violations
        
        disk, from_peg, to_peg = move
        
        # Check disk number validity
        if not isinstance(disk, int) or disk < 1 or disk > self.num_disks:
            violations.append(MoveViolation(
                violation_type=ViolationType.INVALID_DISK_NUMBER,
                move=move,
                step_index=step_index,
                description=f"Invalid disk number {disk}, must be 1-{self.num_disks}",
                severity=1.0
            ))
            return violations
        
        # Check peg number validity
        if not isinstance(from_peg, int) or from_peg < 0 or from_peg > 2:
            violations.append(MoveViolation(
                violation_type=ViolationType.INVALID_PEG_NUMBER,
                move=move,
                step_index=step_index,
                description=f"Invalid source peg {from_peg}, must be 0-2",
                severity=1.0
            ))
            return violations
        
        if not isinstance(to_peg, int) or to_peg < 0 or to_peg > 2:
            violations.append(MoveViolation(
                violation_type=ViolationType.INVALID_PEG_NUMBER,
                move=move,
                step_index=step_index,
                description=f"Invalid destination peg {to_peg}, must be 0-2",
                severity=1.0
            ))
            return violations
        
        # Check source peg is not empty
        if len(state.pegs[from_peg]) == 0:
            violations.append(MoveViolation(
                violation_type=ViolationType.SOURCE_PEG_EMPTY,
                move=move,
                step_index=step_index,
                description=f"Source peg {from_peg} is empty",
                severity=1.0
            ))
            return violations
        
        # Check the disk is on top of the source peg
        top_disk = state.pegs[from_peg][-1]
        if top_disk != disk:
            violations.append(MoveViolation(
                violation_type=ViolationType.DISK_NOT_ON_TOP,
                move=move,
                step_index=step_index,
                description=f"Disk {disk} is not on top of peg {from_peg}, top disk is {top_disk}",
                severity=1.0
            ))
            return violations
        
        # Check destination peg constraint (larger disk on smaller)
        if len(state.pegs[to_peg]) > 0:
            top_dest = state.pegs[to_peg][-1]
            if disk > top_dest:
                violations.append(MoveViolation(
                    violation_type=ViolationType.LARGER_ON_SMALLER,
                    move=move,
                    step_index=step_index,
                    description=f"Cannot place disk {disk} on smaller disk {top_dest}",
                    severity=1.0
                ))
        
        return violations
    
    def check_move_sequence(
        self, 
        moves: List[List[int]]
    ) -> Tuple[List[MoveViolation], TowersOfHanoiState]:
        """
        Check a sequence of moves for constraint violations.
        
        Args:
            moves: List of [disk, from_peg, to_peg] moves
            
        Returns:
            Tuple of (all_violations, final_state)
        """
        state = self.initial_state.copy()
        all_violations = []
        
        for i, move in enumerate(moves):
            violations = self.check_move(move, state, i)
            all_violations.extend(violations)
            
            # Apply move if valid (for state tracking)
            if not violations and len(move) == 3:
                disk, from_peg, to_peg = move
                if (len(state.pegs[from_peg]) > 0 and 
                    state.pegs[from_peg][-1] == disk):
                    state.pegs[from_peg].pop()
                    state.pegs[to_peg].append(disk)
        
        return all_violations, state
    
    def compute_violation_score(self, violations: List[MoveViolation]) -> float:
        """
        Compute a normalized violation score.
        
        Returns:
            Score in [0, 1] where 0 = no violations, 1 = many violations
        """
        if not violations:
            return 0.0
        
        total_severity = sum(v.severity for v in violations)
        # Normalize by expected number of moves (2^n - 1)
        expected_moves = 2 ** self.num_disks - 1
        normalized = total_severity / max(expected_moves, 1)
        return min(normalized, 1.0)


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
# GRPO Training Dataset
# ============================================================================

class TOHDataset(Dataset):
    """Dataset for Towers of Hanoi GRPO training."""
    
    def __init__(
        self, 
        num_problems: int,
        min_disks: int = 3,
        max_disks: int = 5,
        disk_weights: Optional[Dict[int, float]] = None
    ):
        self.num_problems = num_problems
        self.min_disks = min_disks
        self.max_disks = max_disks
        self.disk_weights = disk_weights or {3: 0.4, 4: 0.35, 5: 0.25}
        
        # Pre-generate problems
        self.problems = self._generate_problems()
    
    def _generate_problems(self) -> List[Dict]:
        """Generate all problems for the dataset."""
        problems = []
        dataset = TowersOfHanoiDataset(self.min_disks, self.max_disks)
        
        # Sample disk counts based on weights
        disk_counts = list(range(self.min_disks, self.max_disks + 1))
        weights = [self.disk_weights.get(d, 1.0) for d in disk_counts]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        
        for _ in range(self.num_problems):
            num_disks = random.choices(disk_counts, weights=probs, k=1)[0]
            problem = dataset.generate_problem(num_disks=num_disks)
            system_prompt, user_prompt = create_standard_prompt(num_disks)
            
            problems.append({
                'num_disks': num_disks,
                'goal_peg': 2,
                'system_prompt': system_prompt,
                'user_prompt': user_prompt,
                'optimal_moves': 2 ** num_disks - 1,
            })
        
        return problems
    
    def __len__(self) -> int:
        return self.num_problems
    
    def __getitem__(self, idx: int) -> Dict:
        return self.problems[idx]


# ============================================================================
# GRPO Trainer with Constraint Loss
# ============================================================================

class GRPOTrainer:
    """
    GRPO Trainer with Constraint Loss for Towers of Hanoi.
    
    GRPO (Group Relative Policy Optimization) samples multiple responses
    per prompt and uses relative rewards within the group for policy updates.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        config: TrainingConfig,
        ref_model: Optional[nn.Module] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.ref_model = ref_model
        
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
        
        Returns:
            List of lists, where each inner list has num_samples responses
        """
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
    
    # Load model in bfloat16 (no quantization for training stability)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
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
    
    # Load reference model in bfloat16 (no quantization)
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
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
    print(f"Configuration:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Load reference model if needed
    ref_model = None
    if args.use_ref_model and config.kl_weight > 0:
        ref_model = load_reference_model(config)
    
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
