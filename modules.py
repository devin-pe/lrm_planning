"""
Latent Reasoning Module (LRM) for Planning with GRPO

This module implements a latent reasoning system for improving LLM planning 
capabilities through structured reasoning traces with multiple loss functions:

Architecture:
-------------
1. Base LLM (frozen): Provides initial representations (e.g., Llama-3-8B)
2. LatentModule: Encoder-Reasoner-Decoder bridge for planning
   - Encoder: Compresses LLM hidden states
   - Reasoner: Transformer layers for "System 2" reasoning
   - Decoder: Projects to vocabulary for reasoning tokens
   - Value Head: Predicts expected returns (actor-critic)

Loss Functions:
---------------
1. Task Loss (CE): Token-wise cross-entropy with expert solver traces
   - Provides structural template for correct reasoning
   - Supervised learning from algorithmic planners

2. GRPO Loss: Group Relative Policy Optimization
   - Samples multiple traces per problem (group)
   - Computes advantages relative to group mean reward
   - Includes KL divergence from reference policy
   - Prevents policy collapse and maintains diversity

3. Constraint Loss: Penalizes action precondition violations
   - Domain-specific validation of planning constraints
   - Quadratic penalty for violations
   - Ensures generated plans respect PDDL-like semantics

4. Entropy Loss: Encourages exploration
   - Prevents premature convergence
   - Maintains diversity in reasoning strategies

5. Value Loss (optional): Actor-critic value function
   - Reduces variance in advantage estimation
   - Improves sample efficiency

Training Workflow:
------------------
1. Initialize reference model (copy of current policy)
2. For each batch:
   a. Supervised phase: Train on expert traces
   b. GRPO phase: Sample K traces, validate, compute group advantages
   c. Combine losses and update policy
3. Periodically update reference model (every N steps)

Key Hyperparameters:
--------------------
- beta_kl: KL penalty weight (typical: 0.01)
- beta_entropy: Entropy regularization (typical: 0.01)
- beta_constraint: Constraint violation penalty (typical: 1.0)
- group_size: Number of samples per problem (typical: 4-8)
- update_interval: Steps between reference updates (typical: 500)

Usage:
------
See train_step_example() for a complete training loop example.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaConfig
from typing import List, Dict, Optional, Tuple


class PlanningValidator:
    """
    Utility class for validating planning traces and computing rewards.
    This should be customized based on your specific planning domain.
    """
    
    def __init__(self, domain_name="blocksworld"):
        self.domain_name = domain_name
    
    def validate_trace(self, reasoning_trace: str, problem_state: Dict) -> Tuple[float, int]:
        """
        Validate a reasoning trace against planning constraints.
        
        Args:
            reasoning_trace: Generated reasoning text
            problem_state: Dict with initial state, goal, and domain info
        
        Returns:
            reward: float - Reward score (e.g., 1.0 for valid, 0.0 for invalid)
            violations: int - Number of constraint violations
        """
        # This is a placeholder - implement domain-specific validation
        # Example checks:
        # 1. Action preconditions satisfied
        # 2. Effects correctly applied
        # 3. Goal reached
        # 4. No contradictory states
        
        violations = 0
        
        # Example: Check if actions are valid
        # violations += self._check_action_preconditions(reasoning_trace, problem_state)
        # violations += self._check_state_consistency(reasoning_trace)
        
        # Simple reward: binary based on whether there are violations
        reward = 1.0 if violations == 0 else 0.0
        
        # Could also use partial rewards based on progress to goal
        # reward = self._compute_progress_reward(reasoning_trace, problem_state)
        
        return reward, violations
    
    def batch_validate(self, traces: List[str], problem_states: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Validate a batch of reasoning traces.
        
        Returns:
            rewards: (batch_size,) tensor
            violations: (batch_size,) tensor
        """
        rewards = []
        violations = []
        
        for trace, state in zip(traces, problem_states):
            r, v = self.validate_trace(trace, state)
            rewards.append(r)
            violations.append(v)
        
        return torch.tensor(rewards), torch.tensor(violations)




class LatentModule(nn.Module):
    def __init__(self, hidden_dim=4096, latent_dim=1024, vocab_size=128256, use_value_head=False):
        super().__init__()
        self.use_value_head = use_value_head  # DISABLED: Set to False to remove value head
        
        # Encoder: Compresses LLM hidden states into the planning bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(), 
            nn.Linear(latent_dim, latent_dim)
        )
        
        # Latent Reasoner: the 'System 2' processor
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=8, batch_first=True)
        self.reasoner = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Decoder: Projects back to vocab for reasoning tokens (<think> tags)
        self.decoder = nn.Linear(latent_dim, vocab_size)
        
        # Value head for actor-critic GRPO
        if self.use_value_head:
            self.value_head = nn.Sequential(
                nn.Linear(latent_dim, latent_dim // 2),
                nn.ReLU(),
                nn.Linear(latent_dim // 2, 1)
            )

    def forward(self, x, return_value=False):
        """
        Args:
            x: (batch, seq, hidden_dim) - LLM hidden states
            return_value: bool - Whether to compute value predictions
        
        Returns:
            logits: (batch, seq, vocab) - Reasoning token logits
            refined: (batch, seq, latent_dim) - Latent representations
            values: (batch,) - Value predictions (if return_value=True)
        """
        latent = self.encoder(x)
        refined = self.reasoner(latent)
        logits = self.decoder(refined)
        
        values = None
        if return_value and self.use_value_head:
            # Pool the sequence (mean pooling) for value prediction
            pooled = refined.mean(dim=1)  # (batch, latent_dim)
            values = self.value_head(pooled).squeeze(-1)  # (batch,)
        
        return logits, refined, values
      
      
class PlanningLoss(nn.Module):
    def __init__(self, beta_kl=0.01, beta_entropy=0.01, beta_constraint=1.0, 
                 beta_value=0.5, gamma=0.99, use_value_function=False):
        super().__init__()
        self.beta_kl = beta_kl           # KL penalty coefficient
        self.beta_entropy = beta_entropy  # Entropy regularization
        self.beta_constraint = beta_constraint  # Constraint violation penalty
        self.beta_value = beta_value      # Value function loss weight
        self.gamma = gamma                # Discount factor for returns
        self.use_value_function = use_value_function  # DISABLED: Set to False to remove value loss

    def compute_grpo_loss(self, log_probs, ref_log_probs, advantages, old_log_probs=None):
        """
        Core GRPO loss with group-relative advantages and KL penalty.
        
        Args:
            log_probs: (G, seq) - Current policy log probabilities
            ref_log_probs: (G, seq) - Reference policy log probabilities
            advantages: (G,) - Group-relative advantages
            old_log_probs: (G, seq) - Old policy log probs (for PPO-style clipping, optional)
        """
        # Compute per-sequence log probability
        seq_log_probs = log_probs.sum(dim=-1)  # (G,)
        
        # Policy gradient term with group-relative advantages
        pg_loss = -(seq_log_probs * advantages).mean()
        
        # KL divergence from reference policy (prevents collapse)
        if ref_log_probs is not None:
            kl_div = (torch.exp(log_probs) * (log_probs - ref_log_probs)).sum(dim=-1).mean()
        else:
            kl_div = torch.tensor(0.0, device=log_probs.device)
        
        return pg_loss + self.beta_kl * kl_div
    
    def compute_constraint_loss(self, reasoning_tokens, constraint_violations):
        """
        Penalize action precondition violations in generated reasoning traces.
        
        Args:
            reasoning_tokens: (G, seq) - Generated reasoning token IDs
            constraint_violations: (G,) - Count or binary flag of violations per sequence
        """
        # Higher penalty for sequences with more violations
        # This encourages the model to respect planning constraints
        constraint_loss = (constraint_violations ** 2).mean()
        return self.beta_constraint * constraint_loss
    
    def compute_entropy_loss(self, logits):
        """
        Entropy regularization to encourage exploration.
        
        Args:
            logits: (G, seq, vocab) - Model logits
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        # Negative because we want to maximize entropy (minimize negative entropy)
        return -self.beta_entropy * entropy
    
    def compute_value_loss(self, value_preds, returns):
        """
        Value function loss for better advantage estimation (actor-critic).
        
        Args:
            value_preds: (G,) - Predicted values
            returns: (G,) - Actual returns (discounted rewards)
        """
        # MSE between predicted values and actual returns
        value_loss = F.mse_loss(value_preds, returns)
        return self.beta_value * value_loss

    def forward(self, logits, targets, rewards, ref_logits=None, 
                constraint_violations=None, value_preds=None):
        """
        Complete loss function for planning with GRPO.
        
        Args:
            logits: (G, seq, vocab) - Current policy logits
            targets: (G, seq) - Sampled reasoning tokens from current policy
            rewards: (G,) - Rewards from planning validator/solver
            ref_logits: (G, seq, vocab) - Reference policy logits (optional)
            constraint_violations: (G,) - Number of constraint violations (optional)
            value_preds: (G,) - Value function predictions (optional)
        
        Returns:
            dict with total loss and individual components
        """
        G, seq_len, vocab_size = logits.shape
        
        # 1. Task Loss (Supervised Learning from Expert Trajectories)
        # This provides the structure of correct reasoning traces
        ce_loss = F.cross_entropy(
            logits.view(-1, vocab_size), 
            targets.view(-1),
            reduction='mean'
        )
        
        # 2. GRPO Loss (Group Relative Policy Optimization)
        # Compute group-relative advantages
        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8
        advantages = (rewards - mean_reward) / std_reward
        
        # Get log probabilities for sampled tokens
        log_probs = F.log_softmax(logits, dim=-1)
        per_token_log_probs = torch.gather(
            log_probs, dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)  # (G, seq)
        
        # Reference log probs (from frozen reference model)
        ref_per_token_log_probs = None
        if ref_logits is not None:
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_per_token_log_probs = torch.gather(
                ref_log_probs, dim=-1, index=targets.unsqueeze(-1)
            ).squeeze(-1)
        
        grpo_loss = self.compute_grpo_loss(
            per_token_log_probs, 
            ref_per_token_log_probs, 
            advantages
        )
        
        # 3. Constraint Violation Penalty
        constraint_loss = torch.tensor(0.0, device=logits.device)
        if constraint_violations is not None:
            constraint_loss = self.compute_constraint_loss(targets, constraint_violations)
        
        # 4. Entropy Regularization
        entropy_loss = self.compute_entropy_loss(logits)
        
        # 5. Value Function Loss (if using actor-critic)
        value_loss = torch.tensor(0.0, device=logits.device)
        if self.use_value_function and value_preds is not None:
            # Compute returns (can be enhanced with GAE)
            returns = rewards  # Simple case; could use discounted returns
            value_loss = self.compute_value_loss(value_preds, returns)
        
        # Total loss
        total_loss = ce_loss + grpo_loss + constraint_loss + entropy_loss + value_loss
        
        return {
            'loss': total_loss,
            'ce_loss': ce_loss.item(),
            'grpo_loss': grpo_loss.item(),
            'constraint_loss': constraint_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'value_loss': value_loss.item(),
            'mean_reward': mean_reward.item(),
            'mean_advantage': advantages.mean().item()
        }
      

class LlamaPlanner(nn.Module):
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B", use_value_head=False):
        super().__init__()
        self.llama = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.latent_module = LatentModule(use_value_head=use_value_head)
        self.use_value_head = use_value_head 
        
        # Reference model for KL penalty (frozen copy of latent module)
        self.ref_latent_module = None
        
        # Freeze Llama to focus on the latent reasoning bridge
        for param in self.llama.parameters():
            param.requires_grad = False
    
    def init_reference_model(self):
        """Initialize reference model for KL divergence in GRPO."""
        self.ref_latent_module = LatentModule(use_value_head=False)
        self.ref_latent_module.load_state_dict(self.latent_module.state_dict(), strict=False)
        for param in self.ref_latent_module.parameters():
            param.requires_grad = False
        self.ref_latent_module.eval()
    
    def update_reference_model(self):
        """Periodically update reference model to current policy."""
        if self.ref_latent_module is None:
            self.init_reference_model()
        else:
            self.ref_latent_module.load_state_dict(self.latent_module.state_dict(), strict=False)

    def forward(self, input_ids, reasoning_targets=None, rewards=None, 
                constraint_violations=None, use_reference=True):
        """
        Forward pass with optional loss computation.
        
        Args:
            input_ids: (batch, seq) - Input token IDs
            reasoning_targets: (batch, seq) - Sampled reasoning tokens
            rewards: (batch,) - Rewards from planning validator
            constraint_violations: (batch,) - Constraint violation counts
            use_reference: bool - Whether to use reference model for KL
        
        Returns:
            reasoning_logits: (batch, seq, vocab)
            loss_dict: dict with loss components (if targets provided)
        """
        # Extract hidden states using output_hidden_states=True
        outputs = self.llama(input_ids, output_hidden_states=True)
        
        # Extract from Layer 24 (mid-deep) for high-level reasoning
        hidden_states = outputs.hidden_states[24]
        
        # Pass through latent module
        reasoning_logits, latent_vector, values = self.latent_module(
            hidden_states, 
            return_value=(self.use_value_head and reasoning_targets is not None)
        )
        
        loss_dict = None
        if reasoning_targets is not None and rewards is not None:
            # Get reference logits if requested
            ref_logits = None
            if use_reference and self.ref_latent_module is not None:
                with torch.no_grad():
                    ref_logits, _, _ = self.ref_latent_module(hidden_states, return_value=False)
            
            # Compute loss
            loss_fn = PlanningLoss(use_value_function=self.use_value_head)
            loss_dict = loss_fn(
                logits=reasoning_logits,
                targets=reasoning_targets,
                rewards=rewards,
                ref_logits=ref_logits,
                constraint_violations=constraint_violations,
                value_preds=values
            )
            
        return reasoning_logits, loss_dict, values


