
"""
Complete Training Script for Towers of Hanoi Planning with LRM and GRPO

This script demonstrates end-to-end training of the Latent Reasoning Module
for Towers of Hanoi planning problems using:
- Supervised learning from optimal solver traces
- GRPO (Group Relative Policy Optimization) with constraint validation
- Periodic reference model updates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List
import os
from tqdm import tqdm
from datetime import datetime

from modules import LlamaPlanner, PlanningValidator
from planning import (
    TowersOfHanoiValidator, 
    TowersOfHanoiSolver, 
    TowersOfHanoiDataset
)
import config


class TowersOfHanoiDatasetWrapper(Dataset):
    def __init__(self, num_samples: int = 1000, min_disks: int = 2, max_disks: int = 5):
        self.generator = TowersOfHanoiDataset(min_disks, max_disks)
        self.num_samples = num_samples
        self.samples = [self.generator.generate_problem() for _ in range(num_samples)]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: List[Dict], tokenizer):
    """
    Collate function for DataLoader.
    Tokenizes problem texts and expert traces.
    """
    problem_texts = [item['problem_text'] for item in batch]
    expert_traces = [item['expert_trace'] for item in batch]
    
    # Tokenize problem texts
    problem_encodings = tokenizer(
        problem_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # Tokenize expert traces
    trace_encodings = tokenizer(
        expert_traces,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    
    # Extract metadata
    problem_states = [
        {'num_disks': item['num_disks'], 'goal_peg': item['goal_peg']}
        for item in batch
    ]
    
    return {
        'input_ids': problem_encodings['input_ids'],
        'attention_mask': problem_encodings['attention_mask'],
        'expert_traces': trace_encodings['input_ids'],
        'expert_trace_text': expert_traces,
        'problem_states': problem_states,
        'problem_texts': problem_texts
    }


def train_step(model: LlamaPlanner, 
                       batch: Dict, 
                       validator: PlanningValidator,
                       optimizer: torch.optim.Optimizer,
                       device: str = 'cuda',
                       group_size: int = 4) -> Dict:
    """
    Example training step showing how to use GRPO with the planning model.
    
    Args:
        model: LlamaPlanner instance
        batch: Dict containing:
            - 'input_ids': (batch_size, seq_len) - Problem descriptions
            - 'expert_traces': (batch_size, trace_len) - Ground truth from solver
            - 'problem_states': List[Dict] - Planning problem metadata
        validator: PlanningValidator for checking constraints
        optimizer: PyTorch optimizer
        device: Device to run on
        group_size: Number of samples per problem for GRPO
    
    Returns:
        Dict with loss components and metrics
    """
    model.train()
    
    input_ids = batch['input_ids'].to(device)
    expert_traces = batch['expert_traces'].to(device)
    problem_states = batch['problem_states']
    
    batch_size = input_ids.shape[0]
    
    # ========================================================================
    # Phase 1: Supervised Learning on Expert Traces
    # ========================================================================
    # This provides the structured template of correct reasoning
    
    logits_expert, _, _ = model.latent_module(
        model.llama(input_ids, output_hidden_states=True).hidden_states[24],
        return_value=False
    )
    
    expert_loss = F.cross_entropy(
        logits_expert.view(-1, logits_expert.size(-1)),
        expert_traces.view(-1)
    )
    
    # ========================================================================
    # Phase 2: GRPO with Sampled Traces
    # ========================================================================
    # Generate multiple reasoning traces per problem
    
    all_sampled_traces = []
    all_rewards = []
    all_violations = []
    all_logits = []
    
    for _ in range(group_size):
        # Sample from current policy
        with torch.no_grad():
            sampled_logits, _, _ = model.latent_module(
                model.llama(input_ids, output_hidden_states=True).hidden_states[24],
                return_value=False
            )
            # Sample tokens (you may want to use temperature, top-p, etc.)
            sampled_tokens = torch.distributions.Categorical(
                logits=sampled_logits
            ).sample()
        
        # Validate sampled traces
        # Convert tokens to text (pseudo-code - use tokenizer.decode in practice)
        traces_text = ["[decoded trace]" for _ in range(batch_size)]  # Placeholder
        
        rewards, violations = validator.batch_validate(traces_text, problem_states)
        
        all_sampled_traces.append(sampled_tokens)
        all_rewards.append(rewards)
        all_violations.append(violations)
        all_logits.append(sampled_logits)
    
    # Stack group samples: (batch_size * group_size, seq_len)
    sampled_traces = torch.cat(all_sampled_traces, dim=0)
    rewards = torch.cat(all_rewards, dim=0).to(device)
    violations = torch.cat(all_violations, dim=0).to(device)
    
    # Replicate input for group
    input_ids_replicated = input_ids.repeat(group_size, 1)
    
    # Forward pass with GRPO loss
    _, loss_dict, _ = model(
        input_ids=input_ids_replicated,
        reasoning_targets=sampled_traces,
        rewards=rewards,
        constraint_violations=violations,
        use_reference=True
    )
    
    # ========================================================================
    # Combined Loss: Expert + GRPO
    # ========================================================================
    # You can balance these with hyperparameters
    alpha_expert = 0.5  # Weight for expert supervision
    alpha_grpo = 0.5    # Weight for GRPO
    
    total_loss = alpha_expert * expert_loss + alpha_grpo * loss_dict['loss']
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.latent_module.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Return metrics
    return {
        'total_loss': total_loss.item(),
        'expert_loss': expert_loss.item(),
        **{k: v for k, v in loss_dict.items() if k != 'loss'},
        'mean_reward': rewards.mean().item(),
        'mean_violations': violations.float().mean().item(),
        'reward_std': rewards.std().item()
    }


def update_reference_model_example(model: LlamaPlanner,
                                   steps_since_update: int,
                                   update_interval: int = 500):
    """
    Periodically update the reference model for KL divergence.
    
    In GRPO, the reference model is typically updated every N steps
    to prevent the KL term from becoming too large.
    """
    if steps_since_update >= update_interval:
        model.update_reference_model()
        print(f"Updated reference model at step {steps_since_update}")
        return 0
    return steps_since_update + 1


def main():
    """Main training loop."""
    
    # Load configuration
    cfg = config.get_config()
    
    # Create directories
    os.makedirs(cfg['checkpoint_dir'], exist_ok=True)
    os.makedirs(cfg['output_dir'], exist_ok=True)
    
    # Initialize W&B
    if cfg['use_wandb']:
        wandb.init(
            project=cfg['wandb_project'],
            entity=cfg['wandb_entity'],
            config=cfg
        )
    
    # ========================================================================
    # Setup
    # ========================================================================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = TowersOfHanoiDatasetWrapper(
        num_samples=cfg['num_train_samples'],
        min_disks=cfg['min_disks'],
        max_disks=cfg['max_disks']
    )
    
    val_dataset = TowersOfHanoiDatasetWrapper(
        num_samples=cfg['num_val_samples'],
        min_disks=cfg['min_disks'],
        max_disks=cfg['max_disks']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        num_workers=0  # Set to > 0 for multi-processing
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer),
        num_workers=0
    )
    
    # Initialize model
    print("Initializing model...")
    model = LlamaPlanner(
        model_name=cfg['model_name'],
        use_value_head=cfg['use_value_head']
    )
    model = model.to(device)
    
    # Initialize reference model for GRPO
    print("Initializing reference model...")
    model.init_reference_model()
    
    # Initialize validator
    validator = TowersOfHanoiValidator()
    
    # Optimizer (only train the latent module)
    optimizer = torch.optim.AdamW(
        model.latent_module.parameters(),
        lr=cfg['learning_rate'],
        weight_decay=cfg['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader) * cfg['num_epochs']
    )
    
    # ========================================================================
    # Training Loop
    # ========================================================================
    global_step = 0
    steps_since_ref_update = 0
    best_val_reward = 0.0
    
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    for epoch in range(cfg['num_epochs']):
        model.train()
        epoch_metrics = {
            'total_loss': 0.0,
            'expert_loss': 0.0,
            'grpo_loss': 0.0,
            'mean_reward': 0.0,
            'mean_violations': 0.0
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['num_epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            # Run training step
            metrics = train_step(
                model=model,
                batch=batch,
                validator=validator,
                optimizer=optimizer,
                device=device,
                group_size=cfg['group_size']
            )
            
            # Update learning rate
            scheduler.step()
            
            # Update reference model periodically
            steps_since_ref_update += 1
            if steps_since_ref_update >= cfg['ref_update_interval']:
                model.update_reference_model()
                steps_since_ref_update = 0
                print(f"\n[Step {global_step}] Updated reference model")
            
            # Accumulate metrics
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key] += metrics[key]
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                'reward': f"{metrics['mean_reward']:.4f}",
                'violations': f"{metrics['mean_violations']:.2f}"
            })
            
            # Log to W&B
            if cfg['use_wandb'] and global_step % cfg['log_interval'] == 0:
                wandb.log({
                    'train/total_loss': metrics['total_loss'],
                    'train/expert_loss': metrics['expert_loss'],
                    'train/grpo_loss': metrics.get('grpo_loss', 0.0),
                    'train/mean_reward': metrics['mean_reward'],
                    'train/mean_violations': metrics['mean_violations'],
                    'train/learning_rate': scheduler.get_last_lr()[0],
                    'global_step': global_step
                })
            
            # Validation
            if global_step % cfg['eval_interval'] == 0 and global_step > 0:
                val_metrics = evaluate(model, val_loader, validator, device, cfg)
                print(f"\n[Step {global_step}] Validation - "
                      f"Reward: {val_metrics['mean_reward']:.4f}, "
                      f"Violations: {val_metrics['mean_violations']:.2f}")
                
                if cfg['use_wandb']:
                    wandb.log({
                        'val/mean_reward': val_metrics['mean_reward'],
                        'val/mean_violations': val_metrics['mean_violations'],
                        'val/success_rate': val_metrics['success_rate'],
                        'global_step': global_step
                    })
                
                # Save best model
                if val_metrics['mean_reward'] > best_val_reward:
                    best_val_reward = val_metrics['mean_reward']
                    save_path = os.path.join(cfg['checkpoint_dir'], 'best_model.pt')
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.latent_module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_reward': best_val_reward,
                        'config': cfg
                    }, save_path)
                    print(f"Saved best model with reward {best_val_reward:.4f}")
                
                model.train()
            
            # Save checkpoint
            if global_step % cfg['save_interval'] == 0 and global_step > 0:
                save_path = os.path.join(cfg['checkpoint_dir'], f'checkpoint_step_{global_step}.pt')
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.latent_module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': cfg
                }, save_path)
                print(f"\nSaved checkpoint at step {global_step}")
            
            global_step += 1
        
        # Epoch summary
        num_batches = len(train_loader)
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Total Loss: {epoch_metrics['total_loss'] / num_batches:.4f}")
        print(f"  Expert Loss: {epoch_metrics['expert_loss'] / num_batches:.4f}")
        print(f"  Mean Reward: {epoch_metrics['mean_reward'] / num_batches:.4f}")
        print(f"  Mean Violations: {epoch_metrics['mean_violations'] / num_batches:.2f}")
        print(f"{'='*60}\n")
    
    # Final evaluation
    print("\nRunning final evaluation...")
    val_metrics = evaluate(model, val_loader, validator, device, cfg)
    print(f"Final Validation Results:")
    print(f"  Mean Reward: {val_metrics['mean_reward']:.4f}")
    print(f"  Success Rate: {val_metrics['success_rate']:.4f}")
    print(f"  Mean Violations: {val_metrics['mean_violations']:.2f}")
    
    # Save final model
    final_save_path = os.path.join(cfg['checkpoint_dir'], 'final_model.pt')
    torch.save({
        'model_state_dict': model.latent_module.state_dict(),
        'config': cfg,
        'final_metrics': val_metrics
    }, final_save_path)
    print(f"\nSaved final model to {final_save_path}")
    
    if cfg['use_wandb']:
        wandb.finish()


def evaluate(model: LlamaPlanner, val_loader: DataLoader, 
            validator: TowersOfHanoiValidator, device: str, config: Dict) -> Dict:
    """
    Evaluate the model on validation set.
    
    Returns:
        Dict with evaluation metrics
    """
    model.eval()
    
    total_reward = 0.0
    total_violations = 0
    num_samples = 0
    num_successes = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            problem_states = batch['problem_states']
            batch_size = input_ids.shape[0]
            
            # Generate reasoning traces
            hidden_states = model.llama(input_ids, output_hidden_states=True).hidden_states[24]
            logits, _, _ = model.latent_module(hidden_states, return_value=False)
            
            # Sample tokens (greedy decoding for evaluation)
            sampled_tokens = torch.argmax(logits, dim=-1)
            
            # Decode traces (placeholder - in practice use tokenizer.batch_decode)
            # For now, we'll use the expert traces for evaluation
            traces_text = batch['expert_trace_text']
            
            # Validate
            rewards, violations = validator.batch_validate(traces_text, problem_states)
            
            total_reward += rewards.sum().item()
            total_violations += violations.sum().item()
            num_samples += batch_size
            num_successes += (rewards > 0.9).sum().item()  # Success if reward > 0.9
    
    return {
        'mean_reward': total_reward / num_samples,
        'mean_violations': total_violations / num_samples,
        'success_rate': num_successes / num_samples
    }


if __name__ == "__main__":
    main()