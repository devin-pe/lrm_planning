# Latent Reasoning Module (LRM) for Planning

A PyTorch implementation of a latent reasoning system for improving LLM planning capabilities through GRPO (Group Relative Policy Optimization) on the Towers of Hanoi domain.

## Architecture

```
Input Problem → Base LLM (frozen) → Hidden States → Latent Module → Reasoning Tokens
                                                    ↓
                                                Value Head
```

### Components

1. **Base LLM**: Frozen Llama-3-8B for initial representations
2. **Latent Module**: Encoder-Reasoner-Decoder bridge
   - Encoder: Compresses LLM hidden states
   - Reasoner: Transformer layers for structured reasoning
   - Decoder: Projects to vocabulary space
   - Value Head: Predicts expected returns (actor-critic)

## Loss Functions

1. **Task Loss (CE)**: Supervised learning from optimal solver traces
2. **GRPO Loss**: Group Relative Policy Optimization with KL divergence
3. **Constraint Loss**: Penalizes action precondition violations
4. **Entropy Loss**: Encourages exploration
5. **Value Loss**: Actor-critic value function (optional)

## Quick Start

### Installation

```bash
pip install torch transformers tqdm wandb
```

### Test the Validator

```bash
python planning_algo.py
```

This will test:
- Optimal Towers of Hanoi solver
- Reasoning trace generation
- Constraint validation
- Dataset generation

### Training

```bash
python training.py
```

## Configuration

Edit `training.py` to modify hyperparameters:

```python
config = {
    # Model
    'model_name': 'meta-llama/Meta-Llama-3-8B',
    'use_value_head': True,
    
    # Data
    'num_train_samples': 5000,
    'min_disks': 2,
    'max_disks': 5,
    'batch_size': 4,
    
    # Training
    'num_epochs': 10,
    'learning_rate': 1e-4,
    'group_size': 4,  # GRPO group size
    
    # Loss weights
    'alpha_expert': 0.5,  # Expert supervision weight
    'alpha_grpo': 0.5,    # GRPO weight
}
```

## Key Features

### Towers of Hanoi Validator

The `TowersOfHanoiValidator` class:
- Parses reasoning traces to extract moves
- Validates moves against Hanoi rules (no larger disk on smaller disk)
- Tracks state through the reasoning process
- Computes rewards based on:
  - Goal achievement (1.0)
  - Solution optimality (bonus +0.5 for optimal, +0.2 for near-optimal)
  - Progress towards goal (partial reward)
- Counts constraint violations

### GRPO Training

Group Relative Policy Optimization:
1. Samples multiple traces per problem (group)
2. Computes advantages relative to group mean reward
3. Includes KL divergence from reference policy
4. Prevents policy collapse while maintaining diversity

### Expert Supervision

The `TowersOfHanoiSolver` generates optimal solutions:
- Recursive algorithm for optimal move sequence
- Detailed reasoning traces for supervised learning
- Provides structural template for correct reasoning

## File Structure

```
lrm_planning/
├── modules.py           # Core LRM components
│   ├── LatentModule     # Encoder-Reasoner-Decoder
│   ├── PlanningLoss     # Multi-objective loss
│   └── LlamaPlanner     # Main model
│
├── planning_algo.py     # Towers of Hanoi domain
│   ├── TowersOfHanoiState      # State representation
│   ├── TowersOfHanoiSolver     # Optimal solver
│   ├── TowersOfHanoiValidator  # Constraint validation
│   └── TowersOfHanoiDataset    # Data generation
│
└── training.py          # Complete training script
    ├── TowersOfHanoiDatasetWrapper
    ├── train_step()     # Single training step
    ├── evaluate()       # Validation
    └── main()           # Full training loop
```

## Training Process

1. **Initialization**:
   - Load frozen Llama base model
   - Initialize latent module
   - Create reference model for KL divergence

2. **For each batch**:
   - **Phase 1 (Supervised)**: Train on expert optimal traces
   - **Phase 2 (GRPO)**: 
     - Sample K traces per problem
     - Validate with TowersOfHanoiValidator
     - Compute group-relative advantages
     - Update policy with combined loss

3. **Periodic**:
   - Update reference model (every 500 steps)
   - Evaluate on validation set
   - Save checkpoints

## Monitoring

### W&B Integration

Enable Weights & Biases logging:

```python
config['use_wandb'] = True
```

Tracked metrics:
- Training loss components
- Validation rewards and success rate
- Constraint violations
- Learning rate

### Console Output

```
Epoch 1/10: 100%|██████| 1250/1250 [12:34<00:00, loss=0.4532, reward=0.8234, violations=0.12]

[Step 500] Updated reference model
[Step 1000] Validation - Reward: 0.8456, Violations: 0.08

Epoch 1 Summary:
  Total Loss: 0.4321
  Expert Loss: 0.2134
  Mean Reward: 0.8234
  Mean Violations: 0.15
```

## Extending to Other Domains

To adapt this system to other planning domains:

1. **Create domain-specific validator** (inherit from `PlanningValidator`):
   ```python
   class MyDomainValidator:
       def validate_trace(self, trace: str, problem_state: Dict):
           # Parse trace
           # Check preconditions
           # Compute rewards
           return reward, violations
   ```

2. **Implement optimal solver** (for expert traces):
   ```python
   class MyDomainSolver:
       def solve(self, problem):
           # Return optimal solution
           pass
       
       def generate_reasoning_trace(self, problem):
           # Return detailed reasoning
           pass
   ```

3. **Update training script** to use your validator and dataset

## Hyperparameter Tuning

Recommended starting points:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `beta_kl` | 0.01 | KL penalty (higher = stay closer to reference) |
| `beta_entropy` | 0.01 | Exploration bonus |
| `beta_constraint` | 1.0 | Constraint violation penalty |
| `group_size` | 4-8 | Samples per problem for GRPO |
| `alpha_expert` | 0.5 | Expert supervision weight |
| `alpha_grpo` | 0.5 | GRPO weight |

## Citation

If you use this code, please cite:
```bibtex
@misc{lrm-planning-2026,
  title={Latent Reasoning Module for Planning with GRPO},
  author={Your Name},
  year={2026}
}
```

## License

MIT License
