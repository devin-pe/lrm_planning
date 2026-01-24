"""
Configuration file for LRM Planning Training.

This file contains all hyperparameters and settings for training the
Latent Reasoning Module on Towers of Hanoi planning problems.
"""

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_NAME = 'meta-llama/Meta-Llama-3-8B'
USE_VALUE_HEAD = False 

# ============================================================================
# Dataset Configuration
# ============================================================================
NUM_TRAIN_SAMPLES = 5000
NUM_VAL_SAMPLES = 500
MIN_DISKS = 2  # Minimum number of disks in generated problems
MAX_DISKS = 10  # Maximum number of disks in generated problems
BATCH_SIZE = 4

# ============================================================================
# Training Configuration
# ============================================================================
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
GROUP_SIZE = 4  # Number of samples per problem for GRPO

# ============================================================================
# Loss Weight Configuration
# ============================================================================
ALPHA_EXPERT = 0.5  # Weight for expert supervision
ALPHA_GRPO = 0.5    # Weight for GRPO

# Loss function hyperparameters (in modules.PlanningLoss)
BETA_KL = 0.01           # KL divergence penalty
BETA_ENTROPY = 0.01      # Entropy regularization
BETA_CONSTRAINT = 1.0    # Constraint violation penalty
BETA_VALUE = 0.5         # Value function loss weight
GAMMA = 0.99             # Discount factor for returns

# ============================================================================
# GRPO Configuration
# ============================================================================
REF_UPDATE_INTERVAL = 500  # Steps between reference model updates

# ============================================================================
# Logging and Checkpointing
# ============================================================================
LOG_INTERVAL = 10        # Steps between logging
EVAL_INTERVAL = 100      # Steps between validation evaluations
SAVE_INTERVAL = 500      # Steps between checkpoint saves
USE_WANDB = False        # Whether to use Weights & Biases logging
WANDB_PROJECT = 'lrm-planning'
WANDB_ENTITY = None      # Your W&B username/team (optional)

# ============================================================================
# Paths
# ============================================================================
CHECKPOINT_DIR = './checkpoints'
OUTPUT_DIR = './outputs'

# ============================================================================
# Helper function to get config as dictionary
# ============================================================================
def get_config():
    """Return all config variables as a dictionary."""
    return {
        # Model
        'model_name': MODEL_NAME,
        'use_value_head': USE_VALUE_HEAD,
        
        # Data
        'num_train_samples': NUM_TRAIN_SAMPLES,
        'num_val_samples': NUM_VAL_SAMPLES,
        'min_disks': MIN_DISKS,
        'max_disks': MAX_DISKS,
        'batch_size': BATCH_SIZE,
        
        # Training
        'num_epochs': NUM_EPOCHS,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'gradient_clip': GRADIENT_CLIP,
        'group_size': GROUP_SIZE,
        
        # Loss weights
        'alpha_expert': ALPHA_EXPERT,
        'alpha_grpo': ALPHA_GRPO,
        'beta_kl': BETA_KL,
        'beta_entropy': BETA_ENTROPY,
        'beta_constraint': BETA_CONSTRAINT,
        'beta_value': BETA_VALUE,
        'gamma': GAMMA,
        
        # GRPO
        'ref_update_interval': REF_UPDATE_INTERVAL,
        
        # Logging
        'log_interval': LOG_INTERVAL,
        'eval_interval': EVAL_INTERVAL,
        'save_interval': SAVE_INTERVAL,
        'use_wandb': USE_WANDB,
        'wandb_project': WANDB_PROJECT,
        'wandb_entity': WANDB_ENTITY,
        
        # Paths
        'checkpoint_dir': CHECKPOINT_DIR,
        'output_dir': OUTPUT_DIR
    }
