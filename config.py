# config.py
"""
Configuration settings for the AlphaZero Chess engine.
"""

import torch

# --- Hardware ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Chess Game ---
BOARD_SIZE = 8
# 0–11  |   Current position: 6 piece types × 2 colors
# 12–13 |   Repetition once / twice
# 14–27 |   Position at t–1: same 14 planes
# 28–41 |   Position at t–2: same 14 planes
# … | …
# 98–111|   Position at t–7
# 112   |   Player‑to‑move (all 1’s if White, all 0’s if Black)
# 113   |   White kingside castling right (1 or 0, everywhere)
# 114   |   White queenside castling right
# 115   |   Black kingside castling right
# 116   |   Black queenside castling right
# 117   |   Half‑move clock (no‑progress count)
# 118   |   Full‑move number
# 119   |   En passant
INPUT_CHANNELS = 120
NUM_ACTIONS = 8 * 8 * 73

# --- MCTS ---
NUM_SIMULATIONS = 125  # Number of MCTS simulations per move
CPUCT = 1.0  # Exploration constant in PUCT formula
TEMPERATURE_INITIAL = 1.0  # Initial temperature for action selection during self-play
TEMPERATURE_FINAL = 0.1  # Final temperature
TEMPERATURE_THRESHOLD = 30  # Move number after which temperature changes
DIRICHLET_ALPHA = 0.3  # Alpha value for Dirichlet noise
DIRICHLET_EPSILON = 0.25  # Epsilon value for Dirichlet noise (fraction of noise)

# --- Neural Network ---
RESIDUAL_BLOCKS = 13  # Number of residual blocks in the network
CONV_FILTERS = 256  # Number of filters in convolutional layers

# --- Training ---
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EPOCHS_PER_ITERATION = 25  # Number of training epochs per self-play iteration
NUM_ITERATIONS = 100  # Total number of training iterations (self-play -> train)
CHECKPOINT_INTERVAL = 2  # Save model checkpoint every N iterations
GAME_BUFFER_SIZE = 50000  # Maximum number of games to store for training data

# --- Paths ---
SAVE_DIR = "checkpoints"
LOG_DIR = "logs"
DATA_DIR = "data"

# print(f"Using device: {DEVICE}")
