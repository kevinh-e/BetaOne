# config.py
"""
Configuration settings for the AlphaZero Chess engine.
"""

import torch

# --- Hardware ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = torch.cuda.is_available()

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
NUM_SIMULATIONS = 300  # Number of MCTS simulations per move
CPUCT = 1.0  # Exploration constant in PUCT formula
TEMPERATURE_INITIAL = 1.0  # Initial temperature for action selection during self-play
TEMPERATURE_FINAL = 0.1  # Final temperature
TEMPERATURE_THRESHOLD = 30  # Move number after which temperature changes
DIRICHLET_ALPHA = 0.1  # Alpha value for Dirichlet noise
# Epsilon value for Dirichlet noise (fraction of noise)
DIRICHLET_EPSILON = 0.25
WIDEN_COEFF = 1.5
MCTS_BATCH_SIZE = 128

# --- Neural Network ---
RESIDUAL_BLOCKS = 18  # Number of residual blocks in the network
CONV_FILTERS = 256  # Number of filters in convolutional layers

# --- Pretraining ---
NUM_EPOCHS = 128
PRETRAINING_T_MAX = 29_578
# --- Training ---
NUM_WORKERS = 6
NUM_THREADS = 3
GAMES_MINIMUM = 100

BATCH_SIZE = 256
MAX_GAME_MOVES = 16384
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
LR_MIN = 1e-6
EPOCHS_PER_ITERATION = 19  # Number of training epochs per self-play iteration
NUM_ITERATIONS = 80  # Total number of training iterations (self-play -> train)
CHECKPOINT_INTERVAL = 1  # Save model checkpoint every N iterations
GAME_BUFFER_SIZE = 100000  # Maximum number of games to store for training data

# --- Paths ---
PGN_DATA_DIR = "pgns"
SAVE_DIR = "checkpoints"
LOG_DIR = "logs"
DATA_DIR = "data"

# print(f"Using device: {DEVICE}")
