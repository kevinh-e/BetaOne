# config.py
"""
Configuration settings for the AlphaZero Chess engine.
"""

import torch

# --- Hardware ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Chess Game ---
BOARD_SIZE = 8
INPUT_CHANNELS = 17  # 8 history * 2 players + 1 color = 17 channels for input tensor

# See AlphaGo Zero paper for details on input features:
# (Piece type: 6 planes * 2 players = 12 planes)
# (Repetitions: 2 planes for 1/2 repetitions)
# (Total move count: 1 plane, scaled)
# (Player color: 1 plane)
# (Legality of moves: 1 plane - optional, can be derived)
# Simplified here: 8 previous board states for each player (8*2=16) + player color (1)
NUM_ACTIONS = 4672  # Maximum number of possible moves in chess (including promotions)

# --- MCTS ---
NUM_SIMULATIONS = 100  # Number of MCTS simulations per move
CPUCT = 1.0  # Exploration constant in PUCT formula
TEMPERATURE_INITIAL = 1.0  # Initial temperature for action selection during self-play
TEMPERATURE_FINAL = 0.1  # Final temperature
TEMPERATURE_THRESHOLD = 30  # Move number after which temperature changes
DIRICHLET_ALPHA = 0.3  # Alpha value for Dirichlet noise
DIRICHLET_EPSILON = 0.25  # Epsilon value for Dirichlet noise (fraction of noise)

# --- Neural Network ---
RESIDUAL_BLOCKS = 19  # Number of residual blocks in the network
CONV_FILTERS = 256  # Number of filters in convolutional layers

# --- Training ---
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EPOCHS_PER_ITERATION = 1  # Number of training epochs per self-play iteration
NUM_ITERATIONS = 100  # Total number of training iterations (self-play -> train)
CHECKPOINT_INTERVAL = 10  # Save model checkpoint every N iterations
GAME_BUFFER_SIZE = 50000  # Maximum number of games to store for training data

# --- Paths ---
SAVE_DIR = "checkpoints"
LOG_DIR = "logs"
DATA_DIR = "data"

print(f"Using device: {DEVICE}")
