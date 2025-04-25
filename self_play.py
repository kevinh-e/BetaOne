# self_play.py
"""
Handles the self-play process to generate game data for training.
Uses MCTS guided by the current best neural network.
"""

import os
import time
import chess
import numpy as np
import torch
from typing import List, Tuple, Dict
import pickle  # For saving/loading game data

import config
import utils
from network import PolicyValueNet
from mcts import run_mcts

# Define the structure for storing game data
# Each element in the training data list will be a tuple:
# (encoded_board_state, improved_policy, game_outcome)
SelfPlayData = Tuple[torch.Tensor, np.ndarray, float]


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """
    Applies temperature scaling to probabilities. Lower temp -> more greedy.
    """
    if temperature == 0:  # Choose the best move
        new_probs = np.zeros_like(probs)
        new_probs[np.argmax(probs)] = 1.0
        return new_probs
    elif temperature == 1.0:  # No change
        return probs
    else:
        # Apply temperature: p^(1/T) / sum(p^(1/T))
        scaled_probs = np.power(probs, 1.0 / temperature)

        # Exceptions
        scaled_probs[np.isnan(scaled_probs)] = 0.0
        sum_scaled_probs = np.sum(scaled_probs)
        if sum_scaled_probs > 1e-6:
            return scaled_probs / sum_scaled_probs
        else:
            print("Warning: Sum of scaled probabilities is near zero. Using original.")
            return probs


def select_move_with_temperature(probs: np.ndarray, move_number: int) -> int:
    """
    Selects a move index based on probabilities, applying temperature.
    """
    # Determine temperature based on move number
    if move_number < config.TEMPERATURE_THRESHOLD:
        temp = config.TEMPERATURE_INITIAL
    else:
        temp = config.TEMPERATURE_FINAL

    temp_scaled_probs = apply_temperature(probs, temp)

    # Sample a move index based on the scaled probabilities & normalise
    temp_scaled_probs /= np.sum(temp_scaled_probs)
    try:
        # Use multinomial sampling to choose based on probabilities
        action_index = np.random.choice(len(temp_scaled_probs), p=temp_scaled_probs)
    except ValueError as e:
        print(f"Error sampling move: {e}")
        print(f"Probabilities: {temp_scaled_probs}")
        print(f"Sum: {np.sum(temp_scaled_probs)}")
        # Fallback: action with the highest probability
        action_index = np.argmax(temp_scaled_probs)

    return action_index


def run_self_play_game(
    model: PolicyValueNet, game_id: int
) -> List[SelfPlayData] | None:
    """
    Plays a single game of chess using MCTS guided by the model.

    Args:
        model: The current neural network model.
        game_id: An identifier for the game (for logging/saving).

    Returns:
        A list of training examples (state, policy, value) from the game,
        or None if the game fails unexpectedly.
    """
    board = chess.Board()
    game_data: List[Tuple[chess.Board, np.ndarray]] = []  # Store (board, mcts_policy)
    board_history: List[chess.Board] = []
    tracker = utils.RepetitionTracker(board)

    print(f"Starting self-play game {game_id}...")
    start_time = time.time()

    while not board.is_game_over():
        move_number = board.fullmove_number  # Or len(board.move_stack) for ply count

        # MCTS
        history_for_mcts = board_history[-(config.INPUT_CHANNELS // 2 * 2) :]
        best_move, mcts_policy = run_mcts(board, model, history_for_mcts, tracker)
        if best_move is None:
            print(
                f"Error: MCTS returned no move for game {game_id} at FEN: {board.fen()}"
            )
            return None

        # Store the state and the MCTS policy target (encode later)
        game_data.append((board.copy(), mcts_policy))

        # Choose the move to play using temperature
        action_index = select_move_with_temperature(mcts_policy, move_number)

        played_move = utils.index_to_move(action_index, board)
        if played_move is None or played_move not in board.legal_moves:
            print(
                f"Warning: Sampled index {action_index} led to illegal/invalid move. Falling back to best MCTS move."
            )
            played_move = best_move

        # Play the move
        board.push(played_move)
        board_history.append(board.copy())

        # Optional: Print board state periodically
        # if move_number % 10 == 0:
        #     print(f"\nGame {game_id}, Move {move_number}")
        #     print(board)

    outcome = utils.get_game_outcome(board)
    if outcome is None:
        print(f"Error: Game {game_id} ended but outcome is None. FEN: {board.fen()}")
        # Fallback to draw
        outcome = 0.0

    print(
        f"Game {game_id} finished. Result: {board.result()} ({outcome:.1f}). Moves: {len(board.move_stack)}. Time: {time.time() - start_time:.2f}s"
    )

    # Prepare training data: (encoded_state, mcts_policy, final_outcome)
    # We assign the outcome relative to the player whose turn it was at each state
    training_examples: List[SelfPlayData] = []
    for i, (state_board, policy) in enumerate(game_data):
        # Determine perspective: 1.0 = win, -1.0 if they lost, 0.0 for draw.
        if state_board.turn == chess.WHITE:
            perspective_outcome = outcome
        else:
            perspective_outcome = -outcome

        history_up_to_state = board_history[:i]
        # Encode board state
        encoded_state = utils.encode_board(state_board, history_up_to_state, tracker)

        training_examples.append((encoded_state, policy, perspective_outcome))

    return training_examples


def save_game_data(game_data: List[SelfPlayData], iteration: int, game_id: int):
    """Saves the generated game data to disk."""
    if not game_data:
        return

    data_dir = os.path.join(config.DATA_DIR, f"iter_{iteration}")
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, f"game_{game_id}.pkl")

    try:
        with open(filepath, "wb") as f:
            pickle.dump(game_data, f)
        # print(f"Saved game data to {filepath}")
    except Exception as e:
        print(f"Error saving game data to {filepath}: {e}")


# Example usage (within a larger training loop)
if __name__ == "__main__":
    print("Running self-play example...")
    # Load or initialize the model
    model = PolicyValueNet().to(config.DEVICE)
    # model.load_state_dict(torch.load("path/to/best_model.pth")) # Load weights if available
    model.eval()  # Set model to evaluation mode for MCTS

    # Ensure directories exist
    os.makedirs(config.DATA_DIR, exist_ok=True)

    # Run a single game
    game_result_data = run_self_play_game(model, game_id=0)

    if game_result_data:
        print(f"\nGenerated {len(game_result_data)} training examples from game 0.")
        # Example: Print first training sample details
        state_tensor, policy_target, value_target = game_result_data[0]
        print("First training sample:")
        print("  State Tensor Shape:", state_tensor.shape)
        # print("  Policy Target (sum):", np.sum(policy_target))
        # print("  Policy Target (non-zero indices):", np.where(policy_target > 0)[0])
        print("  Value Target:", value_target)

        # Save the data (example for iteration 0)
        save_game_data(game_result_data, iteration=0, game_id=0)
    else:
        print("Self-play game failed.")
