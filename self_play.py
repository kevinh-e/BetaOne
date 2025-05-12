# self_play.py
"""
Handles the self-play process to generate game data for training
Uses MCTS guided by the current best neural network and revised RepetitionTracker.
"""

import os
import time
import chess
import numpy as np
import torch
from typing import List, Tuple
import pickle

import config
import utils
from network import PolicyValueNet
from mcts import run_mcts

# Define the structure for storing game data
SelfPlayData = Tuple[torch.Tensor, np.ndarray, float]


# --- apply_temperature and select_move_with_temperature ---
def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Applies temperature scaling to probabilities."""
    if temperature == 0:
        # greedy
        new_probs = np.zeros_like(probs)
        max_prob_indices = np.where(probs == np.max(probs))[0]
        if len(max_prob_indices) == 0:
            return new_probs
        chosen_index = np.random.choice(max_prob_indices)
        new_probs[chosen_index] = 1.0
        return new_probs
    elif abs(temperature - 1.0) < 1e-6:
        return probs
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled_probs = np.power(probs.astype(np.float64), 1.0 / temperature)
        scaled_probs[~np.isfinite(scaled_probs)] = 0.0
        sum_scaled_probs = np.sum(scaled_probs)
        if sum_scaled_probs > 1e-9:
            normalized_probs = (scaled_probs / sum_scaled_probs).astype(np.float32)
            renorm_sum = np.sum(normalized_probs)
            if abs(renorm_sum - 1.0) > 1e-6 and renorm_sum > 1e-9:
                normalized_probs /= renorm_sum
            return normalized_probs
        else:
            non_zero_indices = np.where(probs > 1e-9)[0]
            num_non_zero = len(non_zero_indices)
            if num_non_zero > 0:
                uniform_probs = np.zeros_like(probs, dtype=np.float32)
                uniform_probs[non_zero_indices] = 1.0 / num_non_zero
                return uniform_probs
            else:
                return probs.astype(np.float32)


def select_move_with_temperature(probs: np.ndarray, move_number: int) -> int:
    """Selects a move index based on probabilities by applying temperature"""
    if move_number < config.TEMPERATURE_THRESHOLD:
        temp = config.TEMPERATURE_INITIAL
    else:
        temp = config.TEMPERATURE_FINAL
    temp_scaled_probs = apply_temperature(probs, temp)
    try:
        prob_sum = np.sum(temp_scaled_probs)
        if abs(prob_sum - 1.0) > 1e-6:
            if prob_sum > 1e-9:
                temp_scaled_probs /= prob_sum
            else:
                return np.argmax(probs)  # Fallback if sum is zero
        action_index = np.random.choice(len(temp_scaled_probs), p=temp_scaled_probs)
    except ValueError as e:
        print(
            f"Error sampling move: {e}\nProbs: {temp_scaled_probs}\nSum: {np.sum(temp_scaled_probs)}"
        )
        print("Falling back to argmax of original probabilities.")
        action_index = np.argmax(probs)
    return action_index


# --- Main Self-Play Function ---
def run_self_play_game(
    model: PolicyValueNet, game_id: int
) -> List[SelfPlayData] | None:
    """
    Plays a single game of chess using MCTS guided by the model.
    Uses the revised RepetitionTracker.
    """
    board = chess.Board()
    tracker = utils.RepetitionTracker()
    tracker.add_board(board)  # Manually add the initial board state count

    game_states_for_training: List[Tuple[chess.Board, np.ndarray]] = []
    board_history: List[chess.Board] = [board.copy()]

    start_time = time.time()
    move_count = 0

    while (
        not board.is_game_over(claim_draw=True) and move_count < config.MAX_GAME_MOVES
    ):
        move_number = board.fullmove_number
        current_fen = board.fen()
        current_turn = "White" if board.turn == chess.WHITE else "Black"

        # --- Run MCTS ---
        history_for_mcts_root = board_history[max(0, len(board_history) - 8) : -1]
        best_move, mcts_policy = run_mcts(board, model, history_for_mcts_root, tracker)

        if best_move is None:
            if not board.legal_moves:
                break
            else:
                print(
                    f"Error: MCTS returned no move for game {game_id} but legal moves exist! FEN: {current_fen}"
                )
                return None

        # Store the state (before move) and the MCTS policy target
        game_states_for_training.append((board.copy(), mcts_policy))

        # --- Select and Play Move ---
        action_index = select_move_with_temperature(mcts_policy, move_number)
        decoded_move = None
        try:
            decoded_move = utils.index_to_move(action_index, board)
            played_move = decoded_move
        except ValueError as e:
            print(f"\n!!! CRITICAL ERROR in Self-Play (index_to_move) !!!")
            print(
                f"  Game ID: {game_id}, Move Number: {move_number}, FEN: {current_fen}"
            )
            print(f"  Turn: {current_turn}, Sampled Index: {action_index}, Error: {e}")
            print(f"  MCTS Best Move: {best_move.uci()}, Falling back.")
            played_move = best_move

        # --- Legality Check ---
        current_legal_moves = list(board.legal_moves)
        is_played_move_legal = played_move in current_legal_moves
        if not is_played_move_legal:
            print(f"\n!!! CRITICAL ERROR in Self-Play (Legality Check) !!!")
            print(
                f"  Game ID: {game_id}, Move Number: {move_number}, FEN: {current_fen}"
            )
            print(f"  Turn: {current_turn}")
            print(f"  MCTS Best Move: {best_move.uci()}")
            print(f"  Sampled Index: {action_index}")
            print(
                f"  Decoded Move (from index): {decoded_move.uci() if decoded_move else 'None (Decode Error)'}"
            )
            print(
                f"  Move to be played (after potential fallback): {played_move.uci()} - NOT LEGAL!"
            )
            print(f"  Legal Moves UCI: {[m.uci() for m in current_legal_moves]}")

            if best_move != played_move and best_move in current_legal_moves:
                print(
                    f"  Correcting: Using original best MCTS move {best_move.uci()} instead."
                )
                played_move = best_move
            else:
                print(
                    f"  Best MCTS move {best_move.uci()} is also illegal or same as failed move. Aborting game."
                )
                return None

        # --- Apply Move ---
        try:
            board.push(played_move)  # Push onto the main board
        except AssertionError as e:
            print("\n!!! AssertionError during board.push() !!!")
            print(
                f"  Game ID: {game_id}, Move Number: {move_number}, FEN: {current_fen}"
            )
            print(
                f"  Turn: {current_turn}, Attempted Move: {played_move.uci()}, Error: {e}"
            )
            return None

        tracker.add_board(board)  # Add the new board state to the tracker counts

        board_history.append(board.copy())  # Add new state to history
        move_count += 1
    if move_count >= config.MAX_GAME_MOVES:
        print(f"Game {game_id} aborted after {move_count} moves (max).")

    # --- Game Finished ---
    outcome = utils.get_game_outcome(board)
    if (
        outcome is None
    ):  # Handle cases where game ends but outcome calculation needs claim_draw
        if board.is_game_over(claim_draw=True):
            outcome = utils.get_game_outcome(board)
        if outcome is None:
            outcome = 0.0  # Default to draw if still None

    # --- Prepare Final Training Data ---
    training_examples: List[SelfPlayData] = []
    for i, (state_board, policy) in enumerate(game_states_for_training):
        perspective_outcome = outcome if state_board.turn == chess.WHITE else -outcome
        history_slice_for_training = board_history[max(0, i + 1 - 8) : i + 1]
        try:
            encoded_state = utils.encode_board(
                state_board, history_slice_for_training, tracker
            )
            training_examples.append((encoded_state, policy, perspective_outcome))
        except Exception as e:
            print(f"\n!!! ERROR during training data encoding !!!")
            print(f"  Game ID: {game_id}, State Index: {i}, FEN: {state_board.fen()}")
            print(f"  History Slice Length: {len(history_slice_for_training)}")
            print(f"  Error: {e}")
            # return None # Optionally abort game if encoding fails

    return training_examples


# --- save_game_data (No changes needed) ---
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
    except Exception as e:
        print(f"Error saving game data to {filepath}: {e}")
