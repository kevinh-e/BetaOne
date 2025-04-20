# utils.py
"""
Utility functions for board representation and move handling.
"""

import chess
import torch
import numpy as np
from typing import List, Tuple

import config

# --- Board Encoding ---


def encode_board(board: chess.Board, history: List[chess.Board]) -> torch.Tensor:
    """
    Encodes the current board state and recent history into a tensor.

    Args:
        board: The current chess.Board object.
        history: A list of the last few board states (chess.Board objects).

    Returns:
        A PyTorch tensor representing the board state.
        Shape: (INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    """
    # TODO: Implement the actual board encoding based on AlphaZero paper's features.
    # This is a placeholder implementation.
    # A common approach involves creating planes for:
    # - Piece positions (Pawn, Knight, Bishop, Rook, Queen, King) for current player
    # - Piece positions for opponent player
    # - Repetition counts
    # - Player color
    # - Castling rights
    # - En passant square
    # - Move count

    # Placeholder: Simple encoding (just current board, piece types)
    encoded = np.zeros(
        (config.INPUT_CHANNELS, config.BOARD_SIZE, config.BOARD_SIZE), dtype=np.float32
    )

    # Example: Plane for white pawns
    for square in board.pieces(chess.PAWN, chess.WHITE):
        rank, file = chess.square_rank(square), chess.square_file(square)
        encoded[0, rank, file] = 1

    # Example: Plane for black pawns
    for square in board.pieces(chess.PAWN, chess.BLACK):
        rank, file = chess.square_rank(square), chess.square_file(square)
        encoded[1, rank, file] = 1

    # ... add planes for other piece types and colors ...

    # Example: Plane for player color (1.0 for white, 0.0 for black)
    encoded[config.INPUT_CHANNELS - 1, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # --- Incorporate History (Simplified Example) ---
    # You'd typically stack planes from previous board states.
    # history_len = min(len(history), (config.INPUT_CHANNELS - 1) // 2) # Max 8 history states per player
    # for i in range(history_len):
    #     # Encode history[-(i+1)] similar to the current board
    #     # Place encoded planes at appropriate channel indices
    #     pass

    return torch.from_numpy(encoded)


# --- Move Handling ---


def move_to_index(move: chess.Move) -> int:
    """
    Converts a chess.Move object to a unique integer index.
    Needs a consistent mapping for all possible moves.
    See python-chess documentation or AlphaZero resources for common mappings.

    Args:
        move: The chess.Move object.

    Returns:
        An integer index representing the move.
    """
    # TODO: Implement a robust mapping from move to index (0 to NUM_ACTIONS-1)
    # This is highly dependent on the chosen action space representation.
    # A common way is to map based on source square, target square, and promotion piece.
    # Example placeholder:
    # return move.from_square * 64 + move.to_square # Oversimplified, doesn't handle promotions etc.
    if move is None:
        return -1  # Or handle appropriately

    # A more complete approach (still needs refinement for specific action space):
    # Based on https://ai.stackexchange.com/questions/15906/how-are-moves-represented-in-alphazero
    # 1. Queen moves (56 directions * 7 squares max = 392)
    # 2. Knight moves (8 directions * 1 square = 8)
    # 3. Underpromotions (Queen/Rook/Bishop * 3 directions * 64 squares = 576 - needs exact mapping)
    # This requires a precise and consistent mapping. Let's use a placeholder.
    # For a real implementation, use a library or define the full mapping carefully.
    # Placeholder: Use UCI string hash (not ideal for NN output layer)
    # return hash(move.uci()) % config.NUM_ACTIONS

    # Placeholder: Return a simple index based on squares (needs expansion!)
    # This is NOT a valid mapping for AlphaZero's action space.
    idx = move.from_square * 64 + move.to_square
    # Need to add logic for promotion pieces etc. to reach NUM_ACTIONS
    return idx % config.NUM_ACTIONS  # Very basic placeholder


def index_to_move(index: int, board: chess.Board) -> chess.Move | None:
    """
    Converts an integer index back to a chess.Move object,
    considering only legal moves for the current board state.

    Args:
        index: The integer index representing the potential move.
        board: The current chess.Board to check legality against.

    Returns:
        The corresponding chess.Move object if it's legal, otherwise None.
    """
    # TODO: Implement the reverse mapping from index to move.
    # This must be the inverse of move_to_index.
    # After getting the potential move from the index, check if it's legal.

    # Placeholder logic (depends heavily on move_to_index implementation)
    # potential_move_uci = ... # Map index back to UCI string or move components
    # try:
    #    move = board.parse_uci(potential_move_uci)
    #    if move in board.legal_moves:
    #        return move
    # except ValueError:
    #    pass # Invalid UCI or illegal move
    # return None

    # Since move_to_index is a placeholder, this is also a placeholder.
    # Iterate through legal moves and see if one maps to the target index.
    # This is inefficient but works for a placeholder.
    for move in board.legal_moves:
        if move_to_index(move) == index:
            return move
    return None


def get_legal_mask(board: chess.Board) -> torch.Tensor:
    """
    Creates a mask indicating legal moves for the current board state.

    Args:
        board: The current chess.Board object.

    Returns:
        A PyTorch tensor (boolean or float) of shape (NUM_ACTIONS,)
        where legal moves are marked (e.g., True or 1.0).
    """
    mask = torch.zeros(config.NUM_ACTIONS, dtype=torch.bool)
    for move in board.legal_moves:
        idx = move_to_index(move)
        if 0 <= idx < config.NUM_ACTIONS:
            mask[idx] = True
    return mask


# --- Game Outcome ---


def get_game_outcome(board: chess.Board) -> float | None:
    """
    Determines the game outcome from the perspective of the current player.

    Args:
        board: The final chess.Board object.

    Returns:
        1.0 if the current player won.
        -1.0 if the current player lost.
        0.0 for a draw.
        None if the game is not over.
    """
    if not board.is_game_over():
        return None

    result = board.result()
    current_player_turn = board.turn  # Whose turn it WOULD be if game continued

    if result == "1-0":  # White wins
        return (
            1.0 if current_player_turn == chess.BLACK else -1.0
        )  # White won, so if it's Black's turn (hypothetically), Black lost (-1)
    elif result == "0-1":  # Black wins
        return (
            1.0 if current_player_turn == chess.WHITE else -1.0
        )  # Black won, so if it's White's turn (hypothetically), White lost (-1)
    elif result == "1/2-1/2":  # Draw
        return 0.0
    else:  # Other draw conditions (stalemate, insufficient material, etc.)
        # Check specific draw conditions if needed
        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_seventyfive_moves()
            or board.is_fivefold_repetition()
        ):
            return 0.0

    # Should not be reached if board.is_game_over() is true and result is handled
    print(f"Warning: Unhandled game over state? Result: {result}, Board: {board.fen()}")
    return 0.0  # Default to draw for safety
