# utils.py
"""
Utility functions for board representation and move handling.
"""

import enum
from math import pi
import chess
import torch
import numpy as np
from typing import List
from collections import Counter

import config


# List of piece-type/color pairs in the order they occupy channels 0..11
PIECE_ORDER = [
    (chess.PAWN, chess.WHITE),
    (chess.PAWN, chess.BLACK),
    (chess.KNIGHT, chess.WHITE),
    (chess.KNIGHT, chess.BLACK),
    (chess.BISHOP, chess.WHITE),
    (chess.BISHOP, chess.BLACK),
    (chess.ROOK, chess.WHITE),
    (chess.ROOK, chess.BLACK),
    (chess.QUEEN, chess.WHITE),
    (chess.QUEEN, chess.BLACK),
    (chess.KING, chess.WHITE),
    (chess.KING, chess.BLACK),
]
HISTORY_BLOCK_SIZE = len(PIECE_ORDER) + 2  # 12 piece planes + 2 repetition planes = 14


QUEEN_DIRECTIONS = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
]
KNIGHT_DIRECTIONS = [
    (2, 1),
    (1, 2),
    (-1, 2),
    (-2, 1),
    (-2, -1),
    (-1, -2),
    (1, -2),
    (2, -1),
]
# --- Underpromotions ---
PROMO_BASE = {
    chess.ROOK: 0,
    chess.BISHOP: 3,
    chess.KNIGHT: 6,
}
# Direction index per (dx, dy), relative to from‐square
# White promotions (dy = +1)
WHITE_DIR_IDX = {
    (1, 1): 0,  # right‐capture
    (0, 1): 1,  # straight
    (-1, 1): 2,  # left‐capture
}
# Black promotions (dy = -1)
BLACK_DIR_IDX = {
    (1, -1): 0,
    (0, -1): 1,
    (-1, -1): 2,
}
INV_PROMO_BASE = {
    0: chess.ROOK,  # slots 0,1,2
    3: chess.BISHOP,  # slots 3,4,5
    6: chess.KNIGHT,  # slots 6,7,8
}
# inverse dir-idx for white & black
INV_WHITE_DIR = {
    0: (1, 1),  # right-capture
    1: (0, 1),  # straight
    2: (-1, 1),  # left-capture
}
INV_BLACK_DIR = {
    0: (1, -1),
    1: (0, -1),
    2: (-1, -1),
}

ACTIONS_PLANES = 73


# --- Repetition Tracker ---
class RepetitionTracker:
    def __init__(self, board: chess.Board):
        self.board = board
        self.counts = Counter()

        # count the initial position
        self.counts[board._transposition_key()] += 1

    def push(self, move: chess.Move):
        self.board.push(move)
        self.counts[self.board._transposition_key()] += 1

    def pop(self):
        key = self.board._transposition_key()
        self.counts[key] -= 1
        self.board.pop()

    def repetitions(self, board: chess.Board) -> int:
        """
        Returns the how many times this position has happenned
        """
        return self.counts[self.board._transposition_key()] - 1


# --- Board Encoding ---
def encode_board(
    board: chess.Board, history: List[chess.Board], tracker: RepetitionTracker
) -> torch.Tensor:
    """
    Encodes the current board state and recent history into a tensor.

    Args:
        board: The current chess.Board object.
        history: A list of the last few board states (chess.Board objects).
        tracker: Tracker for the number of board repetitions

    Returns:
        A PyTorch tensor representing the board state.
        Shape: (INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    """

    # - Piece positions (Pawn, Knight, Bishop, Rook, Queen, King) for current player
    # - Piece positions for opponent player
    # - Repetition counts
    # - History
    # - Player color
    # - Castling rights
    # - Halfmove clock
    # - Move count
    # - En passant square

    # init 0s
    encoded = np.zeros(
        (config.INPUT_CHANNELS, config.BOARD_SIZE, config.BOARD_SIZE),
        dtype=np.float32,
    )

    def encode_piece(
        board: chess.Board,
        piece_type: chess.PieceType,
        color: bool,
        channel: int,
        encoded: np.ndarray,
    ) -> None:
        """
        Marks all squares of a given piece_type/color in the specified channel.
        """
        for square in board.pieces(piece_type, color):
            r, f = chess.square_rank(square), chess.square_file(square)
            encoded[channel, r, f] = 1.0

    def encode_history(
        history: List[chess.Board],
        tracker: RepetitionTracker,
        encoded: np.ndarray,
        max_steps: int = 8,
    ) -> None:
        """
        Fills `encoded` with piece and repetition features for up to `max_steps` boards.

        - history: list of past boards (oldest first, current last)
        - tracker: tracks repetition counts for any board
        - encoded: numpy array of shape (C, 8, 8) already zero-initialized
        - max_steps: how many time-steps to include (typically 8)

        This writes into channels 0..(max_steps*HISTORY_BLOCK_SIZE - 1).
        """
        # Take only the last max_steps boards (pad if fewer)
        recent = history[-max_steps:]
        # If too few, pad at front with empty boards (all zeros + no repeats)
        if len(recent) < max_steps:
            pad_count = max_steps - len(recent)
            recent = [chess.Board(None)] * pad_count + recent

        for i, b in enumerate(recent):
            base = i * HISTORY_BLOCK_SIZE
            # 1) piece planes
            for j, (ptype, clr) in enumerate(PIECE_ORDER):
                encode_piece(b, ptype, clr, base + j, encoded)
            # 2) repetition planes
            rep = tracker.repetitions(b)
            encoded[base + 12, :, :] = 1.0 if rep >= 1 else 0.0
            encoded[base + 13, :, :] = 1.0 if rep >= 2 else 0.0

    # Player turn
    encoded[112, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # castling P1
    encoded[113, :, :] = board.has_kingside_castling_rights(board.turn)
    encoded[114, :, :] = board.has_queenside_castling_rights(board.turn)

    # castling P2
    opp = not board.turn
    encoded[115, :, :] = board.has_kingside_castling_rights(opp)
    encoded[116, :, :] = board.has_queenside_castling_rights(opp)

    # halfmove_clock
    encoded[117, :, :] = board.halfmove_clock

    # Total move count
    encoded[118, :, :] = board.fullmove_number

    if board.ep_square is not None:
        rank, file = (
            chess.square_rank(board.ep_square),
            chess.square_file(board.ep_square),
        )
        encoded[119, rank, file] = 1.0

    return torch.from_numpy(encoded)


# --- Move Handling ---
def normalise(dx, dy):
    """
    Reduce (dx, dy) to unit directions
    """
    if dx != 0:
        dx //= abs(dx)
    if dy != 0:
        dy //= abs(dy)
    return dx, dy


def move_to_index(move: chess.Move) -> int:
    """
        Converts a chess.Move object to a unique integer index.
        Needs a consistent mapping for all possible moves.
        See python-chess documentation or AlphaZero resources for common mappings.

        Args:
            move: The chess.Move object.
    CTS Enhancement: It uses Monte Carlo Tree Search (MCTS) as its search algorithm du
        Returns:
            An integer index representing the move.
    """
    # 1. Queen moves (56 directions)
    # 2. Knight moves (8 directions)
    # 3. Underpromotions (Queen/Rook/Bishop * 3 directions)
    from_square = move.from_square
    to_square = move.to_square
    promotion = move.promotion

    # get rank and file
    from_r = chess.square_rank(from_square)
    from_f = chess.square_file(from_square)
    to_r = chess.square_rank(to_square)
    to_f = chess.square_file(to_square)

    delta_r = to_r - from_r
    delta_f = to_f - from_f

    ndr, ndf = normalise(delta_r, delta_f)

    # RBQ moves
    if (ndr, ndf) in QUEEN_DIRECTIONS:
        direction = QUEEN_DIRECTIONS.index((ndr, ndf))
        distance = max(abs(delta_r), abs(delta_f)) - 1
        return from_square * ACTIONS_PLANES + direction * 7 + distance

    # knight moves
    elif (ndr, ndf) in KNIGHT_DIRECTIONS:
        direction = KNIGHT_DIRECTIONS.index((ndr, ndf))
        return from_square * ACTIONS_PLANES + 56 + direction

    # Underpromotion
    if promotion and promotion != chess.QUEEN:
        dir_map = WHITE_DIR_IDX if delta_r > 0 else BLACK_DIR_IDX
        try:
            direction = dir_map[(delta_f, delta_r)]
            base = PROMO_BASE[promotion]
        except KeyError:
            raise ValueError(f"Invalid underpromotion vector {(delta_f, delta_r)}")

        return from_square * ACTIONS_PLANES + 64 + base + direction
    return 1


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

    if not (0 <= index < config.NUM_ACTIONS):
        raise ValueError("Index out of valid range (0 to 4671")

    from_sq = index // ACTIONS_PLANES
    offset = index % ACTIONS_PLANES

    from_f = chess.square_file(from_sq)
    from_r = chess.square_rank(from_sq)

    # RBQ moves
    if offset < 56:
        direction = offset // 7
        distance = (offset % 7) + 1
        dr, df = QUEEN_DIRECTIONS[direction]
        to_f = from_f + (df * distance)
        to_r = from_r + (dr * distance)
        to_sq = chess.square(to_f, to_r)
        promo = None
        if (from_r == 6 and to_r == 7) or (from_r == 1 and to_r == 0):
            promo = chess.QUEEN
        move = chess.Move(from_sq, to_sq, promotion=promo)

    # Knight moves
    elif offset < 64:
        direction = offset - 56
        dr, df = KNIGHT_DIRECTIONS[direction]
        to_f = from_f + df
        to_r = from_r + dr
        to_sq = chess.square(to_f, to_r)
        move = chess.Move(from_sq, to_sq)

    # Underpromotions
    else:
        up = offset - 64
        base = (up // 3) * 3
        dir = up % 3
        piece = INV_PROMO_BASE[base]

        dir_map = INV_WHITE_DIR if from_r == 6 else INV_BLACK_DIR
        dr, df = dir_map[dir]

        to_f = from_f + df
        to_r = from_r + dr
        to_sq = chess.square(to_f, to_r)

        move = chess.Move(from_sq, to_sq, promotion=piece)

    if move not in board.legal_moves:
        raise ValueError(f"Generated move {move.uci()} is illegal")
    return move


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
