# utils.py
"""
Utility functions for board representation and move handling.
"""

import chess
import torch
import numpy as np
from typing import List, Tuple
from collections import Counter

import config


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


# --- Move Encoding Constants (Standard 8x8x73 scheme) ---
# Queen move directions (indices 0-7)
QUEEN_DIRECTIONS = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
]  # N, NE, E, SE, S, SW, W, NW
# Knight move directions (indices 0-7)
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
# Underpromotion move directions relative to pawn (indices 0-2)
PROMOTION_DIRECTIONS = [
    (-1, 1),
    (0, 1),
    (1, 1),  # Left capture, Straight, Right capture (for White)
]
# Underpromotion piece types (indices 0-2) mapped to plane offset
PROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
# Total number of action planes per square
ACTIONS_PLANES = 56 + 8 + 9  # 73


# --- Repetition Tracker (Revised - No Internal Board) ---
class RepetitionTracker:
    def __init__(self):
        """Initializes the tracker without an internal board."""
        self.counts = Counter()

        # Note: The initial position count must be added manually
        # by the calling code after initialization using add_board().

    def add_board(self, board: chess.Board):
        """Adds the current board state's key to the counts."""
        self.counts[board._transposition_key()] += 1

    def remove_board(self, board: chess.Board):
        """Removes the current board state's key from the counts (if needed)."""
        key = board._transposition_key()
        if self.counts[key] > 0:
            self.counts[key] -= 1
            if self.counts[key] == 0:
                del self.counts[key]

        # else: # Should not happen if add/remove are paired correctly
        #     print(f"Warning: Tried to remove board key {key} with count <= 0.")

    def repetitions(self, board: chess.Board) -> int:
        """
        Returns how many times this board position has occurred *before*
        the current instance being queried.
        """
        key = board._transposition_key()

        # Subtract 1 because the count includes the current instance if it's already added.
        return max(0, self.counts[key] - 1)

    def get_count(self, board: chess.Board) -> int:
        """Returns the raw count for a given board state."""
        return self.counts[board._transposition_key()]

    def reset(self):
        """Resets the tracker"""
        self.counts.clear()


# --- Board Encoding ---
def encode_board(
    board: chess.Board, history: List[chess.Board], tracker: RepetitionTracker
) -> torch.Tensor:
    """
    Encodes the current board state and recent history into a tensor
    following AlphaZero paper conventions (120 channels).

    Args:
        board: The current chess.Board object.
        history: List of the last 8 board states including current (board = history[-1]).
                 Should contain board objects.
        tracker: Repetition tracker object (revised version).

    Returns:
        A PyTorch tensor representing the board state.
        Shape: (120, 8, 8)
    """
    if not history or board != history[-1]:
        # If history is empty or doesn't end with board, this is an issue reconstruct a valid history if possible
        print(
            f"Warning: History mismatch in encode_board. Board FEN: {board.fen()}. History length: {len(history)}"
        )
        if not history:
            history = [board.copy()]
        # If history doesn't end with board, replace last element (less ideal)
        elif board != history[-1]:
            print("Warning: Replacing last history element with current board.")
            history = history[:-1] + [board.copy()]
            history = history[-8:]  # Ensure max length

    if len(history) > 8:
        print(f"Warning: History length > 8 ({len(history)}). Slicing.")
        history = history[-8:]

    encoded = np.zeros(
        (config.INPUT_CHANNELS, config.BOARD_SIZE, config.BOARD_SIZE), dtype=np.float32
    )

    # --- Helper for placing pieces ---
    def encode_piece_plane(
        target_board: chess.Board,
        piece_type: chess.PieceType,
        color: bool,
        plane_idx: int,
    ):
        for square in target_board.pieces(piece_type, color):
            r, f = chess.square_rank(square), chess.square_file(square)
            encoded[plane_idx, r, f] = 1.0

    # --- Encode History (T-7 to T) ---
    num_history_states = len(history)
    # If len=8, start_channel_idx=0. If len=1, start_channel_idx=7*14=98.
    start_channel_idx = (8 - num_history_states) * HISTORY_BLOCK_SIZE

    for i in range(num_history_states):
        hist_board = history[i]

        # 14 channels per history state
        base = start_channel_idx + i * HISTORY_BLOCK_SIZE
        if not (0 <= base < 112):
            print(
                f"!!! ERROR: Calculated base channel index {base} is out of bounds [0, 111] !!!"
            )
            print(
                f"  num_history_states={num_history_states}, i={i}, start_channel_idx={start_channel_idx}"
            )
            continue

        # 1) Piece planes (12 channels)
        for j, (ptype, clr) in enumerate(PIECE_ORDER):
            encode_piece_plane(hist_board, ptype, clr, base + j)

        # 2) Repetition planes (2 channels)
        rep = tracker.repetitions(hist_board)
        encoded[base + 12, :, :] = 1.0 if rep >= 1 else 0.0
        encoded[base + 13, :, :] = (
            1.0 if rep >= 2 else 0.0
        )  # Rep >= 2 means seen thrice before current

    # Player turn (Channel 112)
    encoded[112, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # Castling rights (Channels 113-116)
    encoded[113, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    encoded[114, :, :] = (
        1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    )
    encoded[115, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    encoded[116, :, :] = (
        1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    )

    # Halfmove clock (Channel 117)
    encoded[117, :, :] = float(board.halfmove_clock)

    # Fullmove number (Channel 118)
    encoded[118, :, :] = float(board.fullmove_number)

    # En passant square (Channel 119)
    if board.ep_square is not None:
        rank, file = (
            chess.square_rank(board.ep_square),
            chess.square_file(board.ep_square),
        )
        encoded[119, rank, file] = 1.0

    return torch.from_numpy(encoded)


# --- Move Handling ---
def move_to_index(move: chess.Move) -> int:
    """
    Converts a chess.Move object to a unique integer index (0-4671)
    based on the 8x8x73 action representation.
    """
    from_square = move.from_square
    to_square = move.to_square
    promotion = move.promotion

    from_r, from_f = chess.square_rank(from_square), chess.square_file(from_square)
    to_r, to_f = chess.square_rank(to_square), chess.square_file(to_square)
    delta_r, delta_f = to_r - from_r, to_f - from_f

    # Underpromotion
    if promotion and promotion != chess.QUEEN:
        try:
            if from_r == 6:  # White pawn promotion
                direction_idx = PROMOTION_DIRECTIONS.index((delta_f, delta_r))
            elif from_r == 1:  # Black pawn promotion
                direction_idx = PROMOTION_DIRECTIONS.index((delta_f, -delta_r))
            else:
                raise ValueError("Promotion from invalid rank")

            piece_idx = PROMOTION_PIECES.index(promotion)
            plane_offset = 64 + piece_idx * 3 + direction_idx
            return from_square * ACTIONS_PLANES + plane_offset
        except (ValueError, IndexError):
            raise ValueError(f"Invalid underpromotion move: {move.uci()}")

    # Knight moves
    is_knight_move = (abs(delta_r), abs(delta_f)) in [(1, 2), (2, 1)]
    if is_knight_move:
        try:
            direction_idx = KNIGHT_DIRECTIONS.index((delta_r, delta_f))
            plane_offset = 56 + direction_idx
            return from_square * ACTIONS_PLANES + plane_offset
        except ValueError:
            raise ValueError(
                f"Invalid knight move delta: {(delta_r, delta_f)} for move {move.uci()}"
            )

    # Handle Queen moves including queen promotion
    is_queen_move = abs(delta_r) == abs(delta_f) or delta_r == 0 or delta_f == 0
    if is_queen_move:
        dr = 0 if delta_r == 0 else delta_r // abs(delta_r)
        df = 0 if delta_f == 0 else delta_f // abs(delta_f)
        try:
            direction_idx = QUEEN_DIRECTIONS.index((dr, df))
            distance = max(abs(delta_r), abs(delta_f))
            if distance == 0:
                raise ValueError("Zero distance move")
            if distance > 7:
                raise ValueError("Move distance > 7")
            plane_offset = direction_idx * 7 + (distance - 1)
            return from_square * ACTIONS_PLANES + plane_offset
        except (ValueError, IndexError):
            raise ValueError(
                f"Invalid queen/sliding move: {move.uci()} with delta {(delta_r, delta_f)}"
            )

    raise ValueError(f"Unhandled move type for move: {move.uci()}")


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """
    Converts an integer index (0-4671) back to a chess.Move object.
    Does NOT check for legality.
    """
    if not (0 <= index < config.NUM_ACTIONS):
        raise ValueError(
            f"Index {index} out of valid range [0, {config.NUM_ACTIONS - 1}]"
        )

    from_sq = index // ACTIONS_PLANES
    plane_offset = index % ACTIONS_PLANES

    from_r, from_f = chess.square_rank(from_sq), chess.square_file(from_sq)

    # Decode Queen moves (planes 0-55)
    if plane_offset < 56:
        direction_idx = plane_offset // 7
        distance = (plane_offset % 7) + 1
        dr, df = QUEEN_DIRECTIONS[direction_idx]
        to_r, to_f = from_r + dr * distance, from_f + df * distance

        if not (0 <= to_r <= 7 and 0 <= to_f <= 7):
            raise ValueError(
                f"Index {index} queen move decodes to off-board square ({to_r}, {to_f})"
            )
        to_sq = chess.square(to_f, to_r)

        promo = None
        piece_at_from = board.piece_at(from_sq)
        if piece_at_from and piece_at_from.piece_type == chess.PAWN:
            if (piece_at_from.color == chess.WHITE and from_r == 6 and to_r == 7) or (
                piece_at_from.color == chess.BLACK and from_r == 1 and to_r == 0
            ):
                promo = chess.QUEEN
        return chess.Move(from_sq, to_sq, promotion=promo)

    # Decode Knight moves (planes 56-63)
    elif plane_offset < 64:
        direction_idx = plane_offset - 56
        dr, df = KNIGHT_DIRECTIONS[direction_idx]
        to_r, to_f = from_r + dr, from_f + df

        if not (0 <= to_r <= 7 and 0 <= to_f <= 7):
            raise ValueError(
                f"Index {index} knight move decodes to off-board square ({to_r}, {to_f})"
            )
        to_sq = chess.square(to_f, to_r)
        return chess.Move(from_sq, to_sq)

    # Decode Underpromotions (planes 64-72)
    else:
        underpromo_offset = plane_offset - 64
        piece_idx = underpromo_offset // 3
        direction_idx = underpromo_offset % 3
        promo_piece = PROMOTION_PIECES[piece_idx]

        # Check piece at from_sq before inferring color/rank
        piece_at_from = board.piece_at(from_sq)
        if not piece_at_from or piece_at_from.piece_type != chess.PAWN:
            raise ValueError(
                f"Index {index} implies underpromotion but no pawn at {chess.square_name(from_sq)}"
            )

        if piece_at_from.color == chess.WHITE and from_r == 6:
            df, dr = PROMOTION_DIRECTIONS[direction_idx]
        elif piece_at_from.color == chess.BLACK and from_r == 1:
            df, dr_rel = PROMOTION_DIRECTIONS[direction_idx]
            dr = -dr_rel
        else:
            raise ValueError(
                f"Index {index} implies underpromotion from invalid rank {from_r} for color {piece_at_from.color}"
            )

        to_r, to_f = from_r + dr, from_f + df

        if not (0 <= to_r <= 7 and 0 <= to_f <= 7):
            raise ValueError(
                f"Index {index} underpromotion decodes to off-board square ({to_r}, {to_f})"
            )
        to_sq = chess.square(to_f, to_r)
        return chess.Move(from_sq, to_sq, promotion=promo_piece)


def get_legal_mask(board: chess.Board) -> torch.Tensor:
    """Creates mask for legal moves."""
    mask = torch.zeros(config.NUM_ACTIONS, dtype=torch.bool)
    for move in board.legal_moves:
        try:
            idx = move_to_index(move)
            if 0 <= idx < config.NUM_ACTIONS:
                mask[idx] = True
            else:
                print(
                    f"Warning: move_to_index({move.uci()}) returned out-of-bounds index {idx}"
                )
        except ValueError as e:
            print(f"Warning: Could not get index for legal move {move.uci()}: {e}")
    return mask


def get_game_outcome(board: chess.Board) -> float | None:
    """Outcome from perspective of player who JUST MOVED."""
    if not board.is_game_over(claim_draw=True):
        return None
    result = board.result(claim_draw=True)
    player_who_moved = not board.turn
    if result == "1-0":
        return 1.0 if player_who_moved == chess.WHITE else -1.0
    elif result == "0-1":
        return 1.0 if player_who_moved == chess.BLACK else -1.0
    else:
        return 0.0


def test_move_indexing(board: chess.Board):
    """Tests move indexing consistency."""
    print(f"\nTesting move indexing for FEN: {board.fen()}")
    errors = 0
    legal_moves = list(board.legal_moves)
    print(f"Found {len(legal_moves)} legal moves.")
    indices_seen = set()
    moves_from_indices = {}
    print("Testing move -> index mapping...")
    for move in legal_moves:
        try:
            idx = move_to_index(move)
            if not (0 <= idx < config.NUM_ACTIONS):
                print(
                    f"  ERROR: Move {move.uci()} -> Index {idx} out of bounds [0, {config.NUM_ACTIONS - 1}]"
                )
                errors += 1
                continue
            if idx in indices_seen:
                print(f"  ERROR: Move {move.uci()} -> Index {idx} already seen!")
                for m_other, i_other in moves_from_indices.items():
                    if i_other == idx:
                        print(f"         (Collision with: {m_other.uci()})")
                errors += 1
            indices_seen.add(idx)
            moves_from_indices[move] = idx
        except Exception as e:
            print(f"  EXCEPTION during move_to_index for move {move.uci()}: {e}")
            errors += 1
    print("Testing index -> move mapping (for indices generated from legal moves)...")
    for move, idx in moves_from_indices.items():
        try:
            retrieved_move = index_to_move(idx, board)
            if retrieved_move != move:
                print(
                    f"  ERROR: Index {idx} (from {move.uci()}) -> Retrieved {retrieved_move.uci()} (Mismatch!)"
                )
                errors += 1
            elif retrieved_move not in board.legal_moves:
                print(
                    f"  ERROR: Index {idx} -> Retrieved {retrieved_move.uci()} which is NOT in board.legal_moves?"
                )
                errors += 1
        except Exception as e:
            print(
                f"  EXCEPTION during index_to_move for index {idx} (from move {move.uci()}): {e}"
            )
            errors += 1
    print(f"Testing get_legal_mask consistency...")
    try:
        legal_mask = get_legal_mask(board).numpy()
        mask_indices = set(np.where(legal_mask)[0])
        if indices_seen != mask_indices:
            print(
                f"  WARNING: Indices from legal moves ({len(indices_seen)}) != indices from get_legal_mask ({len(mask_indices)})."
            )
            print(f"    Only in move_to_index: {indices_seen - mask_indices}")
            print(f"    Only in get_legal_mask: {mask_indices - indices_seen}")
            errors += len((indices_seen - mask_indices) | (mask_indices - indices_seen))
        else:
            print("  Mask indices match indices from legal moves.")
    except Exception as e:
        print(f"  EXCEPTION during get_legal_mask test: {e}")
        errors += 1
    print(f"Finished testing. Found {errors} errors.")
    return errors


# Example Usage
if __name__ == "__main__":
    board = chess.Board()
    test_move_indexing(board)
    board = chess.Board("rnbqkbnr/pppp1Ppp/8/8/8/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
    test_move_indexing(board)
    board = chess.Board(
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    )
    test_move_indexing(board)
