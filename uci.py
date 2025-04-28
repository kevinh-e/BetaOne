# uci.py
"""
UCI for BetaOne
Allows the engine to communicate with UCI GUIs
"""

import os
import sys
import chess
import torch
import time
import threading
import queue
from typing import List, Optional

import config
from network import PolicyValueNet
import utils
from mcts import run_mcts

engine_name = "BetaOne UCI"
engine_author = "Katara S"

model_path = os.path.join(config.SAVE_DIR, "best_model.pth")
if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}", file=sys.stderr)
    sys.exit(1)

print(f"Using device: {config.DEVICE}", file=sys.stderr)
device = config.DEVICE

try:
    model = PolicyValueNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}", file=sys.stderr)
    sys.exit(1)

search_thread: Optional[threading.Thread] = None
stop_event = threading.Event()
result_queue = queue.Queue()


def search_worker(
    board: chess.Board,
    history: List[chess.Board],
    tracker: utils.RepetitionTracker,
    time_limit: Optional[float],
):
    global model, stop_event, result_queue

    start_time = time.time()
    n_simulation = 0
    best_move = None

    try:
        # take max 7 history states
        mcts_hist = history[max(0, len(history) - 7) :]
        best_move, _ = run_mcts(board, model, mcts_hist, tracker)
        if best_move:
            result_queue.put(best_move)
        n_simulation += 1
    except Exception as e:
        print(f"MCTS search error: {e}", file=sys.stderr)
        result_queue.put(None)
        return

    while True:
        if stop_event.is_set():
            print("info string Search stopped by event.")
            break

        elapsed_time = (time.time() - start_time) * 1000
        if time_limit is not None and elapsed_time >= time_limit:
            print(f"info string Time limit reached ({elapsed_time:.0f}ms)")
            break

        try:
            mcts_hist = history[max(0, len(history) - 7) :]
            curr_best_move, policy = run_mcts(board, model, mcts_hist, tracker)
            n_simulation += config.NUM_SIMULATIONS
            if curr_best_move:
                best_move = curr_best_move

            result_queue.put(best_move)

        except Exception as e:
            print(f"Error during subsequent MCTS: {e}", file=sys.stderr)
            break

        # Optional sleep to prevent busy-waiting
        # time.sleep(0.01)

    print(f"info string Search finished ({n_simulation} simulations).")

    # ensure final best move is in queue
    try:
        last_item = result_queue.get_nowait()
        result_queue.put(last_item)
    except queue.Empty:
        result_queue.put(best_move)


def uci_loop():
    """Handles UCI command I/O"""
    global search_thread, stop_event, result_queue

    board = chess.Board()
    tracker = utils.RepetitionTracker()
    tracker.add_board(board)
    history = [board.copy()]

    while True:
        line = sys.stdin.readline().strip()
        if not line:
            continue

        print(f"info string Command received: {line}")
