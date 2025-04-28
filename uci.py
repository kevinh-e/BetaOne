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
import functools

print = functools.partial(print, flush=True)

engine_name = "BetaOne UCI"
engine_author = "Katara S"

model_path = os.path.join(config.SAVE_DIR, "stable_model.pth")
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

    # print bestmove
    try:
        final = result_queue.get_nowait()
    except queue.Empty:
        final = None

    if final:
        # *this* is where you print bestmove unconditionally
        print(f"bestmove {final.uci()}")
    else:
        # fallback
        fallback = next(iter(board.legal_moves), None)
        if fallback:
            print(f"bestmove {fallback.uci()}")


def reset_board(
    board: chess.Board, tracker: utils.RepetitionTracker, history: List[chess.Board]
):
    board.reset()
    tracker = utils.RepetitionTracker()
    tracker.add_board(board)
    history.clear()
    history = [board.copy()]


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

        # print(f"info string Command received: {line}")

        if line == "uci":
            print(f"id name {engine_name}")
            print(f"id author {engine_author}")

            print("uciok")
        elif line == "isready":
            print("readyok")

        elif line == "ucinewgame":
            reset_board(board, tracker, history)
            print("info string New game started.")

        elif line.startswith("position"):
            parts = line.split()
            idx = parts.index("moves") if "moves" in parts else len(parts)

            # Parse FEN
            if parts[1] == "startpos":
                reset_board(board, tracker, history)
                start_move_idx = 2
            elif parts[1] == "fen":
                fen = " ".join(parts[2:idx])
                try:
                    board.set_fen(fen)
                    tracker = utils.RepetitionTracker()
                    tracker.add_board(board)
                    history = [board.copy()]
                except ValueError:
                    print("info string Error: Invalid FEN string")
                    continue
            else:
                print("info string Error: Invalid position command")

            # parse moves
            if idx < len(parts) and parts[idx] == "moves":
                for uci in parts[idx + 1 :]:
                    try:
                        move = board.parse_uci(uci)
                        if move in board.legal_moves:
                            board.push(move)
                            tracker.add_board(board)
                            history.append(board.copy())
                        else:
                            print(f"info string Error: Invalid move UCI ({uci})")
                            break
                    except ValueError:
                        print(f"info string Error: Invalid move UCI ({uci})")
                        break

            history = history[-8:]
            print(f"info string Position set. FEN: {board.fen()}")

        elif line.startswith("go"):
            # Stop previous search
            if search_thread and search_thread.is_alive():
                print("info string Stopping previous search")
                stop_event.set()
                search_thread.join(timeout=1.0)

            # clear result_queue
            stop_event.clear()
            while not result_queue.empty():
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    break

            # time controls
            parts = line.split()
            movetime = None
            wtime, btime, winc, binc = None, None, 0, 0

            try:
                if "infinite" in parts:
                    movetime = float("inf")
                elif "movetime" in parts:
                    movetime = float(parts[parts.index("movetime") + 1])
                else:
                    if board.turn == chess.WHITE and "wtime" in parts:
                        wtime = float(parts[parts.index("wtime") + 1])
                        if "winc" in parts:
                            winc = float(parts[parts.index("winc") + 1])
                    elif board.turn == chess.BLACK and "btime" in parts:
                        btime = float(parts[parts.index("btime") + 1])
                        if "binc" in parts:
                            binc = float(parts[parts.index("binc") + 1])
            except (ValueError, IndexError):
                print("info string Error parsing time controls")
                continue

            time_limit = None
            if movetime:
                time_limit = movetime * 0.95
            elif wtime is not None and board.turn == chess.WHITE:
                time_limit = max(100.0, (wtime / 30.0) + winc * 0.9)
            elif btime is not None and board.turn == chess.BLACK:
                time_limit = max(100.0, (btime / 30.0) + binc * 0.9)

            if time_limit is None:
                print("info string No time control specified, using default 5 seconds")
                time_limit = 5000.0

            print(f"info string Starting search ({time_limit}ms).")

            # start search in another thread
            search_thread = threading.Thread(
                target=search_worker,
                args=(board.copy(), history.copy(), tracker, time_limit),
                daemon=True,
            )
            search_thread.start()
        elif line == "stop":
            print("info string Received stop.")
            if search_thread and search_thread.is_alive():
                stop_event.set()
                search_thread.join(timeout=2.0)
                if search_thread.is_alive():
                    print("info string Warning: Search thread did not stop quickly.")

                # flush best move
                final_best_move = None
                while not result_queue.empty():
                    final_best_move = result_queue.get_nowait()

                if final_best_move:
                    print(f"bestmove {final_best_move.uci()}")
                else:
                    # use any legal move as fallback
                    if board.legal_moves:
                        fallback_move = next(iter(board.legal_moves))
                        print(
                            f"info string No best move found, using fallback ({fallback_move.uci()})"
                        )
                        print(f"bestmove {fallback_move.uci()}")
                    else:
                        print("info string No best move found and no legal moves!")
                        print("bestmove 0000")
            else:
                print("info string No search running to stop.")

        elif line == "quit":
            print("info string Quitting.")
            if search_thread and search_thread.is_alive():
                stop_event.set()
                search_thread.join(timeout=1.0)
            break

        sys.stdout.flush()


if __name__ == "__main__":
    print(f"{engine_name} by {engine_author} starting UCI...", file=sys.stderr)
    uci_loop()
