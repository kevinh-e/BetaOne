# count_pgn_samples_mp.py
import os
import sys
import glob
import math
import time
import chess.pgn
import numpy as np  # Needed for parsing logic
import multiprocessing as mp
from tqdm import tqdm
import traceback  # For detailed error logging

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[
        logging.StreamHandler(),  # console
        logging.FileHandler("pgn_errors.log"),  # file
    ],
)

# --- Verify Dependencies and Imports ---
try:
    # Assuming running from the project root where these are accessible
    import config
    import utils  # Needed by parsing logic inside worker

    # Import necessary external libs used by parsing logic
    import torch  # May be needed by utils.encode_board
    # PGNDataset class itself is not strictly needed, just its parsing logic

    print("Successfully imported necessary modules.")
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("\nPlease ensure config.py and utils.py are accessible.")
    print("This script likely needs to be run from the main project directory.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during imports: {e}")
    sys.exit(1)
# -----------------------------------------


# Define the worker function BEFORE it's used by the Pool
def count_samples_in_file(filepath: str) -> int:
    """
    Parses a single PGN file and counts valid training samples.
    Logs start, errors, and finish (with duration and count).
    """
    local_sample_count = 0
    start_time = time.time()
    logging.info(f"START  parsing '{os.path.basename(filepath)}'")
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as pgn_file:
            read_game = chess.pgn.read_game
            while True:
                game = read_game(pgn_file)
                if game is None:
                    break

                # Skip games without a valid result
                outcome = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}.get(
                    game.headers.get("Result", "*")
                )
                if outcome is None:
                    continue

                # Walk through moves, count only if encode_board succeeds (lengthy)
                #
                # board = game.board()
                # history = [board.copy()]
                # tracker = utils.RepetitionTracker()
                # tracker.add_board(board)
                #
                # for node in game.mainline():
                #     move = node.move
                #     if move is None:
                #         continue
                #     try:
                #         # core validation
                #         _ = utils.encode_board(board, history[-8:], tracker)
                #         local_sample_count += 1
                #     except Exception:
                #         # skip invalid states silently
                #         pass
                #
                #     # advance board
                #     board.push(move)
                #     tracker.add_board(board)
                #     history.append(board.copy())
                #     if len(history) > 8:
                #         history.pop(0)

                # count moves, assume theyre valid lol
                for node in game.mainline():
                    move = node.move
                    if move is None:
                        continue
                    local_sample_count += 1

    except Exception as e:
        logging.error(
            f"ERROR parsing '{os.path.basename(filepath)}': {e}",
            exc_info=True,
        )
    finally:
        duration = time.time() - start_time
        logging.info(
            f" DONE  '{os.path.basename(filepath)}' — "
            f"{local_sample_count:,} samples in {duration:.2f}s"
        )

    return local_sample_count


def run_counting_parallel():
    """Finds PGN files, counts samples in parallel, aggregates, calculates T_max."""
    print("\n--- PGN Sample Counter (Parallel) ---")
    start_time = time.time()

    # 1. Get Configuration
    try:
        pgn_dir = config.PGN_DATA_DIR
        batch_size = config.BATCH_SIZE
        # Use NUM_WORKERS from config if available, else default to CPU count
        # Ensure config defines NUM_WORKERS or handle absence
        num_workers = getattr(config, "NUM_WORKERS", os.cpu_count())
        num_workers = max(1, num_workers)  # Ensure at least 1 worker

        print(f"Using PGN directory: '{pgn_dir}'")
        print(f"Using Batch size: {batch_size}")
        print(f"Attempting to use {num_workers} worker processes.")
    except AttributeError as e:
        print(f"Error accessing configuration value from config.py: {e}")
        sys.exit(1)

    if not os.path.isdir(pgn_dir):
        print(f"Error: PGN directory not found at '{pgn_dir}'")
        sys.exit(1)

    # 2. Find PGN Files
    print("Searching for PGN files recursively...")
    pgn_files = glob.glob(os.path.join(pgn_dir, "**", "*.pgn"), recursive=True)
    if not pgn_files:
        print(f"Error: No .pgn files found in '{pgn_dir}' or its subdirectories.")
        sys.exit(1)
    num_files = len(pgn_files)
    print(f"Found {num_files} PGN files.")

    # Limit workers if fewer files than workers
    num_workers = min(num_workers, num_files)
    if num_workers < getattr(config, "NUM_WORKERS", os.cpu_count()):
        print(f"Adjusted worker count to {num_workers} due to number of files.")

    # 3. Use multiprocessing Pool
    total_samples = 0
    processed_files = 0
    print(f"\nLaunching {num_workers} workers to count samples...")

    results_iterator = None
    pbar = None
    pool = None

    try:
        # Create the pool of worker processes
        pool = mp.Pool(processes=num_workers)

        # Use imap_unordered for potentially better progress reporting with tqdm
        # It yields results as they complete, good if file processing times vary
        results_iterator = pool.imap_unordered(count_samples_in_file, pgn_files)

        # Setup tqdm progress bar based on the number of files (tasks)
        pbar = tqdm(
            results_iterator, total=num_files, desc="Processing files", unit=" file"
        )

        # Process results as they come in from workers
        for file_sample_count in pbar:
            total_samples += file_sample_count
            processed_files += 1
            # Optional: Update postfix to show running sample count
            pbar.set_postfix({"Samples": f"{total_samples:,}"})

        # Close the pool and wait for all workers to finish
        pool.close()
        pool.join()

    except KeyboardInterrupt:
        print("\nCounting interrupted by user.")
        if pool:
            pool.terminate()  # Forcefully stop worker processes
            pool.join()
    except Exception as e:
        print(f"\nAn error occurred during parallel processing: {e}")
        traceback.print_exc()  # Print full traceback for unexpected errors
        if pool:
            pool.terminate()
            pool.join()
    finally:
        if pbar:
            pbar.close()  # Ensure tqdm closes properly
        # Ensure pool is definitely closed and joined if an error occurred mid-processing
        if pool and results_iterator:
            try:
                pool.close()
                pool.join()
            except Exception:
                pass  # Ignore errors during cleanup

        print("\nParallel processing finished or stopped.")
        print(f"Successfully processed {processed_files}/{num_files} files.")

    # 4. Calculate T_max
    if total_samples == 0:
        print("\nWarning: Counted 0 samples. T_max cannot be calculated.")
        t_max = 0
    else:
        # Ensure batch_size is at least 1 to avoid division by zero
        batch_size = max(1, batch_size)
        t_max = math.ceil(total_samples / batch_size)

    # 5. Print Results
    end_time = time.time()
    duration = end_time - start_time

    print("\n--- Counting Complete ---")
    print(f"Duration: {duration:.2f} seconds ({num_workers} workers)")
    print(f"Total samples counted: {total_samples:,}")
    print(f"Configured batch size: {batch_size}")
    print("-------------------------")
    print(f"Calculated T_max (total steps for 1 epoch): {t_max:,}")
    print("-------------------------")

    if t_max > 0:
        print("\nRecommendation:")
        print(f"Set T_max = {t_max} in your CosineAnnealingLR scheduler")
        print("when pretraining for a single epoch on this dataset.")
    elif total_samples > 0:
        print("\nWarning: T_max calculated as 0. Check batch size configuration.")
    else:
        print(
            "\nNo samples found or processed successfully. Please check PGN files and potential errors."
        )


if __name__ == "__main__":
    # Set start method once here if needed, recommended for compatibility
    # Use 'spawn' generally, unless you have specific reasons for 'fork' (Linux default)
    # and know CUDA is not involved in worker processes.
    try:
        current_method = mp.get_start_method(allow_none=True)
        # Set to 'spawn' if not already set, for broader compatibility
        # On Linux, 'fork' might be faster if it works, but 'spawn' is safer
        if current_method is None:
            mp.set_start_method("spawn")
            print("Set multiprocessing start method to 'spawn'.")
        elif current_method != "spawn":
            # If already set to something else (like 'fork'), maybe leave it?
            # Or force 'spawn' if needed: mp.set_start_method("spawn", force=True)
            print(
                f"Note: Multiprocessing start method already set to '{current_method}'."
            )

    except RuntimeError as e:
        print(f"Note: Could not set multiprocessing start method: {e}")
        pass  # Allow it to continue if already set or fails

    run_counting_parallel()
