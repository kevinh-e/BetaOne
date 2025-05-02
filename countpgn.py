# count_pgn_samples.py
import os
import sys
import glob
import math
from tqdm import tqdm
import time

# --- Verify Dependencies and Imports ---
try:
    # Assuming running from the project root where these are accessible
    import config
    import utils  # Needed by PGNDataset.parse
    from train import PGNDataset  # Import the dataset class

    # Import necessary external libs used by PGNDataset/utils
    import torch
    import numpy
    import chess
    import chess.pgn
    import itertools  # Used in PGNDataset.__iter__

    print("Successfully imported necessary modules.")
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("\nPlease ensure config.py, utils.py, and train.py are accessible.")
    print("This script likely needs to be run from the main project directory.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during imports: {e}")
    sys.exit(1)
# -----------------------------------------


def run_counting():
    """Finds PGN files, iterates through PGNDataset, counts samples, and calculates T_max."""
    print("\n--- PGN Sample Counter ---")
    start_time = time.time()

    # 1. Get Configuration
    try:
        pgn_dir = config.PGN_DATA_DIR
        batch_size = config.BATCH_SIZE
        print(f"Using PGN directory: '{pgn_dir}'")
        print(f"Using Batch size: {batch_size}")
    except AttributeError as e:
        print(f"Error accessing configuration value: {e}")
        print("Please ensure config.py defines PGN_DATA_DIR and BATCH_SIZE.")
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
    print(f"Found {len(pgn_files)} PGN files.")

    # 3. Initialize Dataset
    # No max_games limit needed when counting everything
    try:
        # Verbose instantiation
        print("Initializing PGNDataset...")
        dataset = PGNDataset(paths=pgn_files)
        print("PGNDataset initialized.")
    except Exception as e:
        print(f"Error initializing PGNDataset: {e}")
        sys.exit(1)

    # 4. Iterate and Count Samples (Single Worker)
    total_samples = 0
    print("\nCounting samples by iterating through the dataset...")
    print("(This will process all PGN files and may take a significant amount of time)")

    pbar = None  # Initialize pbar to None
    try:
        # Directly iterate the dataset instance.
        # Since this isn't in a DataLoader worker, get_worker_info() in __iter__
        # will return None, causing it to process all files sequentially.
        dataset_iterator = iter(dataset)

        # Setup tqdm progress bar
        pbar = tqdm(
            desc="Counting samples", unit=" samples", smoothing=0.1
        )  # Added smoothing

        for _ in dataset_iterator:
            total_samples += 1
            pbar.update(1)
            # Optional: Add a progress update print every N samples if tqdm is too slow to update
            # if total_samples % 500000 == 0:
            #     print(f"  ... counted {total_samples:,} samples")

    except KeyboardInterrupt:
        print("\nCounting interrupted by user.")
        # Decide if partial count is useful
    except Exception as e:
        print(f"\nAn error occurred during dataset iteration: {e}")
        # Log the full traceback maybe? import traceback; traceback.print_exc()
    finally:
        if pbar:
            pbar.close()  # Ensure tqdm closes properly
            print("\nIteration finished or stopped.")

    # 5. Calculate T_max
    if total_samples == 0:
        print("\nWarning: Counted 0 samples. T_max cannot be calculated.")
        t_max = 0
    else:
        t_max = math.ceil(total_samples / batch_size)

    # 6. Print Results
    end_time = time.time()
    duration = end_time - start_time

    print("\n--- Counting Complete ---")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Total samples counted: {total_samples:,}")  # Added comma formatting
    print(f"Configured batch size: {batch_size}")
    print("-------------------------")
    print(
        f"Calculated T_max (total steps for 1 epoch): {t_max:,}"
    )  # Added comma formatting
    print("-------------------------")

    if t_max > 0:
        print("\nRecommendation:")
        print(f"Set T_max = {t_max} in your CosineAnnealingLR scheduler")
        print("when pretraining for a single epoch on this dataset.")
        print(
            "e.g., In main.py, ensure config.PRETRAINING_T_MAX (or similar) is set to this value."
        )
    elif total_samples > 0:
        print("\nWarning: T_max calculated as 0. Check batch size configuration.")
    else:
        print("\nNo samples found. Please check PGN directory and file contents.")


if __name__ == "__main__":
    run_counting()
