# main.py
"""
Main script to run the AlphaZero training pipeline.
Orchestrates self-play, training, and potentially evaluation.
"""

import os
import torch
import torch.optim as optim
import multiprocessing as mp

import config
import glob
from network import PolicyValueNet
from self_play import run_self_play_game, save_game_data
from train import run_training_iteration, load_checkpoint


def run_self_play_worker(args):
    """Worker function for parallel self-play game generation."""
    model_weights_path, iteration, game_idx = args
    print(f"Worker started for game {iteration}-{game_idx}")

    model = PolicyValueNet().to(config.DEVICE)
    # Load latest model
    try:
        model.load_state_dict(
            torch.load(model_weights_path, map_location=config.DEVICE)
        )
        model.eval()
    except Exception as e:
        print(f"Error loading model weights in worker {game_idx}: {e}")
        return

    # Run a single game
    game_data = run_self_play_game(model, game_id=game_idx)

    # Save the generated data
    if game_data:
        save_game_data(game_data, iteration, game_id=game_idx)
        print(
            f"Worker finished game {iteration}-{game_idx}, saved {len(game_data)} examples."
        )
    else:
        print(f"Worker failed for game {iteration}-{game_idx}.")


def main():
    """Main function to execute the training loop."""
    print("Starting AlphaZero Chess Training...")
    print(f"Using device: {config.DEVICE}")

    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)

    model = PolicyValueNet().to(config.DEVICE)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Try loading the 'best_model.pth' first, then specific checkpoints
    best_model_path = os.path.join(config.SAVE_DIR, "best_model.pth")
    start_iter = 0
    if os.path.exists(best_model_path):
        try:
            print(f"Loading weights from {best_model_path}")
            model.load_state_dict(
                torch.load(best_model_path, map_location=config.DEVICE)
            )
            # Find the latest checkpoint to potentially resume optimizer state and iteration count
            checkpoint_files = sorted(
                glob.glob(os.path.join(config.SAVE_DIR, "checkpoint_iter_*.pth"))
            )
            if checkpoint_files:
                latest_checkpoint = checkpoint_files[-1]
                start_iter = load_checkpoint(model, optimizer, latest_checkpoint)
            else:
                print(
                    "Loaded 'best_model.pth' weights, but no optimizer checkpoint found. Starting optimizer fresh."
                )
        except Exception as e:
            print(f"Error loading best model weights: {e}. Starting fresh.")
            # Fallback restart
            start_iter = 0
    else:
        print("No best_model.pth found. Starting training from scratch.")

    # --- Main Training Loop ---
    for iteration in range(start_iter, config.NUM_ITERATIONS):
        print(f"\n{'=' * 20} Iteration {iteration}/{config.NUM_ITERATIONS} {'=' * 20}")

        # --- Self-Play ---
        print("\n--- Starting Self-Play Phase ---")
        model.eval()

        current_model_path = os.path.join(config.SAVE_DIR, "best_model.pth")
        if not os.path.exists(current_model_path):
            # save if no weights
            print("Saving initial model weights...")
            torch.save(model.state_dict(), current_model_path)

        # Determine number of games to play in this iteration
        # This could be fixed or adaptive
        num_games_this_iteration = 25

        worker_args = [
            (current_model_path, iteration, i) for i in range(num_games_this_iteration)
        ]

        # Use multiprocessing pool for parallel game generation
        num_workers = max(1, mp.cpu_count() // 2)
        print(
            f"Running {num_games_this_iteration} self-play games using {num_workers} workers..."
        )

        # Clear old data for this iteration if necessary (optional)
        # shutil.rmtree(os.path.join(config.DATA_DIR, f"iter_{iteration}"), ignore_errors=True)

        if num_workers > 1:
            try:
                # Set start method for multiprocessing if needed (e.g., 'spawn' for CUDA compatibility)
                mp.set_start_method(
                    "spawn", force=True
                )  # Uncomment if facing CUDA issues with fork
                with mp.Pool(processes=num_workers) as pool:
                    pool.map(run_self_play_worker, worker_args)
            except Exception as e:
                print(f"Multiprocessing pool error: {e}")
                print("Running self-play sequentially as fallback.")
                # Fallback to sequential execution if pool fails
                for args in worker_args:
                    run_self_play_worker(args)
        else:
            print("Running self-play sequentially...")
            for args in worker_args:
                run_self_play_worker(args)

        print("--- Self-Play Phase Finished ---")

        # --- Step 2: Training ---
        print("\n--- Starting Training Phase ---")
        run_training_iteration(model, optimizer, iteration)
        print("--- Training Phase Finished ---")

        # Optional Step 3: Evaluation against a baseline or previous checkpoint
        # (Not implemented in this stub)

    print("\n===== AlphaZero Chess Training Completed =====")


if __name__ == "__main__":
    # Set multiprocessing start method globally if needed, especially for CUDA
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass  # Already set or not needed

    main()
