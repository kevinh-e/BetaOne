# main.py
"""
Main script to run the AlphaZero training pipeline.
Orchestrates self-play, training, and potentially evaluation.
"""

import os
from numpy import ceil
import math
import torch
import torch.optim as optim
import multiprocessing as mp
import torch.backends.cudnn as cudnn
import config
import glob
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from network import PolicyValueNet
from self_play import run_self_play_game, save_game_data
from train import run_training_iteration, run_pretraining, load_checkpoint


def check_existing_self_play_data(iteration: int) -> bool:
    """Checks if theres already self-play examples for this iteration"""
    data_dir = os.path.join(config.DATA_DIR, f"iter_{iteration}")
    if not os.path.isdir(data_dir):
        return False

    files = glob.glob(os.path.join(data_dir, "game_*.pkl"))
    if not files:
        return False

    return True


def run_self_play_worker(args):
    """Worker function for parallel self-play game generation."""
    model_weights_path, iteration, game_idx = args
    tqdm.write(f"Worker started for game {iteration}-{game_idx}")

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
        tqdm.write(
            f"Worker finished game {iteration}-{game_idx}, saved {len(game_data)} examples."
        )
    else:
        tqdm.write(f"Worker failed for game {iteration}-{game_idx}.")


def main():
    """Main function to execute the training loop."""
    cudnn.benchmark = True
    print("Starting AlphaZero Chess Training...")
    print(f"Using device: {config.DEVICE}")

    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)

    writer = SummaryWriter(log_dir=config.LOG_DIR)

    model = PolicyValueNet().to(config.DEVICE)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # scheduler (self-play)
    steps_per_epoch = math.ceil(config.GAME_BUFFER_SIZE / config.BATCH_SIZE)
    total_steps = config.NUM_ITERATIONS * config.EPOCHS_PER_ITERATION * steps_per_epoch
    scheduler = CosineAnnealingLR(optimizer, 1, eta_min=config.LR_MIN)

    scaler = torch.GradScaler(config.DEVICE)

    start_iter = -1  # will be updated if pretrained exists or checkpoint
    best_model_path = os.path.join(config.SAVE_DIR, "best_model.pth")
    pretrained_path = os.path.join(config.SAVE_DIR, "pretrained.pth")

    if os.path.exists(best_model_path):
        print(f"Loading weights from {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE))
        # Find the latest checkpoint to potentially resume optimizer state and iteration count
        checkpoint_files = sorted(
            glob.glob(os.path.join(config.SAVE_DIR, "checkpoint_iter_*.pth"))
        )
        if checkpoint_files:
            latest_checkpoint = max(
                checkpoint_files, key=lambda f: int(f.split("_")[-1].split(".")[0])
            )
            start_iter = load_checkpoint(model, optimizer, scheduler, latest_checkpoint)
        else:
            print(
                "Loaded 'best_model.pth' weights, but no optimizer checkpoint found. Starting optimizer fresh."
            )
            start_iter = 0
            scheduler = CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=config.LR_MIN
            )

    elif os.path.exists(pretrained_path):
        # load pretrained weights
        try:
            model.load_state_dict(
                torch.load(pretrained_path, map_location=config.DEVICE)
            )
            print("Loaded pretrained weights")
            # skip pretraining
            start_iter = 0
        except Exception as e:
            print(f"Loading pretrained model error: {e}")

    # --- Supervised Pre-Training --
    if start_iter == -1:
        # init scheduler for pretraining
        scheduler = CosineAnnealingLR(
            optimizer, T_max=config.PRETRAINING_T_MAX, eta_min=config.LR_MIN
        )
        run_pretraining(model, optimizer, scheduler, scaler, writer)

        # reset scheduler after pretraining
        scheduler.last_epoch = 1
        start_iter = 0

    # --- Self-Play Training Loop ---
    for iteration in range(start_iter, config.NUM_ITERATIONS):
        print(f"\n{'=' * 20} Iteration {iteration}/{config.NUM_ITERATIONS} {'=' * 20}")

        current_model_path = os.path.join(config.SAVE_DIR, "best_model.pth")
        if not os.path.exists(current_model_path):
            print("Saving initial model weights...")
            torch.save(model.state_dict(), current_model_path)

        skip_sp = check_existing_self_play_data(iteration)

        if not skip_sp:
            print("\n--- Starting Self-Play Phase ---")
            model.eval()

            num_workers = config.NUM_THREADS
            num_games_this_iteration = int(
                num_workers * ceil(config.GAMES_MINIMUM / num_workers)
            )
            worker_args = [
                (current_model_path, iteration, i)
                for i in range(num_games_this_iteration)
            ]

            sp_start = time.time()
            if num_workers > 1:
                try:
                    with mp.Pool(processes=num_workers) as pool:
                        list(
                            tqdm(
                                pool.imap_unordered(run_self_play_worker, worker_args),
                                total=num_games_this_iteration,
                                desc="Self-Play Games",
                            )
                        )
                except Exception as e:
                    print(
                        f"Running self-play sequentially as fallback: Multiprocessing error {e}"
                    )
                    # Fallback to sequential execution if pool fails
                    for args in tqdm(worker_args, desc="Self-Play Games (Sequetial)"):
                        run_self_play_worker(args)
            else:
                print("Running self-play sequentially...")
                for args in tqdm(worker_args, desc="Self-Play Games (Sequetial)"):
                    run_self_play_worker(args)

            sp_duration = time.time() - sp_start
            print(f"--- Self-Play Phase Finished [{sp_duration:.2f}s] ---")
        else:
            print("Skipping self-play...")

        # --- Step 2: Training ---
        print("\n--- Starting Training Phase ---")
        tr_start = time.time()

        run_training_iteration(model, optimizer, scheduler, scaler, iteration, writer)

        tr_duration = time.time() - tr_start
        print(f"--- Training Phase Finished [{tr_duration:.2f}s] ---")

        # --- Tensorboard ---
        writer.add_scalar("Time/train_duration", tr_duration, iteration)

    print("\n===== AlphaZero Chess Training Completed =====")

    writer.close()


if __name__ == "__main__":
    # Set multiprocessing start method globally if needed, especially for CUDA
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set or not needed

    main()
