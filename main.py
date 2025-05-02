# main.py
"""
Main script to run the AlphaZero training pipeline.
Orchestrates self-play, training, and potentially evaluation.
"""

import os
from numpy import ceil
import numpy
import torch
from torch.nn import CosineSimilarity
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
from train import (
    run_selfplay_iteration,
    run_sup_training,
    load_checkpoint,
    save_checkpoint,
)


def setup_directories():
    for d in (config.SAVE_DIR, config.LOG_DIR, config.DATA_DIR):
        os.makedirs(d, exist_ok=True)


def load_or_initialize(model, optimizer, scheduler):
    """
    Load the latest full checkpoint if available; else load best_model weights;
    return start_iter and optionally loaded global_step.
    """
    save_dir = config.SAVE_DIR
    # find numeric checkpoints
    ckpts = glob.glob(os.path.join(save_dir, "checkpoint_iter_*.pth"))
    iters = []
    for f in ckpts:
        name = os.path.basename(f)
        try:
            it = int(name.split("_")[-1].split(".")[0])
            iters.append((it, name))
        except ValueError:
            continue
    if iters:
        # pick latest
        start_iter, ckpt_name = max(iters, key=lambda x: x[0])
        print(f"Loading full checkpoint: {ckpt_name}")
        global_step = load_checkpoint(
            model, optimizer, scheduler, ckpt_name, load_optimizer_state=True
        )
        return start_iter, global_step

    best = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(best):
        print(f"Loading best_model weights: {best}")
        load_checkpoint(
            model,
            optimizer=None,
            scheduler=None,
            filename="best_model.pth",
            load_optimizer_state=False,
        )
        start_iter = 0
        # Optionally load pretrain optimizer state
        pretrain = os.path.join(save_dir, "checkpoint_iter_pretrain.pth")
        if os.path.exists(pretrain):
            print("Loading optimizer/scheduler from pretrain checkpoint")
            load_checkpoint(
                model,
                optimizer,
                scheduler,
                "checkpoint_iter_pretrain.pth",
                load_optimizer_state=True,
            )
        return start_iter, 0

    print("No checkpoint found. Starting from scratch.")
    return 0, 0


def run_self_play_pool(model_path, iteration):
    """Generate self-play data in parallel."""
    num_workers = config.NUM_THREADS
    num_games = max(config.GAMES_MINIMUM, num_workers)
    num_games = int(num_workers * ceil(num_games / num_workers))
    devices = [config.DEVICE] * num_workers

    if config.DEVICE == "cuda" and torch.cuda.device_count() > 1:
        devices = [f"cuda:{i % torch.cuda.device_count()}" for i in range(num_workers)]
        print(f"Assigning self-play devices: {devices}")

    args = [
        (model_path, iteration, i, devices[i % num_workers]) for i in range(num_games)
    ]

    start = time.time()
    if num_workers > 1:
        with mp.Pool(processes=num_workers) as pool:
            list(pool.imap_unordered(run_self_play_worker, args))
    else:
        for a in args:
            run_self_play_worker(a)
    duration = time.time() - start
    return num_games, duration


def run_self_play_worker(args):
    model_path, iteration, game_idx, device_str = args
    device = torch.device(device_str)
    model = PolicyValueNet().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Failed to load weights on {device}: {e}")
        return

    data = run_self_play_game(model, game_id=game_idx)
    if data:
        save_game_data(data, iteration, game_id=game_idx)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    cudnn.benchmark = True
    print("Starting AlphaZero Chess Training...")
    print(
        f"Primary device: {config.DEVICE}, Workers: {config.NUM_WORKERS}, AMP: {config.USE_AMP}"
    )

    setup_directories()
    writer = SummaryWriter(log_dir=config.LOG_DIR)

    model = PolicyValueNet().to(config.DEVICE)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scaler = torch.GradScaler(device=config.DEVICE)
    scheduler = CosineAnnealingLR(optimizer, T_max=1, eta_min=config.LR_MIN)

    start_iter, global_step = load_or_initialize(model, optimizer, scheduler)

    # Supervised pre-training
    if start_iter == 0:
        global_step = run_sup_training(
            model,
            optimizer,
            scheduler,
            scaler,
            writer,
        )
        # reset scheduler after pretrain
        scheduler = CosineAnnealingLR(optimizer, T_max=1, eta_min=config.LR_MIN)

    # Self-play + training loop
    for it in range(start_iter, config.NUM_ITERATIONS):
        print(f"Iteration {it}/{config.NUM_ITERATIONS - 1}")

        model.eval()
        model_path = os.path.join(config.SAVE_DIR, "best_model.pth")
        if not os.path.exists(model_path):
            torch.save(model.state_dict(), model_path)

        iter_dir = os.path.join(config.DATA_DIR, f"iter_{it}")
        os.makedirs(iter_dir, exist_ok=True)

        n_games, sp_time = run_self_play_pool(model_path, it)
        print(f"Self-play: {n_games} games in {sp_time:.2f}s")
        writer.add_scalar("Time/self_play_duration_sec", sp_time, it)
        writer.add_scalar("SelfPlay/num_games", n_games, it)

        model.train()
        train_start = time.time()
        global_step = run_selfplay_iteration(
            model,
            optimizer,
            scheduler,
            scaler,
            writer,
            iteration=it,
            lookback=5,
            start_step=global_step,
        )
        train_time = time.time() - train_start
        print(f"Training: completed in {train_time:.2f}s")
        writer.add_scalar("Time/train_duration_sec", train_time, it)

        save_checkpoint(model, optimizer, it, scheduler, is_best=True)

    print("Training completed.")
    writer.close()


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set or not needed

    main()
