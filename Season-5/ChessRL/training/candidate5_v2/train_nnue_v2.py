"""
NNUE v2 Training with TD(lambda) + Binary Cross-Entropy.

Pipeline:
  1. Load self-play data with search scores (from gen_td_data.py)
  2. Compute TD(lambda) targets by blending search scores with future values
  3. Train with BCE loss instead of MSE

TD(lambda) target computation:
  For each game, process positions backward from the end.
  target_T = game_outcome (final position)
  target_t = (1 - lambda) * search_score_t + lambda * target_{t+1}

  lambda=0: pure search score (one-step bootstrap)
  lambda=1: pure Monte Carlo (game outcome propagated to all positions)
  lambda=0.7-0.9: blend that propagates credit backward

Usage:
    python train_nnue_v2.py [--td_lambda 0.8] [--epochs 30]
"""

import os
import sys
import glob
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from nnue_net_v2 import (NNUENetV2, FEATURES_PER_PERSPECTIVE,
                          SQ_TO_FEAT, board_batch_to_features_v2)

# ─── Hyperparameters ─────────────────────────────────────────────────────────
BATCH_SIZE = 4096
LEARNING_RATE = 5e-4
LR_DROP_EPOCH = 20
LR_DROP_FACTOR = 0.1
NUM_EPOCHS = 30
WEIGHT_DECAY = 1e-5
TD_LAMBDA = 0.8
VAL_SPLIT = 0.02
NUM_WORKERS = 4
SAVE_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
DATA_DIR = os.path.join(SCRIPT_DIR, "td_data")


# ─── TD(lambda) target computation ──────────────────────────────────────────

def compute_td_targets(scores, game_ids, turns, td_lambda):
    """
    Compute TD(lambda) targets from search scores.

    Args:
        scores: float32 (N,) — Red-relative search scores in [-1, 1]
        game_ids: int32 (N,) — game ID for each position
        turns: int8 (N,) — side to move (1=Red, -1=Black)
        td_lambda: float — TD lambda parameter

    Returns:
        targets: float32 (N,) — STM-relative TD targets in [0, 1]
    """
    targets = np.zeros(len(scores), dtype=np.float32)
    unique_games = np.unique(game_ids)

    for gid in unique_games:
        mask = game_ids == gid
        idx = np.where(mask)[0]

        if len(idx) == 0:
            continue

        game_scores = scores[idx]  # Red-relative [-1, 1]
        game_turns = turns[idx]

        # Convert to STM-relative [0, 1] for TD computation
        stm_scores = np.zeros(len(game_scores), dtype=np.float32)
        for i in range(len(game_scores)):
            if game_turns[i] == 1:  # Red STM
                stm_scores[i] = 0.5 + 0.5 * game_scores[i]
            else:  # Black STM
                stm_scores[i] = 0.5 - 0.5 * game_scores[i]

        # TD(lambda) backward pass
        # target_T = stm_score_T (last position uses its own search score)
        # target_t = (1-lambda)*stm_score_t + lambda*( 1 - target_{t+1} )
        # Note: 1 - target_{t+1} because the NEXT position is from opponent's view
        game_targets = np.zeros(len(game_scores), dtype=np.float32)
        game_targets[-1] = stm_scores[-1]

        for t in range(len(game_scores) - 2, -1, -1):
            # Next position's target is from opponent's perspective, so flip it
            next_target_from_my_view = 1.0 - game_targets[t + 1]
            game_targets[t] = (1.0 - td_lambda) * stm_scores[t] + td_lambda * next_target_from_my_view

        targets[idx] = game_targets

    # Clamp to valid range
    targets = np.clip(targets, 0.001, 0.999)
    return targets


# ─── Dataset ─────────────────────────────────────────────────────────────────

class TDDataset(Dataset):
    """Dataset with boards, turns, and TD targets."""

    def __init__(self, boards, turns, targets):
        self.boards = boards    # int8 (N, 90)
        self.turns = turns      # int8 (N,)
        self.targets = targets  # float32 (N,)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.boards[idx], self.turns[idx], self.targets[idx]


def td_collate_fn(batch):
    """Custom collate using v2 feature extraction."""
    boards = np.stack([b[0] for b in batch])
    turns = np.array([b[1] for b in batch])
    targets = np.array([b[2] for b in batch])

    stm, nstm = board_batch_to_features_v2(boards, turns)

    return (torch.from_numpy(stm),
            torch.from_numpy(nstm),
            torch.from_numpy(targets).unsqueeze(1))


# ─── Data loading ────────────────────────────────────────────────────────────

def load_td_data(data_dir, td_lambda):
    """Load TD data shards and compute TD(lambda) targets."""
    shard_paths = sorted(glob.glob(os.path.join(data_dir, "td_shard_*.npz")))
    if not shard_paths:
        raise FileNotFoundError(f"No TD data shards found in {data_dir}")

    print(f"Loading {len(shard_paths)} shards from {data_dir}...")

    all_boards = []
    all_turns = []
    all_scores = []
    all_game_ids = []
    game_id_offset = 0

    for path in shard_paths:
        data = np.load(path)
        all_boards.append(data['boards'])
        all_turns.append(data['turns'])
        all_scores.append(data['scores'])
        # Offset game IDs so they're unique across shards
        gids = data['game_ids'] + game_id_offset
        all_game_ids.append(gids)
        game_id_offset = gids.max() + 1

    boards = np.concatenate(all_boards)
    turns = np.concatenate(all_turns)
    scores = np.concatenate(all_scores)
    game_ids = np.concatenate(all_game_ids)

    print(f"Loaded {len(scores):,} positions from {game_id_offset} games")
    print(f"Score stats: mean={scores.mean():.3f} std={scores.std():.3f}")

    # Compute TD targets
    print(f"Computing TD(lambda={td_lambda}) targets...")
    t0 = time.time()
    targets = compute_td_targets(scores, game_ids, turns, td_lambda)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Target stats: mean={targets.mean():.3f} std={targets.std():.3f}")

    return boards, turns, targets


# ─── Training ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--td_lambda', type=float, default=TD_LAMBDA)
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--save_dir', type=str, default=SAVE_DIR)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"TD(lambda): {args.td_lambda}")

    # Load data
    boards, turns, targets = load_td_data(args.data_dir, args.td_lambda)

    # Train/val split
    N = len(targets)
    indices = np.random.permutation(N)
    val_size = int(N * VAL_SPLIT)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}")

    train_dataset = TDDataset(boards[train_idx], turns[train_idx], targets[train_idx])
    val_dataset = TDDataset(boards[val_idx], turns[val_idx], targets[val_idx])
    del boards, turns, targets, indices

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              collate_fn=td_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            collate_fn=td_collate_fn)

    # Model
    model = NNUENetV2().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DROP_EPOCH, gamma=LR_DROP_FACTOR)

    # BCE loss (targets are in [0.001, 0.999])
    criterion = nn.BCELoss()

    best_val_loss = float('inf')
    batches_per_epoch = len(train_loader)
    print(f"Batches per epoch: {batches_per_epoch}")

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        t_start = time.time()

        for batch_idx, (stm, nstm, target) in enumerate(train_loader):
            stm = stm.to(device, non_blocking=True)
            nstm = nstm.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            pred = model(stm, nstm)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(target)
            train_count += len(target)

            if batch_idx > 0 and batch_idx % 100 == 0:
                elapsed = time.time() - t_start
                speed = train_count / elapsed
                eta = (len(train_dataset) - train_count) / speed
                print(f"  [{batch_idx}/{batches_per_epoch}] "
                      f"loss={train_loss_sum/train_count:.4f} "
                      f"speed={speed:.0f} pos/s "
                      f"ETA={eta:.0f}s", flush=True)

        train_loss = train_loss_sum / train_count
        train_time = time.time() - t_start

        # ── Validate ──
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        correct_sign = 0
        total_decisive = 0

        with torch.no_grad():
            for stm, nstm, target in val_loader:
                stm = stm.to(device, non_blocking=True)
                nstm = nstm.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                pred = model(stm, nstm)
                loss = criterion(pred, target)

                val_loss_sum += loss.item() * len(target)
                val_count += len(target)

                # Sign accuracy on decisive positions
                decisive_mask = ((target > 0.7) | (target < 0.3)).squeeze()
                if decisive_mask.any():
                    pred_win = (pred[decisive_mask] > 0.5).float()
                    target_win = (target[decisive_mask] > 0.5).float()
                    correct_sign += (pred_win == target_win).sum().item()
                    total_decisive += decisive_mask.sum().item()

        val_loss = val_loss_sum / val_count
        sign_acc = correct_sign / total_decisive if total_decisive > 0 else 0.0
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"Train BCE: {train_loss:.4f} | "
              f"Val BCE: {val_loss:.4f} | "
              f"Sign Acc: {sign_acc:.3f} | "
              f"LR: {lr:.1e} | "
              f"Time: {train_time:.0f}s")

        scheduler.step()

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'sign_accuracy': sign_acc,
                'param_count': param_count,
                'td_lambda': args.td_lambda,
            }, os.path.join(args.save_dir, "nnue_v2_best.pt"))
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

        # Periodic
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'sign_accuracy': sign_acc,
                'td_lambda': args.td_lambda,
            }, os.path.join(args.save_dir, f"nnue_v2_epoch{epoch}.pt"))

    # Final
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'sign_accuracy': sign_acc,
        'td_lambda': args.td_lambda,
    }, os.path.join(args.save_dir, "nnue_v2_final.pt"))

    print(f"\nTraining complete. Best val BCE: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {args.save_dir}")


if __name__ == "__main__":
    main()
