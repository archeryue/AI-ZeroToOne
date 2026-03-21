"""
NNUE Training Pipeline for Xiangqi.

Supervised training on 11M human game positions.
Target: predict game outcome (value) from board position.
Loss: MSE between predicted eval and game result.

Usage:
    python train_nnue.py
"""

import os
import sys
import glob
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, SCRIPT_DIR)

from nnue_net import (NNUENet, FEATURES_PER_PERSPECTIVE,
                      NUM_PIECE_TYPES, NUM_SQUARES)

# ─── Hyperparameters ─────────────────────────────────────────────────────────
BATCH_SIZE = 8192
LEARNING_RATE = 1e-3
LR_DROP_EPOCH = 15       # drop LR after this epoch
LR_DROP_FACTOR = 0.1
NUM_EPOCHS = 20
WEIGHT_DECAY = 1e-6
DRAW_SUBSAMPLE = 0.3     # keep 30% of draw positions to reduce noise
VAL_SPLIT = 0.02         # 2% of data for validation
NUM_WORKERS = 4
SAVE_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
DATA_DIR = os.path.join(PROJECT_DIR, "data")

# ─── Dataset ─────────────────────────────────────────────────────────────────

class NNUEDataset(Dataset):
    """
    Dataset storing raw boards/turns/values in memory (~1 GB for 9M positions).
    NNUE features are computed on-the-fly per batch via collate_fn.
    """

    def __init__(self, boards, turns, values):
        """
        Args:
            boards: int8 (N, 90)
            turns: int8 (N,)
            values: float32 (N,) — side-to-move relative
        """
        self.boards = boards
        self.turns = turns
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return self.boards[idx], self.turns[idx], self.values[idx]


def nnue_collate_fn(batch):
    """
    Custom collate that converts a batch of (board, turn, value) tuples
    to (stm_features, nstm_features, values) tensors.
    Feature computation is vectorized over the batch.
    """
    boards = np.stack([b[0] for b in batch])   # (B, 90)
    turns = np.array([b[1] for b in batch])    # (B,)
    values = np.array([b[2] for b in batch])   # (B,)

    B = len(boards)
    stm = np.zeros((B, FEATURES_PER_PERSPECTIVE), dtype=np.float32)
    nstm = np.zeros((B, FEATURES_PER_PERSPECTIVE), dtype=np.float32)

    for piece_type in range(1, 8):
        type_idx = piece_type - 1
        for color_sign in [1, -1]:
            piece_val = color_sign * piece_type
            mask = (boards == piece_val)
            if not mask.any():
                continue

            batch_idx, sq_idx = np.where(mask)
            is_friendly = (color_sign == turns[batch_idx])

            # STM features
            stm_color_idx = np.where(is_friendly, 0, 1)
            stm_feat = stm_color_idx * (NUM_PIECE_TYPES * NUM_SQUARES) + type_idx * NUM_SQUARES + sq_idx
            stm[batch_idx, stm_feat] = 1.0

            # NSTM features (mirrored board)
            mirror_sq = 89 - sq_idx
            nstm_color_idx = np.where(is_friendly, 1, 0)
            nstm_feat = nstm_color_idx * (NUM_PIECE_TYPES * NUM_SQUARES) + type_idx * NUM_SQUARES + mirror_sq
            nstm[batch_idx, nstm_feat] = 1.0

    return (torch.from_numpy(stm),
            torch.from_numpy(nstm),
            torch.from_numpy(values).unsqueeze(1))


def load_all_shards(data_dir, draw_subsample=DRAW_SUBSAMPLE):
    """Load all training data shards and optionally subsample draws."""
    shard_paths = sorted(glob.glob(os.path.join(data_dir, "supervised_training_data_shard*.npz")))
    if not shard_paths:
        raise FileNotFoundError(f"No shards found in {data_dir}")

    print(f"Loading {len(shard_paths)} shards from {data_dir}...")

    all_boards = []
    all_values = []
    all_turns = []

    for path in shard_paths:
        data = np.load(path)
        boards = data['boards']    # (N, 90) int8
        values = data['values']    # (N,) float32
        turns = data['turns']      # (N,) int8

        # Subsample draws
        if draw_subsample < 1.0:
            is_draw = (values == 0.0)
            keep = ~is_draw | (np.random.random(len(values)) < draw_subsample)
            boards = boards[keep]
            values = values[keep]
            turns = turns[keep]

        all_boards.append(boards)
        all_values.append(values)
        all_turns.append(turns)

    boards = np.concatenate(all_boards)
    values = np.concatenate(all_values)
    turns = np.concatenate(all_turns)

    print(f"Loaded {len(values):,} positions")
    print(f"  Red wins: {(values == 1.0).sum():,}")
    print(f"  Black wins: {(values == -1.0).sum():,}")
    print(f"  Draws: {(values == 0.0).sum():,}")

    return boards, values, turns


def convert_values_to_stm(values, turns):
    """
    Convert absolute values (1.0=Red wins) to side-to-move relative [0, 1] targets.
    NNUE predicts from STM perspective: 1.0 = win, 0.5 = draw, 0.0 = loss.
    """
    stm_values = values.copy()
    black_mask = (turns == -1)
    stm_values[black_mask] = -stm_values[black_mask]
    # Map from [-1, 1] to [0, 1]: 0.5 + 0.5 * stm_value
    stm_values = 0.5 + 0.5 * stm_values
    return stm_values


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data (raw boards ~810 MB, fits in RAM)
    boards, values, turns = load_all_shards(DATA_DIR)

    # Convert values to side-to-move relative
    stm_values = convert_values_to_stm(values, turns)
    del values  # free absolute values

    # Train/val split
    N = len(stm_values)
    indices = np.random.permutation(N)
    val_size = int(N * VAL_SPLIT)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}")
    mem_mb = (boards.nbytes + stm_values.nbytes + turns.nbytes) / 1e6
    print(f"Raw data in memory: {mem_mb:.0f} MB")

    train_dataset = NNUEDataset(boards[train_idx], turns[train_idx], stm_values[train_idx])
    val_dataset = NNUEDataset(boards[val_idx], turns[val_idx], stm_values[val_idx])

    # Free full arrays
    del boards, turns, stm_values, indices

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              collate_fn=nnue_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True,
                            collate_fn=nnue_collate_fn)

    # Model
    model = NNUENet().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DROP_EPOCH, gamma=LR_DROP_FACTOR)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    # Estimate training time
    batches_per_epoch = len(train_loader)
    print(f"Batches per epoch: {batches_per_epoch}")

    for epoch in range(1, NUM_EPOCHS + 1):
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

            # Progress every 100 batches
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

                # Sign accuracy on decisive games (win/loss, not draw)
                decisive_mask = ((target > 0.75) | (target < 0.25)).squeeze()
                if decisive_mask.any():
                    pred_win = (pred[decisive_mask] > 0.5).float()
                    target_win = (target[decisive_mask] > 0.5).float()
                    correct_sign += (pred_win == target_win).sum().item()
                    total_decisive += decisive_mask.sum().item()

        val_loss = val_loss_sum / val_count
        sign_acc = correct_sign / total_decisive if total_decisive > 0 else 0.0
        lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
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
            }, os.path.join(SAVE_DIR, "nnue_best.pt"))
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

        # Save periodic checkpoint
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'sign_accuracy': sign_acc,
            }, os.path.join(SAVE_DIR, f"nnue_epoch{epoch}.pt"))

    # Save final
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'sign_accuracy': sign_acc,
    }, os.path.join(SAVE_DIR, "nnue_final.pt"))

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {SAVE_DIR}")


if __name__ == "__main__":
    main()
