"""Supervised pre-training on human games for AlphaZero policy+value network.

Loads sharded training data (board states + actions + values) from parse_games.py,
converts boards to observations on-the-fly using C++ engine, and trains
the policy head (cross-entropy on human moves) and value head (MSE on game outcome).
"""

import os
import sys
import time
import glob
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

# --- Path setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
CHESS_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "ChineseChess", "backend")

try:
    import engine_c as cc
    _USE_CPP = True
except ImportError:
    _USE_CPP = False

sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, CHESS_DIR)

from agents.alphazero.network import AlphaZeroNet
from env.action_space import NUM_ACTIONS

# ------ Hyperparameters ------
NUM_BLOCKS = 5
CHANNELS = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 1024
NUM_EPOCHS = 5
POLICY_WEIGHT = 1.0
VALUE_WEIGHT = 1.0
DATA_DIR = os.path.join(PROJECT_DIR, "data")
SAVE_DIR = os.path.join(SCRIPT_DIR, "candidate4")
# ------------------------------


class ShardedDataset(Dataset):
    """Loads all shards into memory as compact int8 boards + int32 actions.

    Observations are computed on-the-fly in collate_fn to save memory.
    Total memory: 11M × (90 bytes + 4 bytes + 4 bytes + 1 byte) ≈ 1.1 GB
    """

    def __init__(self, data_dir, prefix="supervised_training_data"):
        shard_files = sorted(glob.glob(
            os.path.join(data_dir, f"{prefix}_shard*.npz")))
        print(f"Loading {len(shard_files)} shards...")

        all_boards = []
        all_actions = []
        all_values = []
        all_turns = []

        for sf in shard_files:
            data = np.load(sf)
            all_boards.append(data['boards'])
            all_actions.append(data['actions'])
            all_values.append(data['values'])
            all_turns.append(data['turns'])

        self.boards = np.concatenate(all_boards)    # (N, 90) int8
        self.actions = np.concatenate(all_actions)   # (N,) int32
        self.values = np.concatenate(all_values)     # (N,) float32
        self.turns = np.concatenate(all_turns)       # (N,) int8

        print(f"Loaded {len(self.boards)} positions")

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        return (self.boards[idx], self.turns[idx],
                self.actions[idx], self.values[idx])


def board_to_obs_batch(boards, turns):
    """Convert batch of flat int8[90] boards + turns to (N,15,10,9) observations.

    Uses C++ engine for speed.
    """
    batch_size = len(boards)
    obs = np.zeros((batch_size, 15, 10, 9), dtype=np.float32)

    for i in range(batch_size):
        grid = boards[i].reshape(10, 9)
        turn = int(turns[i])

        if _USE_CPP:
            board = cc.Board(grid.tolist())
            obs[i] = cc.board_to_observation(board, turn)
        else:
            from env.observation import board_to_observation
            from engine.board import Board
            board = Board(grid.tolist())
            obs[i] = board_to_observation(board, turn)

    return obs


def collate_fn(batch):
    """Custom collate: compute observations on-the-fly."""
    boards, turns, actions, values = zip(*batch)
    boards = np.array(boards)
    turns = np.array(turns)
    actions = np.array(actions)
    values = np.array(values, dtype=np.float32)

    obs = board_to_obs_batch(boards, turns)

    return (torch.from_numpy(obs),
            torch.from_numpy(actions).long(),
            torch.from_numpy(values))


def train_epoch(network, optimizer, dataloader, device, epoch):
    network.train()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = 0

    for batch_idx, (obs, actions, values) in enumerate(dataloader):
        obs = obs.to(device)
        actions = actions.to(device)
        values = values.to(device)

        # Create a "full" mask (all actions legal for forward pass)
        # The network needs a mask to compute log_softmax properly
        mask = torch.ones(obs.size(0), NUM_ACTIONS,
                          dtype=torch.bool, device=device)

        log_policy, pred_value = network(obs, mask)

        # Policy loss: cross-entropy on the human move
        policy_loss = F.nll_loss(log_policy, actions)

        # Value loss: MSE on game outcome
        value_loss = F.mse_loss(pred_value, values)

        loss = POLICY_WEIGHT * policy_loss + VALUE_WEIGHT * value_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy
        pred_actions = log_policy.argmax(dim=-1)
        correct = (pred_actions == actions).sum().item()
        total_correct += correct

        total_policy_loss += policy_loss.item() * obs.size(0)
        total_value_loss += value_loss.item() * obs.size(0)
        total_samples += obs.size(0)
        num_batches += 1

        if batch_idx % 100 == 0:
            acc = total_correct / total_samples * 100
            avg_pl = total_policy_loss / total_samples
            avg_vl = total_value_loss / total_samples
            print(f"  Epoch {epoch} [{batch_idx}/{len(dataloader)}] "
                  f"PL: {avg_pl:.4f} VL: {avg_vl:.4f} Acc: {acc:.1f}%")

    avg_pl = total_policy_loss / total_samples
    avg_vl = total_value_loss / total_samples
    acc = total_correct / total_samples * 100
    return avg_pl, avg_vl, acc


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Supervised Pre-training ===")
    print(f"Device: {device}")
    print(f"C++ engine: {_USE_CPP}")

    # Load data
    dataset = ShardedDataset(DATA_DIR)
    n_total = len(dataset)
    n_val = min(50_000, n_total // 20)
    n_train = n_total - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=0, pin_memory=True)

    print(f"Train: {n_train}, Val: {n_val}")
    print(f"Batches per epoch: {len(train_loader)}")

    # Model
    network = AlphaZeroNet(num_blocks=NUM_BLOCKS, channels=CHANNELS).to(device)
    num_params = sum(p.numel() for p in network.parameters())
    print(f"Model: {NUM_BLOCKS} blocks, {CHANNELS} channels, "
          f"{num_params:,} params")

    optimizer = Adam(network.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Resume from checkpoint if available
    start_epoch = 1
    best_val_pl = float('inf')
    checkpoint_path = os.path.join(SAVE_DIR, "pretrain_checkpoint.pt")
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        network.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
        best_val_pl = ckpt.get('val_pl', float('inf'))
        # Advance scheduler to correct position
        for _ in range(ckpt['epoch']):
            scheduler.step()
        print(f"Resumed from epoch {ckpt['epoch']} "
              f"(val_pl={ckpt['val_pl']:.4f}, val_acc={ckpt['val_acc']:.1f}%)")

    print(f"\nTraining epochs {start_epoch}-{start_epoch + NUM_EPOCHS - 1}, "
          f"batch_size={BATCH_SIZE}, lr={LR}")
    print()

    start_time = time.time()

    for epoch in range(start_epoch, start_epoch + NUM_EPOCHS):
        epoch_start = time.time()

        # Train
        train_pl, train_vl, train_acc = train_epoch(
            network, optimizer, train_loader, device, epoch)

        # Validate
        network.eval()
        val_policy_loss = 0.0
        val_value_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for obs, actions, values in val_loader:
                obs = obs.to(device)
                actions = actions.to(device)
                values = values.to(device)
                mask = torch.ones(obs.size(0), NUM_ACTIONS,
                                  dtype=torch.bool, device=device)
                log_policy, pred_value = network(obs, mask)
                val_policy_loss += F.nll_loss(
                    log_policy, actions, reduction='sum').item()
                val_value_loss += F.mse_loss(
                    pred_value, values, reduction='sum').item()
                val_correct += (log_policy.argmax(-1) == actions).sum().item()
                val_total += obs.size(0)

        val_pl = val_policy_loss / val_total
        val_vl = val_value_loss / val_total
        val_acc = val_correct / val_total * 100

        scheduler.step()
        epoch_time = time.time() - epoch_start

        end_epoch = start_epoch + NUM_EPOCHS - 1
        print(f"\nEpoch {epoch}/{end_epoch} ({epoch_time:.0f}s) | "
              f"Train PL: {train_pl:.4f} VL: {train_vl:.4f} Acc: {train_acc:.1f}% | "
              f"Val PL: {val_pl:.4f} VL: {val_vl:.4f} Acc: {val_acc:.1f}%")

        # Save best
        if val_pl < best_val_pl:
            best_val_pl = val_pl
            save_path = os.path.join(SAVE_DIR, "az_pretrained.pt")
            torch.save(network.state_dict(), save_path)
            print(f"  >> Best model saved (val_pl={val_pl:.4f})")

        # Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state': network.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'train_pl': train_pl,
            'val_pl': val_pl,
            'val_acc': val_acc,
        }, os.path.join(SAVE_DIR, "pretrain_checkpoint.pt"))

    total_time = time.time() - start_time
    print(f"\nPre-training complete! {total_time/60:.1f} min")
    print(f"Best val policy loss: {best_val_pl:.4f}")
    print(f"Saved to: {os.path.join(SAVE_DIR, 'az_pretrained.pt')}")


if __name__ == "__main__":
    main()
