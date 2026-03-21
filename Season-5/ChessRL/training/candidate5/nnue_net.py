"""
NNUE (Efficiently Updatable Neural Network) for Xiangqi position evaluation.

Architecture: Stockfish-style half-KP inspired design
- Per-perspective piece-square features (1260 binary inputs per side)
- Feature: (piece_color, piece_type, square) for each piece on the board
- Two accumulators (one per perspective), concatenated for final evaluation

  Input(1260) -> Linear(1260, 128) [per perspective, shared weights]
  Concat(128, 128) = 256
  -> ClippedReLU -> Linear(256, 32) -> ClippedReLU
  -> Linear(32, 32) -> ClippedReLU -> Linear(32, 1)

~170K parameters total.
"""

import torch
import torch.nn as nn
import numpy as np


# Piece encoding: 1=General, 2=Advisor, 3=Elephant, 4=Horse, 5=Chariot, 6=Cannon, 7=Soldier
# Colors: positive=Red, negative=Black
NUM_PIECE_TYPES = 7
NUM_SQUARES = 90
NUM_COLORS = 2
# Features per perspective: 2 colors * 7 types * 90 squares = 1260
FEATURES_PER_PERSPECTIVE = NUM_COLORS * NUM_PIECE_TYPES * NUM_SQUARES  # 1260

ACCUMULATOR_SIZE = 128
HIDDEN1_SIZE = 32
HIDDEN2_SIZE = 32


class ClippedReLU(nn.Module):
    """ReLU clamped to [0, 1] — standard in NNUE architectures."""
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


class NNUENet(nn.Module):
    """
    NNUE evaluation network for Xiangqi.

    The key insight: the accumulator layer (input -> 128) is shared between
    perspectives and can be incrementally updated when a single piece moves,
    since the input is sparse binary features.
    """

    def __init__(self):
        super().__init__()
        # Accumulator: shared weights, applied per-perspective
        self.accumulator = nn.Linear(FEATURES_PER_PERSPECTIVE, ACCUMULATOR_SIZE)

        # Output layers after concatenating both perspectives
        self.fc1 = nn.Linear(ACCUMULATOR_SIZE * 2, HIDDEN1_SIZE)
        self.fc2 = nn.Linear(HIDDEN1_SIZE, HIDDEN2_SIZE)
        self.fc_out = nn.Linear(HIDDEN2_SIZE, 1)

        self.clipped_relu = ClippedReLU()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, stm_features, nstm_features):
        """
        Forward pass.

        Args:
            stm_features: (batch, 1260) binary features from side-to-move perspective
            nstm_features: (batch, 1260) binary features from not-side-to-move perspective

        Returns:
            (batch, 1) evaluation in [0, 1] from side-to-move perspective
            0.0 = certain loss, 0.5 = draw, 1.0 = certain win
        """
        # Apply shared accumulator to both perspectives
        stm_acc = self.clipped_relu(self.accumulator(stm_features))
        nstm_acc = self.clipped_relu(self.accumulator(nstm_features))

        # Concatenate: [side-to-move, not-side-to-move]
        x = torch.cat([stm_acc, nstm_acc], dim=1)

        x = self.clipped_relu(self.fc1(x))
        x = self.clipped_relu(self.fc2(x))
        x = torch.sigmoid(self.fc_out(x))

        return x


def board_to_nnue_features(board_flat, turn):
    """
    Convert a flat int8[90] board + turn to NNUE feature vectors.

    Feature index for a perspective:
        color_idx * (7 * 90) + (piece_type - 1) * 90 + square

    Where color_idx: 0 = friendly piece, 1 = enemy piece (relative to perspective).

    For side-to-move perspective:
        - friendly = same color as turn
        - enemy = opposite color
    For not-side-to-move perspective:
        - friendly = opposite color
        - enemy = same color
        - board is mirrored (square 89 - sq) to maintain symmetry

    Args:
        board_flat: numpy int8 array of shape (90,) — piece values
        turn: int8, 1=Red, -1=Black (side to move)

    Returns:
        stm_features: numpy float32 array of shape (1260,)
        nstm_features: numpy float32 array of shape (1260,)
    """
    stm = np.zeros(FEATURES_PER_PERSPECTIVE, dtype=np.float32)
    nstm = np.zeros(FEATURES_PER_PERSPECTIVE, dtype=np.float32)

    for sq in range(90):
        piece = board_flat[sq]
        if piece == 0:
            continue

        piece_color = 1 if piece > 0 else -1
        piece_type = abs(piece)  # 1-7
        type_idx = piece_type - 1  # 0-6

        # Side-to-move perspective
        if piece_color == turn:
            color_idx = 0  # friendly
        else:
            color_idx = 1  # enemy
        feat_idx = color_idx * (NUM_PIECE_TYPES * NUM_SQUARES) + type_idx * NUM_SQUARES + sq
        stm[feat_idx] = 1.0

        # Not-side-to-move perspective (mirrored board)
        mirror_sq = 89 - sq
        if piece_color == turn:
            color_idx = 1  # enemy from opponent's view
        else:
            color_idx = 0  # friendly from opponent's view
        feat_idx = color_idx * (NUM_PIECE_TYPES * NUM_SQUARES) + type_idx * NUM_SQUARES + mirror_sq
        nstm[feat_idx] = 1.0

    return stm, nstm


def board_batch_to_nnue_features(boards, turns):
    """
    Vectorized batch conversion of boards to NNUE features.

    Args:
        boards: numpy int8 array (N, 90)
        turns: numpy int8 array (N,)

    Returns:
        stm_features: numpy float32 (N, 1260)
        nstm_features: numpy float32 (N, 1260)
    """
    N = boards.shape[0]
    stm = np.zeros((N, FEATURES_PER_PERSPECTIVE), dtype=np.float32)
    nstm = np.zeros((N, FEATURES_PER_PERSPECTIVE), dtype=np.float32)

    # For each piece type (1-7) and color (+/-)
    for piece_type in range(1, 8):
        type_idx = piece_type - 1

        for color_sign in [1, -1]:
            piece_val = color_sign * piece_type
            # mask: (N, 90) bool where this piece exists
            mask = (boards == piece_val)

            if not mask.any():
                continue

            # Get batch indices and square indices
            batch_idx, sq_idx = np.where(mask)

            # Determine if friendly or enemy relative to turn
            piece_colors = np.full(len(batch_idx), color_sign, dtype=np.int8)
            stm_turns = turns[batch_idx]

            is_friendly = (piece_colors == stm_turns)

            # STM features
            stm_color_idx = np.where(is_friendly, 0, 1)
            stm_feat = stm_color_idx * (NUM_PIECE_TYPES * NUM_SQUARES) + type_idx * NUM_SQUARES + sq_idx
            stm[batch_idx, stm_feat] = 1.0

            # NSTM features (mirrored)
            mirror_sq = 89 - sq_idx
            nstm_color_idx = np.where(is_friendly, 1, 0)  # flipped
            nstm_feat = nstm_color_idx * (NUM_PIECE_TYPES * NUM_SQUARES) + type_idx * NUM_SQUARES + mirror_sq
            nstm[batch_idx, nstm_feat] = 1.0

    return stm, nstm
