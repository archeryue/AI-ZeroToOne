"""
NNUE v2 for Xiangqi — 692 features with piece-aware square mapping.

Key changes from v1:
- 692 features instead of 1260 (pieces mapped only to reachable squares)
- Board always normalized so perspective's pieces are at bottom (rows 5-9)
- This makes features color-invariant

Feature budget per color (friendly or enemy):
  General:  9 squares (palace 3x3)
  Advisor:  5 squares (palace diagonals)
  Elephant: 7 squares (own-half specific points)
  Horse:    90 squares (any)
  Chariot:  90 squares (any)
  Cannon:   90 squares (any)
  Soldier:  55 squares (own-half restricted + opponent-half full)
  Total:    346 per color, 692 per perspective

Architecture (same structure, smaller input):
  Input(692) -> Accumulator(692, 128) [shared, per perspective]
  Concat(128, 128) = 256
  -> ClippedReLU -> FC(256, 32) -> ClippedReLU
  -> FC(32, 32) -> ClippedReLU -> FC(32, 1) -> Sigmoid

~90K parameters.
"""

import torch
import torch.nn as nn
import numpy as np


# ─── Piece types (matching engine_c/xiangqi.h) ──────────────────────────────
GENERAL = 1
ADVISOR = 2
ELEPHANT = 3
HORSE = 4
CHARIOT = 5
CANNON = 6
SOLDIER = 7

NUM_PIECE_TYPES = 7
NUM_SQUARES = 90

# ─── Reachable squares for each piece type ───────────────────────────────────
# All defined from "friendly at bottom" perspective (friendly = rows 5-9).
# Squares are flat indices: row * 9 + col.

def _palace_squares(row_start):
    """3x3 palace squares starting at row_start."""
    return [row_start * 9 + c for row_start in range(row_start, row_start + 3)
            for c in range(3, 6)]

def _advisor_squares(row_start):
    """5 diagonal positions in palace."""
    r0 = row_start
    return [r0 * 9 + 3, r0 * 9 + 5,
            (r0 + 1) * 9 + 4,
            (r0 + 2) * 9 + 3, (r0 + 2) * 9 + 5]

# Friendly pieces at bottom (rows 5-9)
FRIENDLY_GENERAL_SQ = sorted([r * 9 + c for r in range(7, 10) for c in range(3, 6)])  # 9
FRIENDLY_ADVISOR_SQ = sorted([7*9+3, 7*9+5, 8*9+4, 9*9+3, 9*9+5])  # 5
FRIENDLY_ELEPHANT_SQ = sorted([5*9+2, 5*9+6, 7*9+0, 7*9+4, 7*9+8, 9*9+2, 9*9+6])  # 7
FRIENDLY_SOLDIER_SQ = sorted(
    [r * 9 + c for r in range(5, 7) for c in range(0, 9, 2)] +  # own half: 2 rows x 5 cols = 10
    [r * 9 + c for r in range(0, 5) for c in range(9)]          # crossed river: 5 rows x 9 cols = 45
)  # 55
ALL_SQUARES = list(range(90))

# Enemy pieces at top (rows 0-4)
ENEMY_GENERAL_SQ = sorted([r * 9 + c for r in range(0, 3) for c in range(3, 6)])  # 9
ENEMY_ADVISOR_SQ = sorted([0*9+3, 0*9+5, 1*9+4, 2*9+3, 2*9+5])  # 5
ENEMY_ELEPHANT_SQ = sorted([0*9+2, 0*9+6, 2*9+0, 2*9+4, 2*9+8, 4*9+2, 4*9+6])  # 7
ENEMY_SOLDIER_SQ = sorted(
    [r * 9 + c for r in range(3, 5) for c in range(0, 9, 2)] +  # own half: 2 rows x 5 cols = 10
    [r * 9 + c for r in range(5, 10) for c in range(9)]         # crossed river: 5 rows x 9 cols = 45
)  # 55

# Map from piece_type (1-7) to reachable squares for friendly (color_idx=0) and enemy (color_idx=1)
REACHABLE_SQUARES = {
    # color_idx=0 (friendly, at bottom)
    (0, GENERAL):  FRIENDLY_GENERAL_SQ,
    (0, ADVISOR):  FRIENDLY_ADVISOR_SQ,
    (0, ELEPHANT): FRIENDLY_ELEPHANT_SQ,
    (0, HORSE):    ALL_SQUARES,
    (0, CHARIOT):  ALL_SQUARES,
    (0, CANNON):   ALL_SQUARES,
    (0, SOLDIER):  FRIENDLY_SOLDIER_SQ,
    # color_idx=1 (enemy, at top)
    (1, GENERAL):  ENEMY_GENERAL_SQ,
    (1, ADVISOR):  ENEMY_ADVISOR_SQ,
    (1, ELEPHANT): ENEMY_ELEPHANT_SQ,
    (1, HORSE):    ALL_SQUARES,
    (1, CHARIOT):  ALL_SQUARES,
    (1, CANNON):   ALL_SQUARES,
    (1, SOLDIER):  ENEMY_SOLDIER_SQ,
}

# ─── Build the mapping table ────────────────────────────────────────────────
# sq_to_feat[color_idx][piece_type][square] -> feature index (or -1 if invalid)
# feat_offset[color_idx][piece_type] -> starting feature index for this (color, type)

def _build_feature_map():
    """Build mapping from (color_idx, piece_type, square) -> feature index."""
    sq_to_feat = np.full((2, 8, 90), -1, dtype=np.int32)  # -1 = invalid
    feat_offset = np.zeros((2, 8), dtype=np.int32)
    idx = 0
    for color_idx in range(2):
        for pt in range(1, 8):
            feat_offset[color_idx][pt] = idx
            squares = REACHABLE_SQUARES[(color_idx, pt)]
            for sq in squares:
                sq_to_feat[color_idx][pt][sq] = idx
                idx += 1
    return sq_to_feat, feat_offset, idx

SQ_TO_FEAT, FEAT_OFFSET, FEATURES_PER_PERSPECTIVE = _build_feature_map()

assert FEATURES_PER_PERSPECTIVE == 692, f"Expected 692 features, got {FEATURES_PER_PERSPECTIVE}"

# Export for use in other modules
ACCUMULATOR_SIZE = 128
HIDDEN1_SIZE = 32
HIDDEN2_SIZE = 32


# ─── Model ───────────────────────────────────────────────────────────────────

class ClippedReLU(nn.Module):
    """ReLU clamped to [0, 1]."""
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


class NNUENetV2(nn.Module):
    """NNUE v2 with 692 features."""

    def __init__(self):
        super().__init__()
        self.accumulator = nn.Linear(FEATURES_PER_PERSPECTIVE, ACCUMULATOR_SIZE)
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
        Args:
            stm_features: (batch, 692) binary features from STM perspective
            nstm_features: (batch, 692) binary features from NSTM perspective
        Returns:
            (batch, 1) eval in [0, 1], STM perspective (1.0=win, 0.5=draw, 0.0=loss)
        """
        stm_acc = self.clipped_relu(self.accumulator(stm_features))
        nstm_acc = self.clipped_relu(self.accumulator(nstm_features))
        x = torch.cat([stm_acc, nstm_acc], dim=1)
        x = self.clipped_relu(self.fc1(x))
        x = self.clipped_relu(self.fc2(x))
        x = torch.sigmoid(self.fc_out(x))
        return x


# ─── Feature extraction ─────────────────────────────────────────────────────

def board_to_features_v2(board_flat, turn):
    """
    Convert flat int8[90] board + turn to 692-dim feature vectors.

    Board is normalized so the perspective's pieces are always at bottom:
    - If perspective is Red (turn=1): no mirror (Red already at bottom)
    - If perspective is Black (turn=-1): mirror board (sq -> 89-sq)

    Args:
        board_flat: numpy int8 (90,)
        turn: int8, 1=Red, -1=Black
    Returns:
        stm_features: float32 (692,)
        nstm_features: float32 (692,)
    """
    stm = np.zeros(FEATURES_PER_PERSPECTIVE, dtype=np.float32)
    nstm = np.zeros(FEATURES_PER_PERSPECTIVE, dtype=np.float32)

    for sq in range(90):
        piece = board_flat[sq]
        if piece == 0:
            continue

        piece_color = 1 if piece > 0 else -1
        piece_type = abs(piece)

        # STM perspective: normalize so STM's pieces are at bottom
        if turn == 1:  # Red is STM, Red at bottom already
            stm_sq = sq
        else:  # Black is STM, mirror to put Black at bottom
            stm_sq = 89 - sq

        stm_color_idx = 0 if piece_color == turn else 1
        feat_idx = SQ_TO_FEAT[stm_color_idx][piece_type][stm_sq]
        if feat_idx >= 0:
            stm[feat_idx] = 1.0

        # NSTM perspective: normalize so NSTM's pieces are at bottom
        nstm_turn = -turn
        if nstm_turn == 1:  # Red is NSTM
            nstm_sq = sq
        else:  # Black is NSTM
            nstm_sq = 89 - sq

        nstm_color_idx = 0 if piece_color == nstm_turn else 1
        feat_idx = SQ_TO_FEAT[nstm_color_idx][piece_type][nstm_sq]
        if feat_idx >= 0:
            nstm[feat_idx] = 1.0

    return stm, nstm


def board_batch_to_features_v2(boards, turns):
    """
    Vectorized batch conversion.

    Args:
        boards: int8 (N, 90)
        turns: int8 (N,)
    Returns:
        stm_features: float32 (N, 692)
        nstm_features: float32 (N, 692)
    """
    N = boards.shape[0]
    stm = np.zeros((N, FEATURES_PER_PERSPECTIVE), dtype=np.float32)
    nstm = np.zeros((N, FEATURES_PER_PERSPECTIVE), dtype=np.float32)

    for piece_type in range(1, 8):
        for color_sign in [1, -1]:
            piece_val = color_sign * piece_type
            mask = (boards == piece_val)
            if not mask.any():
                continue

            batch_idx, sq_idx = np.where(mask)
            stm_turns = turns[batch_idx]

            # ─── STM features ───
            # Normalize square: mirror if STM is Black
            stm_sq = np.where(stm_turns == 1, sq_idx, 89 - sq_idx)
            stm_color_idx = np.where(color_sign == stm_turns, 0, 1).astype(np.int32)

            # Look up feature indices
            stm_feat = SQ_TO_FEAT[stm_color_idx, piece_type, stm_sq]
            valid = stm_feat >= 0
            if valid.any():
                stm[batch_idx[valid], stm_feat[valid]] = 1.0

            # ─── NSTM features ───
            nstm_turns = -stm_turns
            nstm_sq = np.where(nstm_turns == 1, sq_idx, 89 - sq_idx)
            nstm_color_idx = np.where(color_sign == nstm_turns, 0, 1).astype(np.int32)

            nstm_feat = SQ_TO_FEAT[nstm_color_idx, piece_type, nstm_sq]
            valid = nstm_feat >= 0
            if valid.any():
                nstm[batch_idx[valid], nstm_feat[valid]] = 1.0

    return stm, nstm


# ─── Feature map export (for C++ engine) ────────────────────────────────────

def get_feature_map_for_export():
    """Return the SQ_TO_FEAT table and FEATURES_PER_PERSPECTIVE for C++ export."""
    return SQ_TO_FEAT, FEATURES_PER_PERSPECTIVE


if __name__ == "__main__":
    # Sanity checks
    print(f"Features per perspective: {FEATURES_PER_PERSPECTIVE}")
    print(f"Feature counts per (color, type):")
    for color_idx in range(2):
        color_name = "Friendly" if color_idx == 0 else "Enemy"
        for pt in range(1, 8):
            pt_names = {1: "General", 2: "Advisor", 3: "Elephant",
                        4: "Horse", 5: "Chariot", 6: "Cannon", 7: "Soldier"}
            n = len(REACHABLE_SQUARES[(color_idx, pt)])
            print(f"  {color_name} {pt_names[pt]}: {n} squares")

    # Test with a starting position
    # Standard Xiangqi starting position
    board = np.zeros(90, dtype=np.int8)
    # Red pieces (bottom, rows 5-9)
    board[9*9+0] = CHARIOT;  board[9*9+8] = CHARIOT
    board[9*9+1] = HORSE;    board[9*9+7] = HORSE
    board[9*9+2] = ELEPHANT; board[9*9+6] = ELEPHANT
    board[9*9+3] = ADVISOR;  board[9*9+5] = ADVISOR
    board[9*9+4] = GENERAL
    board[7*9+1] = CANNON;   board[7*9+7] = CANNON
    board[6*9+0] = SOLDIER;  board[6*9+2] = SOLDIER
    board[6*9+4] = SOLDIER;  board[6*9+6] = SOLDIER; board[6*9+8] = SOLDIER
    # Black pieces (top, rows 0-4)
    board[0*9+0] = -CHARIOT;  board[0*9+8] = -CHARIOT
    board[0*9+1] = -HORSE;    board[0*9+7] = -HORSE
    board[0*9+2] = -ELEPHANT; board[0*9+6] = -ELEPHANT
    board[0*9+3] = -ADVISOR;  board[0*9+5] = -ADVISOR
    board[0*9+4] = -GENERAL
    board[2*9+1] = -CANNON;   board[2*9+7] = -CANNON
    board[3*9+0] = -SOLDIER;  board[3*9+2] = -SOLDIER
    board[3*9+4] = -SOLDIER;  board[3*9+6] = -SOLDIER; board[3*9+8] = -SOLDIER

    stm, nstm = board_to_features_v2(board, 1)  # Red to move
    print(f"\nStarting position (Red STM):")
    print(f"  STM active features: {int(stm.sum())}")
    print(f"  NSTM active features: {int(nstm.sum())}")

    # Test batch
    boards = np.stack([board, board])
    turns = np.array([1, -1], dtype=np.int8)
    stm_b, nstm_b = board_batch_to_features_v2(boards, turns)
    print(f"\nBatch test (2 positions):")
    print(f"  STM active: {int(stm_b[0].sum())}, {int(stm_b[1].sum())}")
    print(f"  NSTM active: {int(nstm_b[0].sum())}, {int(nstm_b[1].sum())}")

    # Model
    model = NNUENetV2()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {param_count:,}")
