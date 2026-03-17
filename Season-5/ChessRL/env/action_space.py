"""Move <-> action index mapping for RL.

Action encoding: flat index from (from_row, from_col, to_row, to_col).
  action = from_row * 9 * 90 + from_col * 90 + to_row * 9 + to_col
  Total action space size: 90 * 90 = 8100

Most actions are illegal at any given state — we use action masking
to restrict the policy to legal moves only.
"""

import numpy as np

from engine.board import Board, ROWS, COLS
from engine.rules import get_legal_moves

NUM_ACTIONS = ROWS * COLS * ROWS * COLS  # 90 * 90 = 8100


def encode_move(from_row: int, from_col: int, to_row: int, to_col: int) -> int:
    """Convert a move (from_row, from_col, to_row, to_col) to a flat action index."""
    from_sq = from_row * COLS + from_col  # 0-89
    to_sq = to_row * COLS + to_col        # 0-89
    return from_sq * (ROWS * COLS) + to_sq


def decode_action(action: int) -> tuple[int, int, int, int]:
    """Convert a flat action index back to (from_row, from_col, to_row, to_col)."""
    from_sq = action // (ROWS * COLS)
    to_sq = action % (ROWS * COLS)
    from_row = from_sq // COLS
    from_col = from_sq % COLS
    to_row = to_sq // COLS
    to_col = to_sq % COLS
    return from_row, from_col, to_row, to_col


def get_action_mask(board: Board, color: int) -> np.ndarray:
    """Return a boolean mask of shape (NUM_ACTIONS,) where True = legal action."""
    mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
    for move in get_legal_moves(board, color):
        idx = encode_move(move.from_row, move.from_col, move.to_row, move.to_col)
        mask[idx] = True
    return mask
