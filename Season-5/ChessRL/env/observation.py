"""Board -> observation tensor conversion.

Observation: 15 x 10 x 9 float32 tensor
- Planes 0-6:  Red pieces (one-hot per piece type: General, Advisor, Elephant, Horse, Chariot, Cannon, Soldier)
- Planes 7-13: Black pieces (same order)
- Plane 14:    Current turn (all 1s for Red's turn, all 0s for Black's turn)
"""

import numpy as np

from engine.board import Board, ROWS, COLS, RED

NUM_PLANES = 15
OBS_SHAPE = (NUM_PLANES, ROWS, COLS)  # (15, 10, 9)


def board_to_observation(board: Board, current_turn: int) -> np.ndarray:
    """Convert board state to a (15, 10, 9) float32 tensor."""
    obs = np.zeros(OBS_SHAPE, dtype=np.float32)

    for r in range(ROWS):
        for c in range(COLS):
            piece = board.get(r, c)
            if piece == 0:
                continue
            piece_type = abs(piece)  # 1-7
            plane_idx = piece_type - 1  # 0-6
            if piece < 0:
                plane_idx += 7  # Black pieces: planes 7-13
            obs[plane_idx, r, c] = 1.0

    # Turn plane
    if current_turn == RED:
        obs[14, :, :] = 1.0

    return obs
