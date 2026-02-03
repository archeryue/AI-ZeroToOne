"""Move <-> action index mapping for RL.

Placeholder for Phase 2.

Action space: ~2086 unique move patterns.
Each action = (from_row, from_col, direction, distance) encoded as flat index.
Illegal actions masked during training.
"""

# TODO: Implement action encoding/decoding
# - encode_move(from_row, from_col, to_row, to_col) -> int
# - decode_action(action_idx) -> (from_row, from_col, to_row, to_col)
# - get_action_mask(board, color) -> np.ndarray[bool]
