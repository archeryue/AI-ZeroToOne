"""Gymnasium wrapper for Chinese Chess environment.

Placeholder for Phase 2 RL training.
"""

# TODO: Implement Gymnasium environment
# - observation_space: Box(0, 1, shape=(15, 10, 9), dtype=float32)
#   14 piece planes (7 types x 2 colors) + 1 turn plane
# - action_space: Discrete(~2086)
#   Pre-computed legal move patterns mapped to flat indices
# - step(action) -> obs, reward, done, truncated, info
# - reset() -> obs, info
# - Illegal action masking via info["action_mask"]
