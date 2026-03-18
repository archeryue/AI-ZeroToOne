"""Reward shaping wrapper for Chinese Chess.

Adds material-based intermediate rewards on top of the base terminal rewards.
Each capture produces an immediate reward proportional to the captured piece's value.
"""

import gymnasium as gym
import numpy as np

from engine.board import ROWS, COLS, GENERAL

# Piece values (same scale as Greedy AI, excluding General)
PIECE_VALUES = {
    # GENERAL: excluded — capturing General ends the game (terminal reward covers it)
    2: 0.1,    # Advisor   (3/30)
    3: 0.1,    # Elephant  (3/30)
    4: 0.167,  # Horse     (5/30)
    5: 0.333,  # Chariot   (10/30)
    6: 0.167,  # Cannon    (5/30)
    7: 0.033,  # Soldier   (1/30)
}

# Terminal reward for win/loss
# Ratio preserved: win(1.0) >> max capturable material(~2.1)
# Win is still the dominant signal
WIN_REWARD = 1.0


class RewardShapingWrapper(gym.Wrapper):
    """Adds material-delta rewards to each step.

    After each move, reward += (material_change) * scale, computed from
    the perspective of the player who just moved.

    With scale=1.0 (default):
      - Capturing a Chariot (0.333): +0.333
      - Capturing a Cannon (0.167):  +0.167
      - Capturing a Soldier (0.033): +0.033
      - Losing a Chariot:            -0.333

    Terminal rewards (±1.0) dominate all possible material gains (~2.1 max).
    """

    def __init__(self, env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale
        self._prev_material = 0.0

    @property
    def game(self):
        return self.env.game

    @property
    def current_turn(self):
        return self.env.current_turn

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_material = self._compute_material()
        return obs, info

    def step(self, action):
        # Who is moving? (before the step, since step switches turns)
        mover = self.env.current_turn

        obs, reward, terminated, truncated, info = self.env.step(action)

        # Compute material delta
        new_material = self._compute_material()
        # Material is from Red's perspective. If Red moved and material went up,
        # that's good for Red (+delta). If Black moved and material went down,
        # that's good for Black (-delta from Red's view = +delta for Black).
        delta = (new_material - self._prev_material) * mover
        shaped_reward = delta * self.scale

        self._prev_material = new_material

        return obs, reward + shaped_reward, terminated, truncated, info

    def _compute_material(self) -> float:
        """Compute material score from Red's perspective (positive = Red advantage)."""
        score = 0.0
        board = self.env.game.board
        for r in range(ROWS):
            for c in range(COLS):
                p = board.get(r, c)
                if p == 0:
                    continue
                piece_type = abs(p)
                value = PIECE_VALUES.get(piece_type, 0)
                if p > 0:
                    score += value
                else:
                    score -= value
        return score
