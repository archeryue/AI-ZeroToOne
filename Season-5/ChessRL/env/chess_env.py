"""Gymnasium wrapper for Chinese Chess environment.

Observation: (15, 10, 9) float32 tensor (14 piece planes + 1 turn plane)
Action:      Discrete(8100) with illegal action masking
Reward:      +1 for win, -1 for loss, 0 otherwise
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from engine.board import Board, RED, BLACK
from engine.game import Game, GameStatus
from engine.rules import get_legal_moves

from env.action_space import NUM_ACTIONS, encode_move, decode_action, get_action_mask
from env.observation import OBS_SHAPE, board_to_observation


class ChineseChessEnv(gym.Env):
    """Chinese Chess environment for RL training.

    The agent always plays as Red. When used in self-play,
    the board is flipped so both agents see themselves as Red.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 200):
        super().__init__()
        self.observation_space = spaces.Box(0, 1, shape=OBS_SHAPE, dtype=np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.max_steps = max_steps
        self.game = None
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game = Game()
        self.step_count = 0
        obs = board_to_observation(self.game.board, self.game.current_turn)
        info = {"action_mask": get_action_mask(self.game.board, self.game.current_turn)}
        return obs, info

    def step(self, action: int):
        """Execute an action. Returns (obs, reward, terminated, truncated, info)."""
        from_row, from_col, to_row, to_col = decode_action(action)

        # Try to make the move
        move = self.game.make_move(from_row, from_col, to_row, to_col)
        self.step_count += 1

        if move is None:
            # Illegal move — penalize and end episode
            obs = board_to_observation(self.game.board, self.game.current_turn)
            info = {"action_mask": get_action_mask(self.game.board, self.game.current_turn)}
            return obs, -1.0, True, False, info

        # Check game status
        terminated = False
        reward = 0.0

        if self.game.status == GameStatus.RED_WIN:
            reward = 1.0 if self.game.current_turn == BLACK else -1.0
            terminated = True
        elif self.game.status == GameStatus.BLACK_WIN:
            reward = -1.0 if self.game.current_turn == RED else 1.0
            terminated = True
        elif self.game.status == GameStatus.DRAW:
            reward = 0.0
            terminated = True

        # Truncate if too many steps
        truncated = self.step_count >= self.max_steps and not terminated

        obs = board_to_observation(self.game.board, self.game.current_turn)
        info = {"action_mask": get_action_mask(self.game.board, self.game.current_turn)}

        return obs, reward, terminated, truncated, info

    def get_legal_actions(self) -> list[int]:
        """Return list of legal action indices for current player."""
        actions = []
        for move in get_legal_moves(self.game.board, self.game.current_turn):
            actions.append(encode_move(move.from_row, move.from_col, move.to_row, move.to_col))
        return actions

    @property
    def current_turn(self) -> int:
        return self.game.current_turn
