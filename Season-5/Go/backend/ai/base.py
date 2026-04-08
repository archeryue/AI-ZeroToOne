"""Base AI interface for Go.

Provides Monte Carlo Score Estimation for resign decisions.
AIs with their own evaluation (e.g., MCTS win rate) can override _should_resign.
"""

import random
from abc import ABC, abstractmethod
from engine.game import Game, GameStatus
from engine.board import Stone, RESIGN


class BaseAI(ABC):
    """Abstract base class for Go AI players.

    choose_move returns:
        (row, col) — place a stone
        (-1, -1)  — pass
        (-2, -2)  — resign
    """

    # --- Resign configuration ---
    # Base threshold for 19×19; scaled proportionally to board area for other sizes
    RESIGN_SCORE_THRESHOLD_19 = 25.0
    RESIGN_LOSS_RATE = 0.85         # Fraction of lost playouts required
    RESIGN_MC_PLAYOUTS = 50         # Playouts for MC score estimation

    @abstractmethod
    def choose_move(self, game: Game) -> tuple[int, int]:
        """Choose a move. Returns (row, col), PASS, or RESIGN."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    # ------------------------------------------------------------------
    # Resign evaluation framework
    # ------------------------------------------------------------------

    def _should_invoke_resign_check(self, game: Game) -> bool:
        """Whether conditions are met to run resign evaluation.

        Default: after enough moves have been played, check every ~5 AI moves.
        Override in subclasses to customize timing.
        """
        move_count = len(game.move_history)
        # Don't resign in the opening
        min_moves = max(20, game.size * 2)
        if move_count < min_moves:
            return False
        # Check every 10 total moves (≈ every 5 AI moves)
        return move_count % 10 == 0

    def _resign_score_threshold(self, game: Game) -> float:
        """Score threshold scaled by board area.

        19×19 → 25 pts, 13×13 → ~12 pts, 9×9 → ~6 pts.
        """
        return self.RESIGN_SCORE_THRESHOLD_19 * (game.size ** 2) / (19 ** 2)

    def _should_resign(self, game: Game, color: int) -> bool:
        """Decide whether to resign using Monte Carlo Score Estimation.

        Runs random playouts from the current position and checks whether the
        AI is consistently losing by a large margin.

        Override in subclasses that have their own evaluation ability.
        """
        avg_diff, loss_rate = self._mc_score_estimate(game, color)
        threshold = self._resign_score_threshold(game)
        return avg_diff < -threshold and loss_rate > self.RESIGN_LOSS_RATE

    def _mc_score_estimate(self, game: Game, color: int) -> tuple[float, float]:
        """Monte Carlo Score Estimation.

        Runs random playouts from the current position to estimate the score.

        Returns:
            (avg_score_diff, loss_rate) where
            - avg_score_diff: mean(AI score − opponent score); negative = losing
            - loss_rate: fraction of playouts the AI loses (0.0–1.0)
        """
        total_diff = 0.0
        losses = 0

        for _ in range(self.RESIGN_MC_PLAYOUTS):
            sim = self._clone_game(game)
            self._random_playout(sim)

            black_score, white_score = sim.board.score(sim.komi)
            if color == Stone.BLACK:
                diff = black_score - white_score
            else:
                diff = white_score - black_score

            total_diff += diff
            if diff < 0:
                losses += 1

        avg_diff = total_diff / self.RESIGN_MC_PLAYOUTS
        loss_rate = losses / self.RESIGN_MC_PLAYOUTS
        return avg_diff, loss_rate

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clone_game(game: Game) -> Game:
        """Lightweight clone of game state for simulation."""
        g = Game.__new__(Game)
        g.board = game.board.copy()
        g.size = game.size
        g.komi = game.komi
        g.current_turn = game.current_turn
        g.status = game.status
        g.move_history = []
        g.board_history = []
        g.captured = dict(game.captured)
        g.consecutive_passes = game.consecutive_passes
        g.final_score = None
        g.resigned_by = None
        return g

    @staticmethod
    def _random_playout(game: Game, max_moves: int = 200):
        """Play random moves until the game ends or *max_moves* is reached."""
        moves = 0
        consecutive_passes = game.consecutive_passes

        while game.status == GameStatus.PLAYING and moves < max_moves:
            legal = game.get_legal_moves()

            if not legal or random.random() < 0.1:
                game.make_move(-1, -1)
                consecutive_passes += 1
                if consecutive_passes >= 2:
                    break
            else:
                move = random.choice(legal)
                try:
                    game.make_move(move[0], move[1])
                    consecutive_passes = 0
                except ValueError:
                    game.make_move(-1, -1)
                    consecutive_passes += 1

            moves += 1
