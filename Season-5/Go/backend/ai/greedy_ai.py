"""Greedy AI: prioritizes captures, influence, and center control."""

import random
from engine.board import Stone, RESIGN
from engine.game import Game
from .base import BaseAI


class GreedyAI(BaseAI):
    def choose_move(self, game: Game) -> tuple[int, int]:
        legal = game.get_legal_moves()
        if not legal:
            return (-1, -1)

        color = game.current_turn

        # Resign check — invoke MC Score Estimation periodically
        if self._should_invoke_resign_check(game) and self._should_resign(game, color):
            return RESIGN

        scored: list[tuple[float, tuple[int, int]]] = []

        for r, c in legal:
            score = self._evaluate_move(game, r, c, color)
            scored.append((score, (r, c)))

        # Sort by score descending, pick from top candidates with noise
        scored.sort(key=lambda x: -x[0])
        # Pick randomly from top 3 to add variety
        top_k = min(3, len(scored))
        candidates = scored[:top_k]
        _, move = random.choice(candidates)

        # Pass if all moves are bad (negative score)
        if scored[0][0] < -2.0 and game.move_history and len(game.move_history) > 20:
            return (-1, -1)

        return move

    def _evaluate_move(self, game: Game, row: int, col: int, color: int) -> float:
        """Heuristic evaluation of a move."""
        board = game.board
        size = board.size
        opponent = Stone.BLACK if color == Stone.WHITE else Stone.WHITE
        score = 0.0

        # 1. Captures: huge bonus
        board_copy = board.copy()
        try:
            captured = board_copy.place_stone(row, col, color)
            score += captured * 10.0
        except ValueError:
            return -100.0

        # 2. Center preference (higher score for center positions)
        center = (size - 1) / 2.0
        dist = abs(row - center) + abs(col - center)
        max_dist = center * 2
        score += (1.0 - dist / max_dist) * 2.0

        # 3. Proximity to existing stones (build connected groups)
        for nr, nc in board._neighbors(row, col):
            if board.grid[nr][nc] == color:
                score += 1.0
            elif board.grid[nr][nc] == opponent:
                # Adjacent to opponent — could be attacking
                group, libs = board._find_group(nr, nc)
                if len(libs) <= 2:
                    score += 3.0  # Threatening a group in atari or near-atari

        # 4. Avoid self-atari (placing stone that immediately has only 1 liberty)
        _, own_libs = board_copy._find_group(row, col)
        if len(own_libs) == 1 and captured == 0:
            score -= 5.0

        # 5. Avoid edges on small boards (first/second line penalty early game)
        if len(game.move_history) < size * 2:
            if row == 0 or row == size - 1 or col == 0 or col == size - 1:
                score -= 2.0
            if row == 1 or row == size - 2 or col == 1 or col == size - 2:
                score -= 0.5

        return score

    @property
    def name(self) -> str:
        return "Greedy"
