"""AI that evaluates moves by material value and picks the best capture."""

import random
from ai.base import BaseAI
from engine.board import GENERAL, ADVISOR, ELEPHANT, HORSE, CHARIOT, CANNON, SOLDIER
from engine.game import Game
from engine.move import Move
from engine.rules import is_in_check

# Material values for piece types
PIECE_VALUES = {
    GENERAL: 10000,
    ADVISOR: 20,
    ELEPHANT: 20,
    HORSE: 40,
    CHARIOT: 90,
    CANNON: 45,
    SOLDIER: 10,
}


def evaluate_board(game: Game) -> float:
    """Evaluate board from Red's perspective. Positive = Red advantage."""
    score = 0.0
    board = game.board
    for r in range(10):
        for c in range(9):
            p = board.get(r, c)
            if p == 0:
                continue
            piece_type = abs(p)
            value = PIECE_VALUES.get(piece_type, 0)
            if p > 0:
                score += value
            else:
                score -= value

    # Bonus for checking the opponent
    color = game.current_turn
    if is_in_check(board, -color):
        score += 5 * color  # bonus for the side that delivered check

    return score


class GreedyAI(BaseAI):
    @property
    def name(self) -> str:
        return "Greedy"

    def choose_move(self, game: Game) -> Move:
        legal = game.get_legal_moves()
        color = game.current_turn  # 1 for red, -1 for black

        best_score = float("-inf")
        best_moves: list[Move] = []

        for move in legal:
            # Simulate the move
            new_game = Game.__new__(Game)
            new_game.board = game.board.copy()
            new_game.current_turn = game.current_turn
            new_game.move_history = []
            new_game.board_history = []
            new_game.status = game.status

            piece = new_game.board.get(move.from_row, move.from_col)
            new_game.board.set(move.from_row, move.from_col, 0)
            new_game.board.set(move.to_row, move.to_col, piece)
            new_game.current_turn = -new_game.current_turn

            # Evaluate from current player's perspective
            raw_score = evaluate_board(new_game)
            score = raw_score * color  # normalize so higher = better for current player

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return random.choice(best_moves)
