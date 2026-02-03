"""Alpha-beta pruning minimax AI for Chinese Chess."""

import random
from ai.base import BaseAI
from ai.greedy_ai import PIECE_VALUES, evaluate_board
from engine.board import Board, ROWS, COLS, RED, BLACK
from engine.game import Game, GameStatus
from engine.move import Move
from engine.rules import get_legal_moves, is_in_check


# Positional bonuses for key pieces (row, col) -> bonus
# These encourage pieces to occupy central/aggressive positions

def _positional_bonus(board: Board) -> float:
    """Small positional bonuses beyond raw material."""
    score = 0.0
    for r in range(ROWS):
        for c in range(COLS):
            p = board.get(r, c)
            if p == 0:
                continue
            piece_type = abs(p)
            color = 1 if p > 0 else -1

            # Soldiers get a bonus for advancing
            if piece_type == 7:
                if color == RED:
                    advance = 9 - r  # higher advance = more rows from start
                    score += advance * 0.5 * color
                else:
                    advance = r
                    score += advance * 0.5 * color

            # Horses get a small centrality bonus
            if piece_type == 4:
                center_dist = abs(c - 4)
                score += (4 - center_dist) * 0.3 * color

            # Chariots on open files get a bonus
            if piece_type == 5:
                open_file = True
                for rr in range(ROWS):
                    if rr != r and abs(board.get(rr, c)) == 7:
                        open_file = False
                        break
                if open_file:
                    score += 2.0 * color

    return score


def evaluate(game: Game) -> float:
    """Full evaluation: material + positional + check bonus."""
    base = evaluate_board(game)
    positional = _positional_bonus(game.board)
    return base + positional


def _simulate_move(game: Game, move: Move) -> Game:
    """Create a new Game state after applying a move (lightweight)."""
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
    return new_game


def _order_moves(moves: list[Move]) -> list[Move]:
    """Order moves for better alpha-beta pruning: captures first, then by piece value."""
    def score(m: Move) -> int:
        if m.captured != 0:
            return PIECE_VALUES.get(abs(m.captured), 0) * 10
        return 0
    return sorted(moves, key=score, reverse=True)


def alphabeta(game: Game, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
    """Alpha-beta pruning minimax. Evaluates from Red's perspective."""
    if depth == 0:
        return evaluate(game)

    legal = get_legal_moves(game.board, game.current_turn)

    if not legal:
        # No legal moves: current player loses
        if game.current_turn == RED:
            return -99999
        else:
            return 99999

    legal = _order_moves(legal)

    if maximizing:
        value = float("-inf")
        for move in legal:
            child = _simulate_move(game, move)
            value = max(value, alphabeta(child, depth - 1, alpha, beta, False))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = float("inf")
        for move in legal:
            child = _simulate_move(game, move)
            value = min(value, alphabeta(child, depth - 1, alpha, beta, True))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


class MinimaxAI(BaseAI):
    def __init__(self, depth: int = 3):
        self.depth = depth

    @property
    def name(self) -> str:
        return f"Minimax(depth={self.depth})"

    def choose_move(self, game: Game) -> Move:
        legal = game.get_legal_moves()
        color = game.current_turn
        maximizing = color == RED

        best_score = float("-inf") if maximizing else float("inf")
        best_moves: list[Move] = []

        ordered = _order_moves(legal)

        for move in ordered:
            child = _simulate_move(game, move)
            score = alphabeta(
                child, self.depth - 1,
                float("-inf"), float("inf"),
                not maximizing,
            )

            if maximizing:
                if score > best_score:
                    best_score = score
                    best_moves = [move]
                elif score == best_score:
                    best_moves.append(move)
            else:
                if score < best_score:
                    best_score = score
                    best_moves = [move]
                elif score == best_score:
                    best_moves.append(move)

        return random.choice(best_moves)
