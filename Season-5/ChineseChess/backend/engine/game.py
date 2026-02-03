"""Game session: turn management, move application, history, game-over detection."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from engine.board import Board, RED, BLACK, GENERAL, PIECE_NAMES_RED, PIECE_NAMES_BLACK
from engine.move import Move
from engine.rules import get_legal_moves, is_in_check, is_checkmate, is_stalemate


class GameStatus(str, Enum):
    PLAYING = "playing"
    RED_WIN = "red_win"
    BLACK_WIN = "black_win"
    DRAW = "draw"


class Game:
    """Manages a single game of Chinese Chess."""

    def __init__(self):
        self.board = Board()
        self.current_turn: int = RED  # Red moves first
        self.move_history: list[Move] = []
        self.board_history: list[list[list[int]]] = []  # for undo
        self.status: GameStatus = GameStatus.PLAYING

    def get_legal_moves(self) -> list[Move]:
        """Get all legal moves for the current player."""
        if self.status != GameStatus.PLAYING:
            return []
        return get_legal_moves(self.board, self.current_turn)

    def get_legal_moves_dict(self) -> dict[str, list[str]]:
        """Return legal moves as a dict: 'r,c' -> ['r,c', ...] for frontend."""
        moves = self.get_legal_moves()
        result: dict[str, list[str]] = {}
        for m in moves:
            key = f"{m.from_row},{m.from_col}"
            if key not in result:
                result[key] = []
            result[key].append(f"{m.to_row},{m.to_col}")
        return result

    def make_move(self, from_row: int, from_col: int, to_row: int, to_col: int) -> Optional[Move]:
        """Attempt to make a move. Returns the Move if successful, None if illegal."""
        if self.status != GameStatus.PLAYING:
            return None

        # Verify it's a legal move
        legal = self.get_legal_moves()
        move = None
        for m in legal:
            if m.from_row == from_row and m.from_col == from_col and m.to_row == to_row and m.to_col == to_col:
                move = m
                break
        if move is None:
            return None

        # Save state for undo
        self.board_history.append([row[:] for row in self.board.grid])

        # Apply move
        piece = self.board.get(from_row, from_col)
        self.board.set(from_row, from_col, 0)
        self.board.set(to_row, to_col, piece)
        self.move_history.append(move)

        # Switch turn
        self.current_turn = -self.current_turn

        # Check game over
        self._check_game_over()

        return move

    def undo(self) -> Optional[Move]:
        """Undo the last move. Returns the undone Move, or None if no history."""
        if not self.move_history:
            return None

        move = self.move_history.pop()
        prev_grid = self.board_history.pop()
        self.board.grid = prev_grid
        self.current_turn = -self.current_turn
        self.status = GameStatus.PLAYING
        return move

    def resign(self, color: int) -> None:
        """Player of given color resigns."""
        if color == RED:
            self.status = GameStatus.BLACK_WIN
        else:
            self.status = GameStatus.RED_WIN

    def _check_game_over(self) -> None:
        """Check if the game is over after a move."""
        # Check if current player (who needs to move next) is in checkmate or stalemate
        if is_checkmate(self.board, self.current_turn):
            # Current player is checkmated — the other player wins
            self.status = GameStatus.RED_WIN if self.current_turn == BLACK else GameStatus.BLACK_WIN
        elif is_stalemate(self.board, self.current_turn):
            # In Chinese Chess, stalemate = loss for stalemated player
            self.status = GameStatus.RED_WIN if self.current_turn == BLACK else GameStatus.BLACK_WIN

    def is_in_check(self) -> bool:
        """Check if the current player is in check."""
        return is_in_check(self.board, self.current_turn)

    def to_dict(self) -> dict:
        """Serialize game state to a dictionary for the frontend."""
        return {
            "board": self.board.grid,
            "current_turn": "red" if self.current_turn == RED else "black",
            "status": self.status.value,
            "in_check": self.is_in_check(),
            "valid_moves": self.get_legal_moves_dict(),
            "move_history": [
                {
                    "from": [m.from_row, m.from_col],
                    "to": [m.to_row, m.to_col],
                    "captured": m.captured,
                    "ucci": m.to_ucci(),
                }
                for m in self.move_history
            ],
            "fen": self.board.to_fen(),
        }
