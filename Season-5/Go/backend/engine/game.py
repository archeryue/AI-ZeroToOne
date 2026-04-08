"""Go game state management.

Wraps the Board with turn tracking, move history, pass/resign handling,
and game-over detection.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from .board import Board, Stone, PASS


class GameStatus(str, Enum):
    PLAYING = "playing"
    BLACK_WIN = "black_win"
    WHITE_WIN = "white_win"


class Game:
    """Full Go game with history, undo, pass, and scoring."""

    def __init__(self, size: int = 19, komi: float = 7.5):
        self.board = Board(size)
        self.size = size
        self.komi = komi
        self.current_turn: int = Stone.BLACK  # Black plays first
        self.status = GameStatus.PLAYING
        self.move_history: list[dict] = []  # list of {row, col, captured, ko_before}
        self.board_history: list[list[list[int]]] = []  # for undo
        self.captured: dict[int, int] = {Stone.BLACK: 0, Stone.WHITE: 0}  # prisoners taken BY each color
        self.consecutive_passes = 0
        self.final_score: Optional[tuple[float, float]] = None
        self.resigned_by: Optional[int] = None

    def make_move(self, row: int, col: int) -> dict:
        """Make a move (place stone or pass).

        Pass is indicated by row=-1, col=-1.

        Returns:
            dict with move info: {row, col, captured, is_pass}
        """
        if self.status != GameStatus.PLAYING:
            raise ValueError("Game is already over")

        # Pass
        if (row, col) == PASS:
            return self._pass()

        # Save state for undo
        self.board_history.append(self.board.to_grid_list())
        ko_before = self.board.ko_point

        # Place stone
        captured = self.board.place_stone(row, col, self.current_turn)
        self.captured[self.current_turn] += captured
        self.consecutive_passes = 0

        move_info = {
            "row": row,
            "col": col,
            "captured": captured,
            "is_pass": False,
            "ko_before": ko_before,
        }
        self.move_history.append(move_info)

        # Switch turn
        self.current_turn = Stone.BLACK if self.current_turn == Stone.WHITE else Stone.WHITE

        return {"row": row, "col": col, "captured": captured, "is_pass": False}

    def _pass(self) -> dict:
        """Handle a pass move."""
        self.board_history.append(self.board.to_grid_list())
        ko_before = self.board.ko_point

        self.consecutive_passes += 1
        self.board.ko_point = None  # Ko resets on pass

        move_info = {
            "row": -1,
            "col": -1,
            "captured": 0,
            "is_pass": True,
            "ko_before": ko_before,
        }
        self.move_history.append(move_info)

        # Two consecutive passes end the game
        if self.consecutive_passes >= 2:
            self._end_game()

        # Switch turn
        self.current_turn = Stone.BLACK if self.current_turn == Stone.WHITE else Stone.WHITE

        return {"row": -1, "col": -1, "captured": 0, "is_pass": True}

    def resign(self, color: int):
        """Player resigns."""
        self.resigned_by = color
        if color == Stone.BLACK:
            self.status = GameStatus.WHITE_WIN
        else:
            self.status = GameStatus.BLACK_WIN

    def _end_game(self):
        """Score the game and determine winner."""
        black_score, white_score = self.board.score(self.komi)
        self.final_score = (black_score, white_score)
        if black_score > white_score:
            self.status = GameStatus.BLACK_WIN
        else:
            self.status = GameStatus.WHITE_WIN

    def undo(self) -> bool:
        """Undo the last move. Returns True if successful."""
        if not self.move_history:
            return False

        move_info = self.move_history.pop()
        prev_grid = self.board_history.pop()

        # Restore board
        self.board.grid = prev_grid
        self.board.ko_point = move_info.get("ko_before")

        # Switch turn back first — now current_turn = the player who made the undone move
        self.current_turn = Stone.BLACK if self.current_turn == Stone.WHITE else Stone.WHITE

        # Restore captures — current_turn is now the player who made the move
        if not move_info["is_pass"]:
            self.captured[self.current_turn] -= move_info["captured"]

        # Restore consecutive passes
        if move_info["is_pass"]:
            # Recount consecutive passes from history
            self.consecutive_passes = 0
            for m in reversed(self.move_history):
                if m["is_pass"]:
                    self.consecutive_passes += 1
                else:
                    break
        else:
            self.consecutive_passes = 0

        # Reset game status if it was over
        self.status = GameStatus.PLAYING
        self.final_score = None

        return True

    def get_legal_moves(self) -> list[tuple[int, int]]:
        """Get all legal moves for current player (excluding pass)."""
        if self.status != GameStatus.PLAYING:
            return []
        return self.board.get_legal_moves(self.current_turn)

    def to_dict(self) -> dict:
        """Serialize game state for frontend."""
        legal_moves = self.get_legal_moves()

        # Format move history for frontend
        history = []
        for i, m in enumerate(self.move_history):
            color = "black" if (i % 2 == 0) else "white"
            if m["is_pass"]:
                history.append({"color": color, "position": "pass", "captured": 0})
            else:
                history.append({
                    "color": color,
                    "position": f"{m['row']},{m['col']}",
                    "captured": m["captured"],
                })

        result = {
            "board": self.board.to_grid_list(),
            "board_size": self.size,
            "current_turn": "black" if self.current_turn == Stone.BLACK else "white",
            "status": self.status.value,
            "legal_moves": [[r, c] for r, c in legal_moves],
            "move_history": history,
            "move_count": len(self.move_history),
            "captured": {
                "black": self.captured[Stone.BLACK],
                "white": self.captured[Stone.WHITE],
            },
            "komi": self.komi,
        }

        if self.resigned_by is not None:
            result["resigned_by"] = "black" if self.resigned_by == Stone.BLACK else "white"

        if self.final_score is not None:
            result["score"] = {
                "black": self.final_score[0],
                "white": self.final_score[1],
            }

        # Territory map for end-game display
        if self.status != GameStatus.PLAYING:
            result["territory"] = self._compute_territory()

        return result

    def _compute_territory(self) -> list[list[int]]:
        """Compute territory map: 0=neutral, 1=black territory, 2=white territory."""
        territory = [[0] * self.size for _ in range(self.size)]
        visited: set[tuple[int, int]] = set()

        for r in range(self.size):
            for c in range(self.size):
                if self.board.grid[r][c] != Stone.EMPTY or (r, c) in visited:
                    continue
                region: set[tuple[int, int]] = set()
                borders: set[int] = set()
                stack = [(r, c)]
                while stack:
                    er, ec = stack.pop()
                    if (er, ec) in region:
                        continue
                    region.add((er, ec))
                    for nr, nc in self.board._neighbors(er, ec):
                        cell = self.board.grid[nr][nc]
                        if cell == Stone.EMPTY and (nr, nc) not in region:
                            stack.append((nr, nc))
                        elif cell != Stone.EMPTY:
                            borders.add(cell)
                visited |= region

                if borders == {Stone.BLACK}:
                    for er, ec in region:
                        territory[er][ec] = Stone.BLACK
                elif borders == {Stone.WHITE}:
                    for er, ec in region:
                        territory[er][ec] = Stone.WHITE

        return territory
