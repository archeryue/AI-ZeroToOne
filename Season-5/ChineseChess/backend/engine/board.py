"""Board representation for Chinese Chess (Xiangqi).

Board is 10 rows x 9 columns.
Positive values = Red pieces, negative = Black pieces.
Piece type codes (absolute value):
  1 = General (帅/将)
  2 = Advisor (仕/士)
  3 = Elephant (相/象)
  4 = Horse (马)
  5 = Chariot (车)
  6 = Cannon (炮)
  7 = Soldier (兵/卒)

Row 0 is Black's back rank; row 9 is Red's back rank.
"""

from __future__ import annotations

import copy
from typing import Optional

ROWS = 10
COLS = 9

GENERAL = 1
ADVISOR = 2
ELEPHANT = 3
HORSE = 4
CHARIOT = 5
CANNON = 6
SOLDIER = 7

RED = 1   # positive
BLACK = -1  # negative

PIECE_NAMES_RED = {1: "帅", 2: "仕", 3: "相", 4: "马", 5: "车", 6: "炮", 7: "兵"}
PIECE_NAMES_BLACK = {1: "将", 2: "士", 3: "象", 4: "马", 5: "车", 6: "炮", 7: "卒"}

# Initial board setup — row 0 is Black's back rank, row 9 is Red's back rank
INITIAL_BOARD = [
    [-5, -4, -3, -2, -1, -2, -3, -4, -5],  # row 0: Black back rank
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],  # row 1
    [ 0, -6,  0,  0,  0,  0,  0, -6,  0],  # row 2: Black cannons
    [-7,  0, -7,  0, -7,  0, -7,  0, -7],  # row 3: Black soldiers
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],  # row 4: river
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],  # row 5: river
    [ 7,  0,  7,  0,  7,  0,  7,  0,  7],  # row 6: Red soldiers
    [ 0,  6,  0,  0,  0,  0,  0,  6,  0],  # row 7: Red cannons
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],  # row 8
    [ 5,  4,  3,  2,  1,  2,  3,  4,  5],  # row 9: Red back rank
]


class Board:
    """10x9 Chinese Chess board."""

    def __init__(self, grid: Optional[list[list[int]]] = None):
        if grid is not None:
            self.grid = [row[:] for row in grid]
        else:
            self.grid = [row[:] for row in INITIAL_BOARD]

    def get(self, row: int, col: int) -> int:
        return self.grid[row][col]

    def set(self, row: int, col: int, piece: int) -> None:
        self.grid[row][col] = piece

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < ROWS and 0 <= col < COLS

    def color_of(self, row: int, col: int) -> int:
        """Return RED (1), BLACK (-1), or 0 if empty."""
        p = self.grid[row][col]
        if p > 0:
            return RED
        elif p < 0:
            return BLACK
        return 0

    def piece_type(self, row: int, col: int) -> int:
        return abs(self.grid[row][col])

    def find_general(self, color: int) -> tuple[int, int]:
        """Find the position of the general for the given color."""
        target = GENERAL * color
        for r in range(ROWS):
            for c in range(COLS):
                if self.grid[r][c] == target:
                    return (r, c)
        raise ValueError(f"General not found for color {color}")

    def copy(self) -> Board:
        return Board(self.grid)

    def to_fen(self) -> str:
        """Serialize board to a FEN-like string."""
        piece_chars = {
            5: "R", 4: "N", 3: "B", 2: "A", 1: "K", 6: "C", 7: "P",
            -5: "r", -4: "n", -3: "b", -2: "a", -1: "k", -6: "c", -7: "p",
        }
        rows = []
        for r in range(ROWS):
            empty = 0
            row_str = ""
            for c in range(COLS):
                p = self.grid[r][c]
                if p == 0:
                    empty += 1
                else:
                    if empty > 0:
                        row_str += str(empty)
                        empty = 0
                    row_str += piece_chars[p]
            if empty > 0:
                row_str += str(empty)
            rows.append(row_str)
        return "/".join(rows)

    @classmethod
    def from_fen(cls, fen: str) -> Board:
        """Create board from a FEN-like string."""
        char_to_piece = {
            "R": 5, "N": 4, "B": 3, "A": 2, "K": 1, "C": 6, "P": 7,
            "r": -5, "n": -4, "b": -3, "a": -2, "k": -1, "c": -6, "p": -7,
        }
        grid = []
        for row_str in fen.split("/"):
            row = []
            for ch in row_str:
                if ch.isdigit():
                    row.extend([0] * int(ch))
                else:
                    row.append(char_to_piece[ch])
            grid.append(row)
        return cls(grid)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Board):
            return NotImplemented
        return self.grid == other.grid

    def __repr__(self) -> str:
        return f"Board(fen='{self.to_fen()}')"
