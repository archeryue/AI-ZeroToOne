"""Go board representation and rules.

Implements:
- Stone placement with capture detection
- Group/liberty tracking via flood-fill
- Ko rule (simple ko via board hash)
- Suicide prevention
- Tromp-Taylor area scoring
"""

from __future__ import annotations

import hashlib
from enum import IntEnum
from typing import Optional


class Stone(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2

    @property
    def opponent(self) -> Stone:
        if self == Stone.BLACK:
            return Stone.WHITE
        if self == Stone.WHITE:
            return Stone.BLACK
        return Stone.EMPTY


# Sentinel for "pass" move
PASS = (-1, -1)

# Sentinel for "resign" move
RESIGN = (-2, -2)


class Board:
    """Go board with full rule enforcement."""

    def __init__(self, size: int = 19):
        assert size in (9, 13, 19), f"Unsupported board size: {size}"
        self.size = size
        self.grid: list[list[int]] = [[Stone.EMPTY] * size for _ in range(size)]
        self.ko_point: Optional[tuple[int, int]] = None  # forbidden point due to ko

    def copy(self) -> Board:
        b = Board.__new__(Board)
        b.size = self.size
        b.grid = [row[:] for row in self.grid]
        b.ko_point = self.ko_point
        return b

    def get(self, row: int, col: int) -> int:
        return self.grid[row][col]

    def set(self, row: int, col: int, stone: int):
        self.grid[row][col] = stone

    def _neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        result = []
        if row > 0:
            result.append((row - 1, col))
        if row < self.size - 1:
            result.append((row + 1, col))
        if col > 0:
            result.append((row, col - 1))
        if col < self.size - 1:
            result.append((row, col + 1))
        return result

    def _find_group(self, row: int, col: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        """Find connected group at (row, col) and its liberties via BFS."""
        color = self.grid[row][col]
        if color == Stone.EMPTY:
            return set(), set()

        group: set[tuple[int, int]] = set()
        liberties: set[tuple[int, int]] = set()
        stack = [(row, col)]

        while stack:
            r, c = stack.pop()
            if (r, c) in group:
                continue
            group.add((r, c))
            for nr, nc in self._neighbors(r, c):
                cell = self.grid[nr][nc]
                if cell == Stone.EMPTY:
                    liberties.add((nr, nc))
                elif cell == color and (nr, nc) not in group:
                    stack.append((nr, nc))

        return group, liberties

    def _remove_group(self, group: set[tuple[int, int]]):
        """Remove all stones in a group from the board."""
        for r, c in group:
            self.grid[r][c] = Stone.EMPTY

    def place_stone(self, row: int, col: int, color: int) -> int:
        """Place a stone and handle captures.

        Returns:
            Number of opponent stones captured (0 if none).

        Raises:
            ValueError: if the move is illegal.
        """
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise ValueError(f"Position ({row}, {col}) out of bounds")
        if self.grid[row][col] != Stone.EMPTY:
            raise ValueError(f"Position ({row}, {col}) is occupied")
        if (row, col) == self.ko_point:
            raise ValueError(f"Position ({row}, {col}) is forbidden by ko rule")

        # Tentatively place stone
        self.grid[row][col] = color
        opponent = Stone.BLACK if color == Stone.WHITE else Stone.WHITE

        # Check captures of opponent groups adjacent to placed stone
        captured = 0
        captured_group: set[tuple[int, int]] = set()
        for nr, nc in self._neighbors(row, col):
            if self.grid[nr][nc] == opponent:
                group, liberties = self._find_group(nr, nc)
                if len(liberties) == 0:
                    captured += len(group)
                    captured_group |= group
                    self._remove_group(group)

        # Check suicide: if our own group has no liberties and no captures
        if captured == 0:
            _, own_liberties = self._find_group(row, col)
            if len(own_liberties) == 0:
                self.grid[row][col] = Stone.EMPTY
                raise ValueError(f"Suicide at ({row}, {col}) is not allowed")

        # Update ko point: ko requires ALL three conditions:
        # 1. Exactly 1 stone captured
        # 2. Capturing group is a single stone (not connected to friendly group)
        # 3. Capturing group has exactly 1 liberty (the captured position)
        if captured == 1:
            cap_pos = next(iter(captured_group))
            placed_group, placed_liberties = self._find_group(row, col)
            if len(placed_group) == 1 and len(placed_liberties) == 1:
                self.ko_point = cap_pos
            else:
                self.ko_point = None
        else:
            self.ko_point = None

        return captured

    def is_legal(self, row: int, col: int, color: int) -> bool:
        """Check if placing a stone at (row, col) is legal without modifying the board."""
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        if self.grid[row][col] != Stone.EMPTY:
            return False
        if (row, col) == self.ko_point:
            return False

        # Tentatively place and check
        self.grid[row][col] = color
        opponent = Stone.BLACK if color == Stone.WHITE else Stone.WHITE

        # Would it capture anything?
        captures = False
        for nr, nc in self._neighbors(row, col):
            if self.grid[nr][nc] == opponent:
                _, liberties = self._find_group(nr, nc)
                if len(liberties) == 0:
                    captures = True
                    break

        # If no captures, check suicide
        legal = True
        if not captures:
            _, own_liberties = self._find_group(row, col)
            if len(own_liberties) == 0:
                legal = False

        self.grid[row][col] = Stone.EMPTY
        return legal

    def get_legal_moves(self, color: int) -> list[tuple[int, int]]:
        """Return all legal positions for the given color."""
        moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.is_legal(r, c, color):
                    moves.append((r, c))
        return moves

    def score(self, komi: float = 7.5) -> tuple[float, float]:
        """Tromp-Taylor area scoring.

        Returns:
            (black_score, white_score) where white includes komi.
        """
        black_score = 0.0
        white_score = komi

        visited: set[tuple[int, int]] = set()

        for r in range(self.size):
            for c in range(self.size):
                cell = self.grid[r][c]
                if cell == Stone.BLACK:
                    black_score += 1
                elif cell == Stone.WHITE:
                    white_score += 1
                elif (r, c) not in visited:
                    # Flood-fill to find connected empty region
                    region: set[tuple[int, int]] = set()
                    borders: set[int] = set()
                    stack = [(r, c)]
                    while stack:
                        er, ec = stack.pop()
                        if (er, ec) in region:
                            continue
                        region.add((er, ec))
                        for nr, nc in self._neighbors(er, ec):
                            ncell = self.grid[nr][nc]
                            if ncell == Stone.EMPTY and (nr, nc) not in region:
                                stack.append((nr, nc))
                            elif ncell != Stone.EMPTY:
                                borders.add(ncell)
                    visited |= region
                    if borders == {Stone.BLACK}:
                        black_score += len(region)
                    elif borders == {Stone.WHITE}:
                        white_score += len(region)

        return black_score, white_score

    def to_grid_list(self) -> list[list[int]]:
        return [row[:] for row in self.grid]

    def __hash__(self) -> int:
        return hash(tuple(tuple(row) for row in self.grid))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Board):
            return False
        return self.grid == other.grid
