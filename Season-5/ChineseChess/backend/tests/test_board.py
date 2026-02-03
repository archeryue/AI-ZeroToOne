"""Tests for board.py."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.board import Board, ROWS, COLS, RED, BLACK, GENERAL, CHARIOT


class TestBoard:
    def test_initial_board_dimensions(self):
        board = Board()
        assert len(board.grid) == ROWS
        assert all(len(row) == COLS for row in board.grid)

    def test_initial_red_pieces(self):
        board = Board()
        # Red back rank (row 9)
        assert board.get(9, 4) == GENERAL  # Red general at center
        assert board.get(9, 0) == CHARIOT  # Red chariot at corner
        assert board.get(9, 8) == CHARIOT

    def test_initial_black_pieces(self):
        board = Board()
        # Black back rank (row 0)
        assert board.get(0, 4) == -GENERAL  # Black general
        assert board.get(0, 0) == -CHARIOT

    def test_get_set(self):
        board = Board()
        board.set(5, 4, GENERAL)
        assert board.get(5, 4) == GENERAL

    def test_color_of(self):
        board = Board()
        assert board.color_of(9, 4) == RED
        assert board.color_of(0, 4) == BLACK
        assert board.color_of(5, 4) == 0  # empty

    def test_find_general(self):
        board = Board()
        assert board.find_general(RED) == (9, 4)
        assert board.find_general(BLACK) == (0, 4)

    def test_copy_independence(self):
        board = Board()
        copy = board.copy()
        copy.set(9, 4, 0)
        assert board.get(9, 4) == GENERAL  # original unchanged

    def test_fen_roundtrip(self):
        board = Board()
        fen = board.to_fen()
        restored = Board.from_fen(fen)
        assert board == restored

    def test_in_bounds(self):
        board = Board()
        assert board.in_bounds(0, 0)
        assert board.in_bounds(9, 8)
        assert not board.in_bounds(-1, 0)
        assert not board.in_bounds(10, 0)
        assert not board.in_bounds(0, 9)

    def test_piece_type(self):
        board = Board()
        assert board.piece_type(9, 4) == GENERAL
        assert board.piece_type(0, 4) == GENERAL
        assert board.piece_type(5, 4) == 0
