"""Tests for pieces.py — movement rules."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.board import Board, RED, BLACK, GENERAL, ADVISOR, ELEPHANT, HORSE, CHARIOT, CANNON, SOLDIER
from engine.pieces import (
    general_moves, advisor_moves, elephant_moves, horse_moves,
    chariot_moves, cannon_moves, soldier_moves, get_piece_moves,
)


def empty_board():
    return Board([[0]*9 for _ in range(10)])


class TestGeneral:
    def test_center_of_palace(self):
        board = empty_board()
        board.set(8, 4, GENERAL)
        moves = general_moves(board, 8, 4, RED)
        assert (7, 4) in moves
        assert (9, 4) in moves
        assert (8, 3) in moves
        assert (8, 5) in moves
        assert len(moves) == 4

    def test_corner_of_palace(self):
        board = empty_board()
        board.set(7, 3, GENERAL)
        moves = general_moves(board, 7, 3, RED)
        assert (7, 4) in moves
        assert (8, 3) in moves
        # Cannot go outside palace
        assert (7, 2) not in moves
        assert (6, 3) not in moves


class TestAdvisor:
    def test_center_of_palace(self):
        board = empty_board()
        board.set(8, 4, ADVISOR)
        moves = advisor_moves(board, 8, 4, RED)
        assert len(moves) == 4
        assert (7, 3) in moves
        assert (7, 5) in moves
        assert (9, 3) in moves
        assert (9, 5) in moves


class TestElephant:
    def test_basic_move(self):
        board = empty_board()
        board.set(7, 2, ELEPHANT * RED)
        moves = elephant_moves(board, 7, 2, RED)
        assert (5, 0) in moves
        assert (5, 4) in moves
        assert (9, 0) in moves
        assert (9, 4) in moves

    def test_blocked_eye(self):
        board = empty_board()
        board.set(7, 2, ELEPHANT * RED)
        board.set(6, 1, SOLDIER)  # block the eye
        moves = elephant_moves(board, 7, 2, RED)
        assert (5, 0) not in moves  # blocked

    def test_cannot_cross_river(self):
        board = empty_board()
        board.set(5, 2, ELEPHANT * RED)
        moves = elephant_moves(board, 5, 2, RED)
        # All diagonal moves from (5,2) go to row 3 or row 7
        for r, c in moves:
            assert r >= 5  # Red elephant stays on Red side


class TestHorse:
    def test_center_moves(self):
        board = empty_board()
        board.set(5, 4, HORSE)
        moves = horse_moves(board, 5, 4, RED)
        assert len(moves) == 8  # all 8 L-shaped positions

    def test_blocked_leg(self):
        board = empty_board()
        board.set(5, 4, HORSE)
        board.set(4, 4, SOLDIER)  # block upward
        moves = horse_moves(board, 5, 4, RED)
        assert (3, 3) not in moves
        assert (3, 5) not in moves
        assert len(moves) == 6  # 2 blocked


class TestChariot:
    def test_open_board(self):
        board = empty_board()
        board.set(5, 4, CHARIOT)
        moves = chariot_moves(board, 5, 4, RED)
        # Can reach all squares in row 5 and column 4
        assert len(moves) == 8 + 9  # 8 horizontal + 9 vertical = 17

    def test_blocked_by_piece(self):
        board = empty_board()
        board.set(5, 4, CHARIOT)
        board.set(5, 6, -SOLDIER)  # enemy piece
        moves = chariot_moves(board, 5, 4, RED)
        assert (5, 5) in moves  # can move to empty square
        assert (5, 6) in moves  # can capture
        assert (5, 7) not in moves  # blocked


class TestCannon:
    def test_non_capture_moves(self):
        board = empty_board()
        board.set(5, 4, CANNON)
        # Without any other pieces, cannon moves like a chariot for non-capture
        moves = cannon_moves(board, 5, 4, RED)
        assert len(moves) == 17

    def test_capture_over_platform(self):
        board = empty_board()
        board.set(5, 4, CANNON)
        board.set(5, 6, SOLDIER)  # platform
        board.set(5, 8, -SOLDIER)  # target behind platform
        moves = cannon_moves(board, 5, 4, RED)
        assert (5, 5) in moves  # non-capture
        assert (5, 6) not in moves  # platform, can't go there
        assert (5, 7) not in moves  # behind platform, empty
        assert (5, 8) in moves  # capture over platform


class TestSoldier:
    def test_before_river_red(self):
        board = empty_board()
        board.set(6, 4, SOLDIER)
        moves = soldier_moves(board, 6, 4, RED)
        assert moves == [(5, 4)]  # forward only

    def test_after_river_red(self):
        board = empty_board()
        board.set(4, 4, SOLDIER)
        moves = soldier_moves(board, 4, 4, RED)
        assert (3, 4) in moves  # forward
        assert (4, 3) in moves  # left
        assert (4, 5) in moves  # right
        assert len(moves) == 3

    def test_before_river_black(self):
        board = empty_board()
        board.set(3, 4, -SOLDIER)
        moves = soldier_moves(board, 3, 4, BLACK)
        assert moves == [(4, 4)]  # forward only

    def test_after_river_black(self):
        board = empty_board()
        board.set(5, 4, -SOLDIER)
        moves = soldier_moves(board, 5, 4, BLACK)
        assert (6, 4) in moves  # forward
        assert (5, 3) in moves  # left
        assert (5, 5) in moves  # right
