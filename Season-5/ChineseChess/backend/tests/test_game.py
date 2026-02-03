"""Tests for game.py — game session management."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.board import RED, BLACK, CHARIOT, CANNON
from engine.game import Game, GameStatus


class TestGame:
    def test_initial_state(self):
        game = Game()
        assert game.current_turn == RED
        assert game.status == GameStatus.PLAYING
        assert not game.is_in_check()

    def test_legal_moves_exist(self):
        game = Game()
        moves = game.get_legal_moves()
        assert len(moves) > 0

    def test_make_valid_move(self):
        game = Game()
        # Move Red chariot: (9,0) -> (8,0)
        move = game.make_move(9, 0, 8, 0)
        assert move is not None
        assert game.current_turn == BLACK

    def test_make_invalid_move(self):
        game = Game()
        # Try to move empty square
        move = game.make_move(5, 4, 4, 4)
        assert move is None
        assert game.current_turn == RED  # turn unchanged

    def test_undo(self):
        game = Game()
        game.make_move(9, 0, 8, 0)  # Red chariot moves
        assert game.current_turn == BLACK
        undone = game.undo()
        assert undone is not None
        assert game.current_turn == RED
        assert game.board.get(9, 0) == CHARIOT

    def test_undo_empty(self):
        game = Game()
        assert game.undo() is None

    def test_resign_red(self):
        game = Game()
        game.resign(RED)
        assert game.status == GameStatus.BLACK_WIN

    def test_resign_black(self):
        game = Game()
        game.resign(BLACK)
        assert game.status == GameStatus.RED_WIN

    def test_no_moves_after_game_over(self):
        game = Game()
        game.resign(RED)
        assert game.get_legal_moves() == []

    def test_to_dict(self):
        game = Game()
        d = game.to_dict()
        assert "board" in d
        assert "current_turn" in d
        assert "status" in d
        assert "valid_moves" in d
        assert "move_history" in d
        assert d["current_turn"] == "red"
        assert d["status"] == "playing"

    def test_get_legal_moves_dict(self):
        game = Game()
        moves_dict = game.get_legal_moves_dict()
        assert isinstance(moves_dict, dict)
        # Chariot at (9,0) should have at least one move
        assert "9,0" in moves_dict

    def test_cannon_opening(self):
        """Test a common cannon opening move."""
        game = Game()
        # Move Red cannon from (7,1) to (7,4) — 当头炮
        move = game.make_move(7, 1, 7, 4)
        assert move is not None
        assert game.board.get(7, 4) == CANNON
        assert game.board.get(7, 1) == 0

    def test_multiple_moves(self):
        game = Game()
        # Red cannon
        assert game.make_move(7, 1, 7, 4) is not None  # Red
        # Black horse
        assert game.make_move(0, 1, 2, 2) is not None  # Black
        # Red horse
        assert game.make_move(9, 1, 7, 2) is not None  # Red
        assert game.current_turn == BLACK
        assert len(game.move_history) == 3
