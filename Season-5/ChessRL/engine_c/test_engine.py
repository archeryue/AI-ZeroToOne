"""Correctness tests: compare C++ engine vs Python engine."""

import sys
import time
import random
import numpy as np

# Python engine
sys.path.insert(0, "/home/start-up/AI-ZeroToOne/Season-5/ChessRL")
sys.path.insert(0, "/home/start-up/AI-ZeroToOne/Season-5/ChineseChess/backend")

from engine.board import Board as PyBoard, RED as PY_RED, BLACK as PY_BLACK
from engine.game import Game as PyGame, GameStatus
from engine.rules import get_legal_moves as py_get_legal_moves, is_in_check as py_is_in_check
from env.observation import board_to_observation as py_board_to_obs
from env.action_space import get_action_mask as py_get_action_mask, encode_move as py_encode_move

# C++ engine
import engine_c as cc


def py_board_to_cc(py_board):
    """Convert Python Board to C++ Board."""
    return cc.Board(py_board.grid)


def test_initial_board():
    """Test initial board matches."""
    py_b = PyBoard()
    cc_b = cc.Board()

    # Compare FEN
    assert py_b.to_fen() == cc_b.to_fen(), f"FEN mismatch: {py_b.to_fen()} vs {cc_b.to_fen()}"

    # Compare all squares
    for r in range(10):
        for c in range(9):
            assert py_b.get(r, c) == cc_b.get(r, c), f"Mismatch at ({r},{c})"

    # Compare general positions
    assert py_b.find_general(PY_RED) == cc_b.find_general(cc.RED)
    assert py_b.find_general(PY_BLACK) == cc_b.find_general(cc.BLACK)

    print("  [PASS] Initial board")


def test_legal_moves_initial():
    """Test legal moves from initial position."""
    py_b = PyBoard()
    cc_b = cc.Board()

    py_moves = py_get_legal_moves(py_b, PY_RED)
    cc_moves = cc.get_legal_moves(cc_b, cc.RED)

    py_set = {(m.from_row, m.from_col, m.to_row, m.to_col) for m in py_moves}
    cc_set = {(m.from_row, m.from_col, m.to_row, m.to_col) for m in cc_moves}

    assert py_set == cc_set, f"Move mismatch!\nPython only: {py_set - cc_set}\nC++ only: {cc_set - py_set}"
    print(f"  [PASS] Initial legal moves (Red): {len(py_moves)} moves")

    # Black too
    py_moves_b = py_get_legal_moves(py_b, PY_BLACK)
    cc_moves_b = cc.get_legal_moves(cc_b, cc.BLACK)
    py_set_b = {(m.from_row, m.from_col, m.to_row, m.to_col) for m in py_moves_b}
    cc_set_b = {(m.from_row, m.from_col, m.to_row, m.to_col) for m in cc_moves_b}
    assert py_set_b == cc_set_b
    print(f"  [PASS] Initial legal moves (Black): {len(py_moves_b)} moves")


def test_observation():
    """Test board_to_observation matches."""
    py_b = PyBoard()
    cc_b = cc.Board()

    py_obs = py_board_to_obs(py_b, PY_RED)
    cc_obs = cc.board_to_observation(cc_b, cc.RED)
    assert np.allclose(py_obs, cc_obs), "Observation mismatch (Red turn)"

    py_obs_b = py_board_to_obs(py_b, PY_BLACK)
    cc_obs_b = cc.board_to_observation(cc_b, cc.BLACK)
    assert np.allclose(py_obs_b, cc_obs_b), "Observation mismatch (Black turn)"

    print("  [PASS] board_to_observation")


def test_action_mask():
    """Test get_action_mask matches."""
    py_b = PyBoard()
    cc_b = cc.Board()

    py_mask = py_get_action_mask(py_b, PY_RED)
    cc_mask = cc.get_action_mask(cc_b, cc.RED)
    assert np.array_equal(py_mask, cc_mask), f"Mask mismatch: py={py_mask.sum()}, cc={cc_mask.sum()}"

    print("  [PASS] get_action_mask")


def test_random_games(num_games=100, max_steps=200):
    """Play random games with both engines, verify agreement at every step."""
    mismatches = 0

    for game_idx in range(num_games):
        py_game = PyGame()
        cc_game = cc.Game()

        for step in range(max_steps):
            # Compare board state
            py_fen = py_game.board.to_fen()
            cc_fen = cc_game.board.to_fen()
            if py_fen != cc_fen:
                print(f"  [FAIL] Game {game_idx} step {step}: FEN mismatch")
                print(f"    Python: {py_fen}")
                print(f"    C++:    {cc_fen}")
                mismatches += 1
                break

            # Compare game status
            py_playing = py_game.status == GameStatus.PLAYING
            cc_playing = cc_game.status == cc.STATUS_PLAYING
            if py_playing != cc_playing:
                print(f"  [FAIL] Game {game_idx} step {step}: Status mismatch")
                print(f"    Python: {py_game.status}, C++: {cc_game.status}")
                mismatches += 1
                break

            if not py_playing:
                break

            # Get legal moves from both
            py_moves = py_get_legal_moves(py_game.board, py_game.current_turn)
            cc_b = py_board_to_cc(py_game.board)
            cc_moves = cc.get_legal_moves(cc_b, cc_game.current_turn)

            py_set = {(m.from_row, m.from_col, m.to_row, m.to_col) for m in py_moves}
            cc_set = {(m.from_row, m.from_col, m.to_row, m.to_col) for m in cc_moves}

            if py_set != cc_set:
                print(f"  [FAIL] Game {game_idx} step {step}: Legal move mismatch")
                print(f"    Python only: {py_set - cc_set}")
                print(f"    C++ only: {cc_set - py_set}")
                print(f"    FEN: {py_fen}")
                mismatches += 1
                break

            if not py_moves:
                break

            # Pick random move
            move = random.choice(py_moves)
            fr, fc, tr, tc = move.from_row, move.from_col, move.to_row, move.to_col

            py_game.make_move(fr, fc, tr, tc)
            cc_game.make_move(fr, fc, tr, tc)

    if mismatches == 0:
        print(f"  [PASS] {num_games} random games (up to {max_steps} steps each)")
    else:
        print(f"  [FAIL] {mismatches}/{num_games} games had mismatches")

    return mismatches == 0


def test_simulate_action():
    """Test simulate_action matches Python _simulate_game pattern."""
    from agents.alphazero.mcts import _simulate_game as py_simulate
    from env.action_space import encode_move, decode_action

    py_game = PyGame()
    cc_game = cc.Game()

    # Make a few moves
    moves_to_play = [(6, 4, 5, 4), (3, 4, 4, 4), (7, 1, 4, 1)]
    for fr, fc, tr, tc in moves_to_play:
        py_game.make_move(fr, fc, tr, tc)
        cc_game.make_move(fr, fc, tr, tc)

    # Now test simulate_action
    py_legal = py_get_legal_moves(py_game.board, py_game.current_turn)
    for move in py_legal[:10]:
        action = encode_move(move.from_row, move.from_col, move.to_row, move.to_col)

        py_sim = py_simulate(py_game, action)
        cc_sim = cc_game.simulate_action(action)

        py_fen = py_sim.board.to_fen()
        cc_fen = cc_sim.board.to_fen()
        assert py_fen == cc_fen, f"simulate_action FEN mismatch for action {action}"

    print("  [PASS] simulate_action")


def benchmark():
    """Benchmark C++ vs Python engine."""
    print("\n--- Performance Benchmark ---")

    # get_legal_moves from initial position
    py_b = PyBoard()
    cc_b = cc.Board()
    N = 10000

    t0 = time.time()
    for _ in range(N):
        py_get_legal_moves(py_b, PY_RED)
    py_time = time.time() - t0

    t0 = time.time()
    for _ in range(N):
        cc.get_legal_moves(cc_b, cc.RED)
    cc_time = time.time() - t0

    print(f"  get_legal_moves x{N}: Python={py_time:.3f}s, C++={cc_time:.3f}s, "
          f"speedup={py_time/cc_time:.1f}x")

    # board_to_observation
    t0 = time.time()
    for _ in range(N):
        py_board_to_obs(py_b, PY_RED)
    py_time = time.time() - t0

    t0 = time.time()
    for _ in range(N):
        cc.board_to_observation(cc_b, cc.RED)
    cc_time = time.time() - t0

    print(f"  board_to_observation x{N}: Python={py_time:.3f}s, C++={cc_time:.3f}s, "
          f"speedup={py_time/cc_time:.1f}x")

    # get_action_mask
    t0 = time.time()
    for _ in range(N):
        py_get_action_mask(py_b, PY_RED)
    py_time = time.time() - t0

    t0 = time.time()
    for _ in range(N):
        cc.get_action_mask(cc_b, cc.RED)
    cc_time = time.time() - t0

    print(f"  get_action_mask x{N}: Python={py_time:.3f}s, C++={cc_time:.3f}s, "
          f"speedup={py_time/cc_time:.1f}x")

    # Full random game
    num_games = 500
    t0 = time.time()
    for _ in range(num_games):
        g = PyGame()
        for _ in range(200):
            if g.status != GameStatus.PLAYING:
                break
            moves = g.get_legal_moves()
            if not moves:
                break
            m = random.choice(moves)
            g.make_move(m.from_row, m.from_col, m.to_row, m.to_col)
    py_time = time.time() - t0

    t0 = time.time()
    for _ in range(num_games):
        g = cc.Game()
        for _ in range(200):
            if g.status != cc.STATUS_PLAYING:
                break
            moves = cc.get_legal_moves(g.board, g.current_turn)
            if not moves:
                break
            m = random.choice(moves)
            g.make_move(m.from_row, m.from_col, m.to_row, m.to_col)
    cc_time = time.time() - t0

    print(f"  Full random game x{num_games}: Python={py_time:.3f}s, C++={cc_time:.3f}s, "
          f"speedup={py_time/cc_time:.1f}x")


if __name__ == "__main__":
    random.seed(42)

    print("=== Correctness Tests ===")
    test_initial_board()
    test_legal_moves_initial()
    test_observation()
    test_action_mask()
    test_simulate_action()
    all_pass = test_random_games(100, 200)

    if all_pass:
        benchmark()
    else:
        print("\nSkipping benchmark due to correctness failures.")
