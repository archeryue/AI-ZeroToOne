"""Python tests: compare C++ engine vs Python Go engine for correctness."""

import sys
import random
import time
import numpy as np

# Add Python engine to path
sys.path.insert(0, "/Users/bytedance/AI-ZeroToOne/Season-5/Go/backend")

import go_engine
from engine.board import Board as PyBoard, Stone, PASS
from engine.game import Game as PyGame


def test_basic_operations():
    """Test basic stone placement matches between C++ and Python."""
    cpp = go_engine.Board9()
    py_b = PyBoard(9)

    # Place some stones
    moves = [(2, 3, go_engine.BLACK), (4, 5, go_engine.WHITE), (0, 0, go_engine.BLACK)]
    for r, c, clr in moves:
        py_color = Stone.BLACK if clr == go_engine.BLACK else Stone.WHITE
        cpp.place_stone(r, c, clr)
        py_b.place_stone(r, c, py_color)

    # Compare grids
    cpp_grid = np.array(cpp.board_grid())
    for r in range(9):
        for c in range(9):
            py_val = py_b.grid[r][c]
            cpp_val = cpp_grid[r, c]
            assert py_val == cpp_val, f"Mismatch at ({r},{c}): py={py_val} cpp={cpp_val}"

    print("  PASS: basic operations match")


def test_capture_match():
    """Test that captures produce identical results."""
    cpp = go_engine.Board9()
    py_b = PyBoard(9)

    # Surround a white stone
    moves = [
        (1, 1, go_engine.WHITE),
        (0, 1, go_engine.BLACK),
        (2, 1, go_engine.BLACK),
        (1, 0, go_engine.BLACK),
        (1, 2, go_engine.BLACK),  # captures
    ]
    for r, c, clr in moves:
        py_color = Stone.BLACK if clr == go_engine.BLACK else Stone.WHITE
        cpp_cap = cpp.place_stone(r, c, clr)
        py_cap = py_b.place_stone(r, c, py_color)
        assert cpp_cap == py_cap, f"Capture mismatch at ({r},{c}): cpp={cpp_cap} py={py_cap}"

    # Verify board matches
    cpp_grid = np.array(cpp.board_grid())
    for r in range(9):
        for c in range(9):
            assert py_b.grid[r][c] == cpp_grid[r, c], f"Board mismatch at ({r},{c})"

    print("  PASS: capture match")


def test_legal_moves_match():
    """Test legal moves match at every step of a random game."""
    random.seed(42)
    mismatches = 0

    for trial in range(200):
        cpp = go_engine.Board9()
        py_b = PyBoard(9)
        turn = go_engine.BLACK

        for move_num in range(100):
            py_turn = Stone.BLACK if turn == go_engine.BLACK else Stone.WHITE

            # Get legal moves from both
            cpp_legal = set(cpp.get_legal_moves(turn))
            py_legal = set(py_b.get_legal_moves(py_turn))

            if cpp_legal != py_legal:
                mismatches += 1
                # Print first mismatch detail for debugging
                if mismatches <= 3:
                    only_cpp = cpp_legal - py_legal
                    only_py = py_legal - cpp_legal
                    print(f"  Legal move mismatch trial={trial} move={move_num}")
                    if only_cpp: print(f"    Only in C++: {only_cpp}")
                    if only_py: print(f"    Only in Python: {only_py}")

            # Pick a random legal move (from intersection to avoid divergence)
            common = list(cpp_legal & py_legal)
            if not common or random.random() < 0.03:
                # Pass
                cpp = go_engine.Board9()  # can't pass on board directly, skip
                py_b.ko_point = None
                break
            else:
                r, c = random.choice(common)
                cpp.place_stone(r, c, turn)
                py_b.place_stone(r, c, py_turn)
                turn = go_engine.WHITE if turn == go_engine.BLACK else go_engine.BLACK

    assert mismatches == 0, f"Found {mismatches} legal move mismatches"
    print("  PASS: legal moves match (200 random games)")


def test_scoring_match():
    """Test scoring matches between engines."""
    random.seed(123)

    for trial in range(50):
        cpp = go_engine.Board9()
        py_b = PyBoard(9)
        turn = go_engine.BLACK

        # Play random moves
        for _ in range(random.randint(10, 80)):
            py_turn = Stone.BLACK if turn == go_engine.BLACK else Stone.WHITE
            cpp_legal = cpp.get_legal_moves(turn)
            py_legal = py_b.get_legal_moves(py_turn)
            common = list(set(cpp_legal) & set(py_legal))
            if not common:
                break
            r, c = random.choice(common)
            cpp.place_stone(r, c, turn)
            py_b.place_stone(r, c, py_turn)
            turn = go_engine.WHITE if turn == go_engine.BLACK else go_engine.BLACK

        # Compare scores
        cpp_bs, cpp_ws = cpp.score(5.5)
        py_bs, py_ws = py_b.score(5.5)
        assert cpp_bs == py_bs and cpp_ws == py_ws, \
            f"Score mismatch trial={trial}: cpp=({cpp_bs},{cpp_ws}) py=({py_bs},{py_ws})"

    print("  PASS: scoring match (50 random positions)")


def test_ko_match():
    """Test ko detection matches."""
    cpp = go_engine.Board9()
    py_b = PyBoard(9)

    # Classic ko setup
    moves = [
        (0, 1, go_engine.BLACK),
        (0, 2, go_engine.WHITE),
        (1, 0, go_engine.BLACK),
        (1, 1, go_engine.WHITE),
        (1, 3, go_engine.WHITE),
        (2, 1, go_engine.BLACK),
        (2, 2, go_engine.WHITE),
    ]
    for r, c, clr in moves:
        py_color = Stone.BLACK if clr == go_engine.BLACK else Stone.WHITE
        cpp.place_stone(r, c, clr)
        py_b.place_stone(r, c, py_color)

    # Black captures at (1,2)
    cpp_cap = cpp.place_stone(1, 2, go_engine.BLACK)
    py_cap = py_b.place_stone(1, 2, Stone.BLACK)
    assert cpp_cap == py_cap == 1

    # Ko: white can't recapture at (1,1)
    assert not cpp.is_legal(1, 1, go_engine.WHITE)
    assert not py_b.is_legal(1, 1, Stone.WHITE)

    print("  PASS: ko match")


def test_observation_shape():
    """Test observation tensor shape and basic correctness."""
    g = go_engine.Game9(5.5)
    g.make_move(2, 3)  # black
    g.make_move(4, 5)  # white

    obs = g.to_observation()
    assert obs.shape == (17, 9, 9), f"Bad shape: {obs.shape}"

    # Current player is black (after 2 moves: black, white, now black's turn? No...)
    # After black plays, white plays, it's black's turn
    # Wait — Game starts with BLACK, make_move switches turn
    # Move 1: black plays (2,3), turn → WHITE
    # Move 2: white plays (4,5), turn → BLACK
    # So current_turn = BLACK
    assert g.current_turn == go_engine.BLACK

    # Plane 0: current player (BLACK) stones → (2,3) should be 1
    assert obs[0, 2, 3] == 1.0
    # Plane 1: opponent (WHITE) stones → (4,5) should be 1
    assert obs[1, 4, 5] == 1.0
    # Plane 16: color = BLACK → all 1.0
    assert obs[16, 0, 0] == 1.0

    print("  PASS: observation encoding")


def test_action_mask():
    """Test action mask shape and correctness."""
    g = go_engine.Game9(5.5)
    mask = g.get_action_mask()
    assert mask.shape == (82,)
    assert mask[81] == True  # pass always legal
    assert all(mask[:81])    # all positions legal on empty board

    g.make_move(4, 4)
    mask = g.get_action_mask()
    assert mask[4 * 9 + 4] == False  # occupied

    print("  PASS: action mask")


def bench_comparison():
    """Benchmark C++ vs Python engine."""
    print("\n--- Speed comparison ---")

    # C++ random games
    random.seed(42)
    t0 = time.time()
    cpp_games = 1000
    for _ in range(cpp_games):
        g = go_engine.Game9(5.5)
        for move in range(150):
            if g.status != go_engine.PLAYING:
                break
            legal = g.get_legal_moves()
            if not legal or random.random() < 0.03:
                g.pass_move()
            else:
                r, c = random.choice(legal)
                g.make_move(r, c)
    cpp_time = time.time() - t0

    # Python random games
    random.seed(42)
    t0 = time.time()
    py_games = 1000
    for _ in range(py_games):
        g = PyGame(9, 5.5)
        for move in range(150):
            if g.status.value != "playing":
                break
            legal = g.get_legal_moves()
            if not legal or random.random() < 0.03:
                g.make_move(-1, -1)
            else:
                r, c = random.choice(legal)
                g.make_move(r, c)
    py_time = time.time() - t0

    speedup = py_time / cpp_time
    print(f"  C++:    {cpp_games} games in {cpp_time:.2f}s ({cpp_games/cpp_time:.0f} games/sec)")
    print(f"  Python: {py_games} games in {py_time:.2f}s ({py_games/py_time:.0f} games/sec)")
    print(f"  Speedup: {speedup:.1f}x")


if __name__ == "__main__":
    print("=== C++ vs Python Engine Comparison ===\n")

    test_basic_operations()
    test_capture_match()
    test_legal_moves_match()
    test_scoring_match()
    test_ko_match()
    test_observation_shape()
    test_action_mask()

    bench_comparison()

    print("\n=== All comparison tests passed! ===")
