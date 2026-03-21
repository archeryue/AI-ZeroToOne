"""
Evaluate NNUE with C++ alpha-beta search against various opponents.

Usage:
    python eval_nnue_cpp.py [--games 100] [--depth 4] [--opponent all]
"""

import os
import sys
import argparse
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))

import engine_c._xiangqi as xq
from engine_c._xiangqi import (
    Board, Game, NNUESearch, get_legal_moves,
    RED, BLACK, STATUS_PLAYING, STATUS_RED_WIN, STATUS_BLACK_WIN, STATUS_DRAW,
    GENERAL, ADVISOR, ELEPHANT, HORSE, CHARIOT, CANNON, SOLDIER,
)

# ─── Material eval for Greedy/Minimax opponents ─────────────────────────────

PIECE_VALUES = {GENERAL: 0, ADVISOR: 2, ELEPHANT: 2, HORSE: 4,
                CHARIOT: 9, CANNON: 4.5, SOLDIER: 1}


def material_eval(board, color):
    score = 0.0
    for r in range(10):
        for c in range(9):
            piece = board.get(r, c)
            if piece == 0:
                continue
            pt = abs(piece)
            val = PIECE_VALUES.get(pt, 0)
            if (piece > 0 and color == RED) or (piece < 0 and color == BLACK):
                score += val
            else:
                score -= val
    return score


# ─── Opponents ───────────────────────────────────────────────────────────────

def random_move(board, color):
    moves = get_legal_moves(board, color)
    if not moves:
        return None
    m = moves[np.random.randint(len(moves))]
    return (m.from_sq // 9, m.from_sq % 9, m.to_sq // 9, m.to_sq % 9)


def greedy_move(board, color):
    moves = get_legal_moves(board, color)
    if not moves:
        return None
    best_val = -1e9
    best_move = None
    for m in moves:
        fr, fc = m.from_sq // 9, m.from_sq % 9
        tr, tc = m.to_sq // 9, m.to_sq % 9
        b2 = board.copy()
        b2.set(fr, fc, 0)
        b2.set(tr, tc, board.get(fr, fc))
        val = material_eval(b2, color)
        if val > best_val:
            best_val = val
            best_move = (fr, fc, tr, tc)
    return best_move


def minimax_move(board, color, depth=3):
    def _search(board, color, d, alpha, beta, maximizing):
        if d == 0:
            return material_eval(board, color), None
        side = color if maximizing else -color
        moves = get_legal_moves(board, side)
        if not moves:
            return (-1000 if maximizing else 1000), None
        best_move = None
        if maximizing:
            best_val = -1e9
            for m in moves:
                fr, fc = m.from_sq // 9, m.from_sq % 9
                tr, tc = m.to_sq // 9, m.to_sq % 9
                b2 = board.copy()
                b2.set(fr, fc, 0)
                b2.set(tr, tc, board.get(fr, fc))
                val, _ = _search(b2, color, d - 1, alpha, beta, False)
                if val > best_val:
                    best_val = val
                    best_move = (fr, fc, tr, tc)
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return best_val, best_move
        else:
            best_val = 1e9
            for m in moves:
                fr, fc = m.from_sq // 9, m.from_sq % 9
                tr, tc = m.to_sq // 9, m.to_sq % 9
                b2 = board.copy()
                b2.set(fr, fc, 0)
                b2.set(tr, tc, board.get(fr, fc))
                val, _ = _search(b2, color, d - 1, alpha, beta, True)
                if val < best_val:
                    best_val = val
                    best_move = (fr, fc, tr, tc)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return best_val, best_move

    _, move = _search(board, color, depth, -1e9, 1e9, True)
    return move


# ─── Game Loop ───────────────────────────────────────────────────────────────

def play_game(search_engine, depth, opponent_fn, nnue_color=RED, max_moves=200):
    game = Game()
    current = RED

    for step in range(max_moves):
        if game.status != STATUS_PLAYING:
            break

        moves = get_legal_moves(game.board, current)
        if not moves:
            if current == nnue_color:
                return 'opponent_win'
            else:
                return 'nnue_win'

        if current == nnue_color:
            result = search_engine.search(game.board, current, depth)
            move = (result['from_row'], result['from_col'],
                    result['to_row'], result['to_col'])
        else:
            move = opponent_fn(game.board, current)

        if move is None:
            if current == nnue_color:
                return 'opponent_win'
            else:
                return 'nnue_win'

        fr, fc, tr, tc = move
        success = game.make_move(fr, fc, tr, tc)
        if not success:
            if current == nnue_color:
                return 'opponent_win'
            else:
                return 'nnue_win'

        current = -current

    if game.status == STATUS_RED_WIN:
        return 'nnue_win' if nnue_color == RED else 'opponent_win'
    elif game.status == STATUS_BLACK_WIN:
        return 'nnue_win' if nnue_color == BLACK else 'opponent_win'
    else:
        return 'draw'


def evaluate(search_engine, depth, opponent_name, opponent_fn, num_games=100):
    results = {'nnue_win': 0, 'opponent_win': 0, 'draw': 0}

    t0 = time.time()
    for i in range(num_games):
        nnue_color = RED if i % 2 == 0 else BLACK
        result = play_game(search_engine, depth, opponent_fn, nnue_color=nnue_color)
        results[result] += 1

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{num_games}] "
                  f"W:{results['nnue_win']} L:{results['opponent_win']} D:{results['draw']} "
                  f"({elapsed:.0f}s)")

    total = time.time() - t0
    win_rate = results['nnue_win'] / num_games * 100
    loss_rate = results['opponent_win'] / num_games * 100
    draw_rate = results['draw'] / num_games * 100

    print(f"\n=== NNUE (depth {depth}) vs {opponent_name} ({num_games} games) ===")
    print(f"  Wins:   {results['nnue_win']:3d} ({win_rate:.1f}%)")
    print(f"  Losses: {results['opponent_win']:3d} ({loss_rate:.1f}%)")
    print(f"  Draws:  {results['draw']:3d} ({draw_rate:.1f}%)")
    print(f"  Time:   {total:.0f}s ({total/num_games:.1f}s/game)")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--opponent', type=str, default='all',
                        choices=['random', 'greedy', 'minimax', 'all'])
    parser.add_argument('--weights', type=str,
                        default=os.path.join(SCRIPT_DIR, 'checkpoints', 'nnue_weights.bin'))
    args = parser.parse_args()

    print(f"Loading NNUE weights: {args.weights}")
    search = NNUESearch()
    ok = search.load_weights(args.weights)
    if not ok:
        print("ERROR: Failed to load weights")
        return
    print(f"Search depth: {args.depth}")

    # Sanity check
    game = Game()
    val = search.evaluate(game.board, RED)
    print(f"Starting position eval (Red STM): {val:.4f}")
    r = search.search(game.board, RED, args.depth)
    print(f"Best opening move: ({r['from_row']},{r['from_col']})->({r['to_row']},{r['to_col']}) "
          f"score={r['score']:.4f} nodes={r['nodes']:,}")
    print()

    opponents = {
        'random': ('Random', random_move),
        'greedy': ('Greedy (material)', greedy_move),
    }

    if args.opponent == 'all':
        for name, (label, fn) in opponents.items():
            evaluate(search, args.depth, label, fn, args.games)
        print("\n--- Minimax (depth 3) ---")
        evaluate(search, args.depth, 'Minimax-d3',
                 lambda b, c: minimax_move(b, c, depth=3),
                 min(args.games, 20))
    elif args.opponent == 'minimax':
        evaluate(search, args.depth, 'Minimax-d3',
                 lambda b, c: minimax_move(b, c, depth=3),
                 args.games)
    else:
        label, fn = opponents[args.opponent]
        evaluate(search, args.depth, label, fn, args.games)


if __name__ == "__main__":
    main()
