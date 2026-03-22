"""
Generate TD training data via self-play with HIGH NOISE using v1 NNUE engine.

Same as candidate5_v2/gen_td_data.py but with much more epsilon-greedy noise
to test whether position diversity is the key factor in training quality.

Noise schedule (vs v2 baseline):
  v2: moves 1-6 random, 7-16 eps=0.15, 17+ eps=0.05
  v4: moves 1-10 random, 11-30 eps=0.30, 31+ eps=0.15

Usage:
    python gen_td_data_noisy.py [--games 5000] [--depth 4] [--workers 12]
"""

import os
import sys
import argparse
import time
import numpy as np
from multiprocessing import Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_DIR)


def board_to_flat(board):
    """Convert Board object to flat int8[90] array."""
    flat = np.zeros(90, dtype=np.int8)
    for r in range(10):
        for c in range(9):
            flat[r * 9 + c] = board.get(r, c)
    return flat


def play_game_with_scores(args_tuple):
    """
    Play a self-play game with HIGH NOISE.

    Noise schedule:
      - Moves 1-10: fully random (no search)
      - Moves 11-30: search + epsilon=0.30
      - Moves 31+:   search + epsilon=0.15
    """
    game_id, weights_path, depth, max_moves = args_tuple

    from engine_c._xiangqi import (
        Game, NNUESearch, get_legal_moves,
        RED, BLACK, STATUS_PLAYING, STATUS_RED_WIN, STATUS_BLACK_WIN,
    )

    rng = np.random.RandomState(game_id)

    engine = NNUESearch()
    engine.load_weights(weights_path)

    game = Game()
    current = RED

    boards = []
    turns = []
    scores = []

    for step in range(max_moves):
        if game.status != STATUS_PLAYING:
            break

        moves = get_legal_moves(game.board, current)
        if not moves:
            break

        # HIGH NOISE: 10 fully random moves (vs 6 in v2)
        if step < 10:
            m = moves[rng.randint(len(moves))]
            fr, fc = m.from_sq // 9, m.from_sq % 9
            tr, tc = m.to_sq // 9, m.to_sq % 9
            boards.append(board_to_flat(game.board))
            turns.append(current)
            scores.append(0.0)
        else:
            boards.append(board_to_flat(game.board))
            turns.append(current)

            result = engine.search(game.board, current, depth)
            scores.append(result['score'])

            # HIGH NOISE: eps=0.30 for moves 11-30, eps=0.15 for 31+
            # (vs eps=0.15 for 7-16, eps=0.05 for 17+ in v2)
            epsilon = 0.30 if step < 30 else 0.15
            if rng.random() < epsilon:
                m = moves[rng.randint(len(moves))]
                fr, fc = m.from_sq // 9, m.from_sq % 9
                tr, tc = m.to_sq // 9, m.to_sq % 9
            else:
                fr, fc = result['from_row'], result['from_col']
                tr, tc = result['to_row'], result['to_col']

        success = game.make_move(fr, fc, tr, tc)
        if not success:
            break

        current = -current

    if game.status == STATUS_RED_WIN:
        outcome = 1.0
    elif game.status == STATUS_BLACK_WIN:
        outcome = -1.0
    else:
        outcome = 0.0

    # Convert scores to Red-relative [-1, 1]
    for i in range(len(boards)):
        abs_score = (2.0 * scores[i] - 1.0)
        if turns[i] == BLACK:
            abs_score = -abs_score
        scores[i] = abs_score

    return {
        'game_id': game_id,
        'boards': boards,
        'turns': turns,
        'scores': scores,
        'outcome': outcome,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate HIGH-NOISE TD self-play data")
    parser.add_argument('--games', type=int, default=5000, help='Number of self-play games')
    parser.add_argument('--depth', type=int, default=4, help='Search depth')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--weights', type=str,
                        default=os.path.join(PROJECT_DIR, 'training/candidate5/checkpoints/nnue_weights.bin'),
                        help='Path to v1 NNUE weights')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(SCRIPT_DIR, 'td_data_noisy'),
                        help='Output directory')
    parser.add_argument('--shard_size', type=int, default=1000)
    parser.add_argument('--max_moves', type=int, default=200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== HIGH-NOISE self-play data generation ===")
    print(f"Noise: moves 1-10 random, 11-30 eps=0.30, 31+ eps=0.15")
    print(f"Weights: {args.weights}")
    print(f"Search depth: {args.depth}")
    print(f"Games: {args.games}, Workers: {args.workers}")
    print(flush=True)

    tasks = [(i, args.weights, args.depth, args.max_moves) for i in range(args.games)]

    total_positions = 0
    total_games = 0
    outcomes = {'red_win': 0, 'black_win': 0, 'draw': 0}

    shard_boards = []
    shard_turns = []
    shard_scores = []
    shard_game_ids = []
    shard_idx = 0
    games_in_shard = 0

    t0 = time.time()

    with Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(play_game_with_scores, tasks, chunksize=4):
            if len(result['boards']) == 0:
                continue

            shard_boards.extend(result['boards'])
            shard_turns.extend(result['turns'])
            shard_scores.extend(result['scores'])
            shard_game_ids.extend([result['game_id']] * len(result['boards']))

            total_positions += len(result['boards'])
            total_games += 1
            games_in_shard += 1

            if result['outcome'] > 0:
                outcomes['red_win'] += 1
            elif result['outcome'] < 0:
                outcomes['black_win'] += 1
            else:
                outcomes['draw'] += 1

            if games_in_shard >= args.shard_size:
                shard_path = os.path.join(args.output_dir, f"td_shard_{shard_idx:04d}.npz")
                np.savez_compressed(shard_path,
                    boards=np.array(shard_boards, dtype=np.int8),
                    turns=np.array(shard_turns, dtype=np.int8),
                    scores=np.array(shard_scores, dtype=np.float32),
                    game_ids=np.array(shard_game_ids, dtype=np.int32),
                )
                print(f"  Saved shard {shard_idx}: {len(shard_boards)} positions "
                      f"from {games_in_shard} games -> {shard_path}", flush=True)
                shard_boards = []
                shard_turns = []
                shard_scores = []
                shard_game_ids = []
                shard_idx += 1
                games_in_shard = 0

            if total_games % 100 == 0:
                elapsed = time.time() - t0
                speed = total_games / elapsed
                eta = (args.games - total_games) / speed
                print(f"[{total_games}/{args.games}] "
                      f"positions={total_positions:,} "
                      f"speed={speed:.1f} games/s "
                      f"ETA={eta:.0f}s "
                      f"W/L/D={outcomes['red_win']}/{outcomes['black_win']}/{outcomes['draw']}",
                      flush=True)

    if shard_boards:
        shard_path = os.path.join(args.output_dir, f"td_shard_{shard_idx:04d}.npz")
        np.savez_compressed(shard_path,
            boards=np.array(shard_boards, dtype=np.int8),
            turns=np.array(shard_turns, dtype=np.int8),
            scores=np.array(shard_scores, dtype=np.float32),
            game_ids=np.array(shard_game_ids, dtype=np.int32),
        )
        print(f"  Saved shard {shard_idx}: {len(shard_boards)} positions "
              f"from {games_in_shard} games -> {shard_path}", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone! {total_games} games, {total_positions:,} positions in {elapsed:.0f}s")
    print(f"Outcomes: Red={outcomes['red_win']} Black={outcomes['black_win']} Draw={outcomes['draw']}")
    print(f"Avg positions/game: {total_positions/max(total_games,1):.0f}")
    print(f"Throughput: {total_games/elapsed:.1f} games/s ({args.workers} workers)")


if __name__ == "__main__":
    main()
