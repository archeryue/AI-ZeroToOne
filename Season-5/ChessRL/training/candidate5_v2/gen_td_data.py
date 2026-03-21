"""
Generate TD training data via self-play using the v1 NNUE C++ engine.

For each game:
  - v1 NNUE engine plays both sides at configurable depth
  - At each position, we record (board, turn, search_score)
  - search_score is the minimax-backed eval from the engine (much more
    accurate than game outcome because it integrates subtree information)

Output: .npz shards with boards (int8), turns (int8), scores (float32)

Uses multiprocessing to run games in parallel across CPU cores.

Usage:
    python gen_td_data.py [--games 10000] [--depth 4] [--workers 12]
"""

import os
import sys
import argparse
import time
import numpy as np
from multiprocessing import Pool, Queue

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
    Play a self-play game (worker function for multiprocessing).

    Each worker creates its own engine instance (C++ objects can't be pickled).
    Epsilon-greedy noise is injected to diversify games:
      - Moves 1-10: epsilon=0.3 (30% chance of random move)
      - Moves 11+:  epsilon=0.05 (5% chance of random move)

    The search score is always recorded from the engine's best eval,
    regardless of whether the actual move was random or best.

    Args:
        args_tuple: (game_id, weights_path, depth, max_moves)

    Returns:
        dict with boards, turns, scores, outcome, game_id
    """
    game_id, weights_path, depth, max_moves = args_tuple

    # Each worker loads its own engine (can't share across processes)
    from engine_c._xiangqi import (
        Game, NNUESearch, get_legal_moves,
        RED, BLACK, STATUS_PLAYING, STATUS_RED_WIN, STATUS_BLACK_WIN,
    )

    rng = np.random.RandomState(game_id)  # deterministic per game for reproducibility

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

        # First 6 moves (3 per side): fully random, no search (fast + diverse openings)
        # Moves 7-16: search but epsilon=0.15 for some exploration
        # Moves 17+: search with epsilon=0.05 for quality play
        if step < 6:
            m = moves[rng.randint(len(moves))]
            fr, fc = m.from_sq // 9, m.from_sq % 9
            tr, tc = m.to_sq // 9, m.to_sq % 9
            # Still record position but use a neutral score (no search)
            boards.append(board_to_flat(game.board))
            turns.append(current)
            scores.append(0.0)  # placeholder, will be smoothed by TD
        else:
            boards.append(board_to_flat(game.board))
            turns.append(current)

            result = engine.search(game.board, current, depth)
            scores.append(result['score'])

            epsilon = 0.15 if step < 16 else 0.05
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
    parser = argparse.ArgumentParser(description="Generate TD training data via self-play")
    parser.add_argument('--games', type=int, default=10000, help='Number of self-play games')
    parser.add_argument('--depth', type=int, default=4, help='Search depth')
    parser.add_argument('--workers', type=int, default=12,
                        help='Number of parallel workers (default: 12)')
    parser.add_argument('--weights', type=str,
                        default=os.path.join(PROJECT_DIR, 'training/candidate5/checkpoints/nnue_weights.bin'),
                        help='Path to v1 NNUE weights')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(SCRIPT_DIR, 'td_data'),
                        help='Output directory for data shards')
    parser.add_argument('--shard_size', type=int, default=1000,
                        help='Games per shard')
    parser.add_argument('--max_moves', type=int, default=200,
                        help='Max moves per game')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Weights: {args.weights}")
    print(f"Search depth: {args.depth}")
    print(f"Games to play: {args.games}")
    print(f"Workers: {args.workers}")
    print(f"Shard size: {args.shard_size} games")
    print(flush=True)

    # Build task list
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
        # imap_unordered for best throughput — results come back as they finish
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

            # Save shard
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

            # Progress
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

    # Save final shard
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
    print(f"Data saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
