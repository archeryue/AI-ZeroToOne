"""Score human game positions with NNUE v2 engine for TD(lambda) training.

Parses DhtmlXQ human games, replays moves to get board states,
then scores each position with the v2 engine at a given search depth.
Output format matches self-play data: (boards, turns, scores, game_ids).

Uses multiprocessing: each worker scores one game (creates its own engine).
"""

import os
import sys
import re
import time
import argparse
import numpy as np
from multiprocessing import Pool

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_DIR)

# Piece encoding
RED_PIECE_ORDER = [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7]
BLACK_PIECE_ORDER = [-1, -2, -2, -3, -3, -4, -4, -5, -5, -6, -6, -7, -7, -7, -7, -7]

STANDARD_GRID = [
    [-5, -4, -3, -2, -1, -2, -3, -4, -5],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0, -6,  0,  0,  0,  0,  0, -6,  0],
    [-7,  0, -7,  0, -7,  0, -7,  0, -7],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 7,  0,  7,  0,  7,  0,  7,  0,  7],
    [ 0,  6,  0,  0,  0,  0,  0,  6,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0],
    [ 5,  4,  3,  2,  1,  2,  3,  4,  5],
]

DEFAULT_WEIGHTS = os.path.join(
    PROJECT_DIR, "training", "candidate5_v2", "checkpoints", "nnue_v2_weights.bin"
)


def parse_dhtmlxq_file(filepath):
    """Parse a single DhtmlXQ file. Returns (grid, moves, result) or None."""
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            content = f.read()
    except (UnicodeDecodeError, FileNotFoundError):
        try:
            with open(filepath, 'r', encoding='gbk') as f:
                content = f.read()
        except Exception:
            return None

    movelist_match = re.search(r'\[DhtmlXQ_movelist\](.*?)\[/DhtmlXQ_movelist\]', content)
    result_match = re.search(r'\[DhtmlXQ_result\](.*?)\[/DhtmlXQ_result\]', content)

    if not movelist_match:
        return None

    movelist_str = movelist_match.group(1).strip()
    if len(movelist_str) < 4 or len(movelist_str) % 4 != 0:
        return None

    # Parse result
    result_str = result_match.group(1).strip() if result_match else ""
    if "红胜" in result_str or "红先胜" in result_str:
        result = "red"
    elif "黑胜" in result_str or "黑先胜" in result_str:
        result = "black"
    elif "和" in result_str:
        result = "draw"
    else:
        title_match = re.search(r'\[DhtmlXQ_title\](.*?)\[/DhtmlXQ_title\]', content)
        title = title_match.group(1) if title_match else ""
        if "先胜" in title or "红胜" in title:
            result = "red"
        elif "先负" in title or "黑胜" in title:
            result = "black"
        elif "和" in title:
            result = "draw"
        else:
            result = "unknown"

    # Parse initial board
    binit_match = re.search(r'\[DhtmlXQ_binit\](.*?)\[/DhtmlXQ_binit\]', content)
    grid = None
    if binit_match:
        binit = binit_match.group(1).strip()
        if len(binit) >= 64:
            grid = [[0]*9 for _ in range(10)]
            for i in range(16):
                c, r = binit[i*2], binit[i*2+1]
                if c == '9' and r == '9':
                    continue
                col, row = int(c), int(r)
                if 0 <= row <= 9 and 0 <= col <= 8:
                    grid[row][col] = RED_PIECE_ORDER[i]
            for i in range(16):
                c, r = binit[32+i*2], binit[32+i*2+1]
                if c == '9' and r == '9':
                    continue
                col, row = int(c), int(r)
                if 0 <= row <= 9 and 0 <= col <= 8:
                    grid[row][col] = BLACK_PIECE_ORDER[i]

    if grid is None:
        grid = [row[:] for row in STANDARD_GRID]

    # Parse moves
    moves = []
    for i in range(0, len(movelist_str) - 3, 4):
        try:
            fc, fr, tc, tr = int(movelist_str[i]), int(movelist_str[i+1]), int(movelist_str[i+2]), int(movelist_str[i+3])
            moves.append((fr, fc, tr, tc))
        except (ValueError, IndexError):
            break

    if not moves:
        return None

    return grid, moves, result


def replay_game(grid, moves):
    """Replay moves on a grid, return list of (board_flat, stm) per position."""
    positions = []
    current_grid = [row[:] for row in grid]
    stm = 1  # Red first

    for fr, fc, tr, tc in moves:
        piece = current_grid[fr][fc]
        if piece == 0:
            break
        if (piece > 0 and stm != 1) or (piece < 0 and stm != -1):
            break
        positions.append((
            np.array(current_grid, dtype=np.int8).flatten(),
            stm
        ))
        current_grid[tr][tc] = current_grid[fr][fc]
        current_grid[fr][fc] = 0
        stm = -stm

    return positions


def score_one_game(args_tuple):
    """
    Score all positions of one human game with v2 engine.
    Worker function for multiprocessing — creates engine inside worker.
    """
    game_id, positions_data, weights_path, depth = args_tuple

    # Lazy import inside worker — same pattern as gen_td_data.py
    from engine_c._xiangqi import NNUESearchV2, Board as CBoard

    engine = NNUESearchV2()
    engine.load_weights(weights_path)
    engine.set_nnue_weight(1.0)

    boards_list = []
    turns_list = []
    scores_list = []

    for board_flat, stm in positions_data:
        try:
            grid_2d = board_flat.reshape(10, 9).tolist()
            c_board = CBoard(grid_2d)
            result_dict = engine.search(c_board, stm, depth)
            score = result_dict['score']
            # Convert to red-perspective absolute score
            if stm == 1:
                red_score = score * 2 - 1
            else:
                red_score = -(score * 2 - 1)
            boards_list.append(board_flat)
            turns_list.append(stm)
            scores_list.append(red_score)
        except Exception:
            break

    return {
        'game_id': game_id,
        'boards': boards_list,
        'turns': turns_list,
        'scores': scores_list,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=5000)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument("--shard_size", type=int, default=1000)
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(SCRIPT_DIR, "td_data_human"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all game files
    data_dir = os.path.join(PROJECT_DIR, "data", "community-xiangqi-games-database", "data")
    game_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if not f.endswith('.json') and not f.endswith('.md'):
                game_files.append(os.path.join(root, f))

    print(f"Found {len(game_files)} game files")

    # Shuffle and take extra in case of parse failures
    rng = np.random.RandomState(42)
    rng.shuffle(game_files)
    game_files = game_files[:args.games * 2]

    # Phase 1: Parse all games (fast, single-threaded)
    print("Phase 1: Parsing human games...")
    t0 = time.time()
    parsed_games = []
    for filepath in game_files:
        parsed = parse_dhtmlxq_file(filepath)
        if parsed is None:
            continue
        grid, moves, result = parsed
        if result == "unknown":
            continue
        positions = replay_game(grid, moves)
        if len(positions) >= 5:
            parsed_games.append(positions)
        if len(parsed_games) >= args.games:
            break

    total_positions = sum(len(g) for g in parsed_games)
    print(f"  Parsed {len(parsed_games)} games, {total_positions:,} positions in {time.time()-t0:.1f}s")

    print(f"\nPhase 2: Scoring with NNUE v2 (depth {args.depth})")
    print(f"  Weights: {args.weights}")
    print(f"  Workers: {args.workers}")
    print(f"  Estimated time: ~{total_positions * 0.008 / args.workers / 60:.0f} minutes\n")

    # Phase 2: Score with engine using multiprocessing
    # Each task = one game's positions, worker creates its own engine
    tasks = [
        (i, positions, args.weights, args.depth)
        for i, positions in enumerate(parsed_games)
    ]

    games_done = 0
    positions_total = 0
    t0 = time.time()
    shard_idx = 0
    shard_boards, shard_turns, shard_scores, shard_game_ids = [], [], [], []
    shard_games = 0
    red_wins = black_wins = draws = 0

    with Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(score_one_game, tasks, chunksize=4):
            if len(result['boards']) < 5:
                continue

            boards = np.array(result['boards'], dtype=np.int8)
            turns_arr = np.array(result['turns'], dtype=np.int8)
            scores_arr = np.array(result['scores'], dtype=np.float32)

            shard_boards.append(boards)
            shard_turns.append(turns_arr)
            shard_scores.append(scores_arr)
            shard_game_ids.append(np.full(len(boards), games_done, dtype=np.int32))
            shard_games += 1

            final = scores_arr[-1]
            if final > 0.3:
                red_wins += 1
            elif final < -0.3:
                black_wins += 1
            else:
                draws += 1

            positions_total += len(boards)
            games_done += 1

            # Save shard if ready
            if shard_games >= args.shard_size:
                out_path = os.path.join(args.output_dir, f"td_shard_{shard_idx:04d}.npz")
                np.savez_compressed(
                    out_path,
                    boards=np.concatenate(shard_boards),
                    turns=np.concatenate(shard_turns),
                    scores=np.concatenate(shard_scores),
                    game_ids=np.concatenate(shard_game_ids),
                )
                n_pos = sum(len(b) for b in shard_boards)
                print(f"  Saved shard {shard_idx}: {n_pos} positions from {shard_games} games -> {out_path}")
                shard_boards, shard_turns, shard_scores, shard_game_ids = [], [], [], []
                shard_games = 0
                shard_idx += 1

            if games_done % 100 == 0:
                elapsed = time.time() - t0
                speed = games_done / elapsed if elapsed > 0 else 0
                eta = (len(parsed_games) - games_done) / speed if speed > 0 else 0
                print(f"[{games_done}/{len(parsed_games)}] positions={positions_total:,} "
                      f"speed={speed:.1f} games/s ETA={eta:.0f}s "
                      f"W/L/D={red_wins}/{black_wins}/{draws}",
                      flush=True)

    # Save remaining
    if shard_boards:
        out_path = os.path.join(args.output_dir, f"td_shard_{shard_idx:04d}.npz")
        np.savez_compressed(
            out_path,
            boards=np.concatenate(shard_boards),
            turns=np.concatenate(shard_turns),
            scores=np.concatenate(shard_scores),
            game_ids=np.concatenate(shard_game_ids),
        )
        n_pos = sum(len(b) for b in shard_boards)
        print(f"  Saved shard {shard_idx}: {n_pos} positions from {shard_games} games -> {out_path}")

    elapsed = time.time() - t0
    print(f"\nDone! {games_done} games, {positions_total:,} positions in {elapsed:.0f}s")
    print(f"Outcomes: Red={red_wins} Black={black_wins} Draw={draws}")
    print(f"Avg positions/game: {positions_total // max(games_done, 1)}")
    print(f"Throughput: {games_done / elapsed:.1f} games/s ({args.workers} workers)")
    print(f"Data saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
