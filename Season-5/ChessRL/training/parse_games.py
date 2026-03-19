"""Parse DhtmlXQ game files into (observation, policy_target, value_target) training data.

DhtmlXQ format:
  [DhtmlXQ_binit] = 64-char initial board encoding (or standard if absent)
  [DhtmlXQ_movelist] = sequence of 4-digit moves (from_col from_row to_col to_row)
  [DhtmlXQ_result] = 红胜 (red wins), 黑胜 (black wins), 和棋 (draw)

Board encoding in DhtmlXQ:
  64 chars = 32 pieces × 2 chars each (col, row)
  First 16 pairs = Red pieces, next 16 = Black pieces
  Piece order: King, Advisor×2, Bishop×2, Knight×2, Rook×2, Cannon×2, Pawn×5
  Value "99" means piece is captured/absent.
  Coordinates: col 0-8 (left to right), row 0-9 (top to bottom, Black side = 0)
"""

import os
import sys
import re
import glob
import pickle
import numpy as np
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# Import C++ engine before adding paths
try:
    import engine_c as cc
    _USE_CPP = True
except ImportError:
    _USE_CPP = False

sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(PROJECT_DIR), "ChineseChess", "backend"))

from env.action_space import encode_move, NUM_ACTIONS

# DhtmlXQ standard initial board piece order:
# Red:  King, Adv, Adv, Ele, Ele, Horse, Horse, Chariot, Chariot, Cannon, Cannon, P, P, P, P, P
# Black: King, Adv, Adv, Ele, Ele, Horse, Horse, Chariot, Chariot, Cannon, Cannon, P, P, P, P, P
# Our piece encoding: General=1, Advisor=2, Elephant=3, Horse=4, Chariot=5, Cannon=6, Soldier=7
# Red = positive, Black = negative
RED_PIECE_ORDER = [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 7, 7, 7]
BLACK_PIECE_ORDER = [-1, -2, -2, -3, -3, -4, -4, -5, -5, -6, -6, -7, -7, -7, -7, -7]


def parse_dhtmlxq_file(filepath):
    """Parse a single DhtmlXQ file. Returns (board_grid, movelist, result) or None."""
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            content = f.read()
    except (UnicodeDecodeError, FileNotFoundError):
        try:
            with open(filepath, 'r', encoding='gbk') as f:
                content = f.read()
        except:
            return None

    # Extract fields
    binit_match = re.search(r'\[DhtmlXQ_binit\](.*?)\[/DhtmlXQ_binit\]', content)
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
        # Try to infer from title
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
    grid = [[0]*9 for _ in range(10)]

    if binit_match:
        binit = binit_match.group(1).strip()
        if len(binit) >= 64:
            # Red pieces (first 16 pairs)
            for i in range(16):
                col_char = binit[i*2]
                row_char = binit[i*2 + 1]
                if col_char == '9' and row_char == '9':
                    continue  # captured
                col = int(col_char)
                row = int(row_char)
                if 0 <= row <= 9 and 0 <= col <= 8:
                    grid[row][col] = RED_PIECE_ORDER[i]

            # Black pieces (next 16 pairs)
            for i in range(16):
                col_char = binit[32 + i*2]
                row_char = binit[32 + i*2 + 1]
                if col_char == '9' and row_char == '9':
                    continue
                col = int(col_char)
                row = int(row_char)
                if 0 <= row <= 9 and 0 <= col <= 8:
                    grid[row][col] = BLACK_PIECE_ORDER[i]
    else:
        # Standard initial position
        grid = [
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

    # Parse moves: each move is 4 digits: from_col, from_row, to_col, to_row
    # DhtmlXQ coordinates: col 0-8 left-to-right, row 0-9 top-to-bottom
    moves = []
    for i in range(0, len(movelist_str) - 3, 4):
        try:
            fc = int(movelist_str[i])
            fr = int(movelist_str[i+1])
            tc = int(movelist_str[i+2])
            tr = int(movelist_str[i+3])
            moves.append((fr, fc, tr, tc))
        except (ValueError, IndexError):
            break

    if not moves:
        return None

    return grid, moves, result


def process_games(data_dir, output_path, max_games=None, shard_size=500_000):
    """Process all DhtmlXQ game files into training data, saving in shards.

    Stores only (action, value) per position — observations are reconstructed
    during training by replaying moves. This reduces memory from ~150GB to ~100MB.
    We save: board grids (int8), actions (int32), values (float32), turn (int8).
    """
    game_files = []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if not f.endswith('.json') and not f.endswith('.md'):
                game_files.append(os.path.join(root, f))

    print(f"Found {len(game_files)} game files")
    if max_games:
        game_files = game_files[:max_games]

    # Accumulate compact data: board state as flat int8[90], action, value, turn
    boards_buf = []    # int8[90]
    actions_buf = []   # int32
    values_buf = []    # float32
    turns_buf = []     # int8

    stats = defaultdict(int)
    shard_idx = 0
    total_positions = 0

    def flush_shard():
        nonlocal shard_idx, total_positions
        if not boards_buf:
            return
        shard_path = f"{output_path}_shard{shard_idx:03d}"
        np.savez_compressed(shard_path,
                            boards=np.array(boards_buf, dtype=np.int8),
                            actions=np.array(actions_buf, dtype=np.int32),
                            values=np.array(values_buf, dtype=np.float32),
                            turns=np.array(turns_buf, dtype=np.int8))
        n = len(boards_buf)
        total_positions += n
        size_mb = os.path.getsize(shard_path + '.npz') / 1e6
        print(f"  Saved shard {shard_idx}: {n} positions, {size_mb:.1f} MB")
        boards_buf.clear()
        actions_buf.clear()
        values_buf.clear()
        turns_buf.clear()
        shard_idx += 1

    for idx, filepath in enumerate(game_files):
        if idx % 10000 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(game_files)} files, "
                  f"{total_positions + len(boards_buf)} positions...")

        parsed = parse_dhtmlxq_file(filepath)
        if parsed is None:
            stats['parse_fail'] += 1
            continue

        grid, moves, result = parsed
        stats[f'result_{result}'] += 1

        if result == "unknown":
            stats['skip_unknown'] += 1
            continue

        # Replay moves on a board
        try:
            # Work with flat grid as list for speed
            flat = [0] * 90
            for r in range(10):
                for c in range(9):
                    flat[r * 9 + c] = grid[r][c]

            current_turn = 1  # Red starts

            for fr, fc, tr, tc in moves:
                # Store compact board state
                boards_buf.append(list(flat))

                action = encode_move(fr, fc, tr, tc)
                actions_buf.append(action)

                if result == "red":
                    value = 1.0 if current_turn == 1 else -1.0
                elif result == "black":
                    value = 1.0 if current_turn == -1 else -1.0
                else:
                    value = 0.0
                values_buf.append(value)
                turns_buf.append(current_turn)

                # Apply move
                flat[tr * 9 + tc] = flat[fr * 9 + fc]
                flat[fr * 9 + fc] = 0
                current_turn = -current_turn

            stats['games_ok'] += 1

        except Exception as e:
            stats['game_error'] += 1
            if stats['game_error'] <= 5:
                print(f"  Error in {filepath}: {e}")

        # Flush shard if buffer is large enough
        if len(boards_buf) >= shard_size:
            flush_shard()

    # Flush remaining
    flush_shard()

    print(f"\nDone! {total_positions} training positions from "
          f"{stats['games_ok']} games in {shard_idx} shards")
    print(f"Stats: {dict(stats)}")

    # Write manifest
    manifest_path = output_path + "_manifest.txt"
    with open(manifest_path, 'w') as f:
        f.write(f"total_positions={total_positions}\n")
        f.write(f"num_shards={shard_idx}\n")
        f.write(f"games_ok={stats['games_ok']}\n")
        for k, v in sorted(stats.items()):
            f.write(f"{k}={v}\n")
    print(f"Manifest: {manifest_path}")

    return total_positions


if __name__ == "__main__":
    data_dir = os.path.join(PROJECT_DIR, "data", "community-xiangqi-games-database", "data")
    output_path = os.path.join(PROJECT_DIR, "data", "supervised_training_data")

    os.makedirs(os.path.join(PROJECT_DIR, "data"), exist_ok=True)
    process_games(data_dir, output_path)
