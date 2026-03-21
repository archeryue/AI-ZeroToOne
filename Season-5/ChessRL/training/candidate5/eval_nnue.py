"""
Evaluate NNUE model by playing games against Random/Greedy/Minimax opponents.

Uses greedy 1-ply search: for each legal move, evaluate resulting position,
pick the move with the best eval from the side-to-move's perspective.

Usage:
    python eval_nnue.py [--games 100] [--depth 1] [--opponent random]
"""

import os
import sys
import argparse
import time
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, SCRIPT_DIR)

# Import C++ engine BEFORE anything else
import engine_c
from engine_c._xiangqi import (
    Board, Game, get_legal_moves, is_in_check, is_checkmate, is_stalemate,
    RED, BLACK, STATUS_PLAYING, STATUS_RED_WIN, STATUS_BLACK_WIN, STATUS_DRAW,
    GENERAL, ADVISOR, ELEPHANT, HORSE, CHARIOT, CANNON, SOLDIER,
)

from nnue_net import NNUENet, FEATURES_PER_PERSPECTIVE, NUM_PIECE_TYPES, NUM_SQUARES


def board_to_features_single(board, turn):
    """Convert a Board object + turn to NNUE feature vectors."""
    stm = np.zeros(FEATURES_PER_PERSPECTIVE, dtype=np.float32)
    nstm = np.zeros(FEATURES_PER_PERSPECTIVE, dtype=np.float32)

    for sq in range(90):
        r, c = sq // 9, sq % 9
        piece = board.get(r, c)
        if piece == 0:
            continue

        piece_color = 1 if piece > 0 else -1
        piece_type = abs(piece)
        type_idx = piece_type - 1

        # STM perspective
        color_idx = 0 if piece_color == turn else 1
        feat_idx = color_idx * (NUM_PIECE_TYPES * NUM_SQUARES) + type_idx * NUM_SQUARES + sq
        stm[feat_idx] = 1.0

        # NSTM perspective (mirrored)
        mirror_sq = 89 - sq
        nstm_color_idx = 1 if piece_color == turn else 0
        feat_idx = nstm_color_idx * (NUM_PIECE_TYPES * NUM_SQUARES) + type_idx * NUM_SQUARES + mirror_sq
        nstm[feat_idx] = 1.0

    return stm, nstm


def evaluate_position(model, board, turn, device):
    """Evaluate a single position. Returns eval from side-to-move's perspective."""
    stm, nstm = board_to_features_single(board, turn)
    stm_t = torch.from_numpy(stm).unsqueeze(0).to(device)
    nstm_t = torch.from_numpy(nstm).unsqueeze(0).to(device)
    with torch.no_grad():
        val = model(stm_t, nstm_t).item()
    return val


def evaluate_positions_batch(model, boards_and_turns, device):
    """Batch evaluate multiple positions."""
    N = len(boards_and_turns)
    stm_all = np.zeros((N, FEATURES_PER_PERSPECTIVE), dtype=np.float32)
    nstm_all = np.zeros((N, FEATURES_PER_PERSPECTIVE), dtype=np.float32)

    for i, (board, turn) in enumerate(boards_and_turns):
        stm_all[i], nstm_all[i] = board_to_features_single(board, turn)

    stm_t = torch.from_numpy(stm_all).to(device)
    nstm_t = torch.from_numpy(nstm_all).to(device)
    with torch.no_grad():
        vals = model(stm_t, nstm_t).squeeze(1).cpu().numpy()
    return vals


# ─── Material eval for Greedy/Minimax opponents ─────────────────────────────

PIECE_VALUES = {
    GENERAL: 0, ADVISOR: 2, ELEPHANT: 2, HORSE: 4,
    CHARIOT: 9, CANNON: 4.5, SOLDIER: 1
}


def material_eval(board, color):
    """Simple material evaluation from `color`'s perspective."""
    score = 0.0
    for r in range(10):
        for c in range(9):
            piece = board.get(r, c)
            if piece == 0:
                continue
            piece_type = abs(piece)
            val = PIECE_VALUES.get(piece_type, 0)
            if (piece > 0 and color == RED) or (piece < 0 and color == BLACK):
                score += val
            else:
                score -= val
    return score


# ─── Opponents ───────────────────────────────────────────────────────────────

def random_move(board, color):
    """Pick a random legal move."""
    moves = get_legal_moves(board, color)
    if not moves:
        return None
    idx = np.random.randint(len(moves))
    m = moves[idx]
    return (m.from_sq // 9, m.from_sq % 9, m.to_sq // 9, m.to_sq % 9)


def greedy_move(board, color):
    """Pick move that maximizes material advantage."""
    moves = get_legal_moves(board, color)
    if not moves:
        return None

    best_val = -1e9
    best_move = None
    for m in moves:
        fr, fc = m.from_sq // 9, m.from_sq % 9
        tr, tc = m.to_sq // 9, m.to_sq % 9
        b2 = board.copy()
        piece = b2.get(fr, fc)
        captured = b2.get(tr, tc)
        b2.set(fr, fc, 0)
        b2.set(tr, tc, piece)
        val = material_eval(b2, color)
        if val > best_val:
            best_val = val
            best_move = (fr, fc, tr, tc)

    return best_move


def minimax_move(board, color, depth=3):
    """Minimax with alpha-beta pruning using material eval."""

    def _search(board, color, depth, alpha, beta, maximizing):
        if depth == 0:
            return material_eval(board, color), None

        moves = get_legal_moves(board, color if maximizing else (-color))
        if not moves:
            # No legal moves = loss
            return (-1000 if maximizing else 1000), None

        best_move = None
        if maximizing:
            best_val = -1e9
            for m in moves:
                fr, fc = m.from_sq // 9, m.from_sq % 9
                tr, tc = m.to_sq // 9, m.to_sq % 9
                b2 = board.copy()
                piece = b2.get(fr, fc)
                b2.set(fr, fc, 0)
                b2.set(tr, tc, piece)
                val, _ = _search(b2, color, depth - 1, alpha, beta, False)
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
                piece = b2.get(fr, fc)
                b2.set(fr, fc, 0)
                b2.set(tr, tc, piece)
                val, _ = _search(b2, color, depth - 1, alpha, beta, True)
                if val < best_val:
                    best_val = val
                    best_move = (fr, fc, tr, tc)
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return best_val, best_move

    _, move = _search(board, color, depth, -1e9, 1e9, True)
    return move


# ─── NNUE Player ─────────────────────────────────────────────────────────────

def nnue_move(model, board, color, device, depth=2):
    """NNUE-based move with alpha-beta search."""

    def _apply_move(board, m):
        fr, fc = m.from_sq // 9, m.from_sq % 9
        tr, tc = m.to_sq // 9, m.to_sq % 9
        b2 = board.copy()
        piece = b2.get(fr, fc)
        b2.set(fr, fc, 0)
        b2.set(tr, tc, piece)
        return b2, (fr, fc, tr, tc)

    def _nnue_eval(board, color):
        """Evaluate from `color`'s perspective using NNUE."""
        return evaluate_position(model, board, color, device)

    def _search(board, stm, depth, alpha, beta):
        """Negamax with alpha-beta. Returns (value, move) from stm's perspective.
        Value in [0, 1]: 1.0 = stm wins, 0.0 = stm loses."""
        moves = get_legal_moves(board, stm)
        if not moves:
            return 0.0, None  # Loss (no legal moves)

        if depth == 0:
            return _nnue_eval(board, stm), None

        best_val = -1.0
        best_move = None
        for m in moves:
            b2, move_tuple = _apply_move(board, m)
            # Recurse from opponent's perspective, flip: their win = our loss
            val, _ = _search(b2, -stm, depth - 1, 1.0 - beta, 1.0 - alpha)
            val = 1.0 - val
            if val > best_val:
                best_val = val
                best_move = move_tuple
            alpha = max(alpha, val)
            if alpha >= beta:
                break
        return best_val, best_move

    _, move = _search(board, color, depth, 0.0, 1.0)
    return move


# ─── Game Loop ───────────────────────────────────────────────────────────────

def play_game(nnue_model, device, opponent_fn, nnue_color=RED, max_moves=200):
    """Play one game. Returns 'nnue_win', 'opponent_win', or 'draw'."""
    game = Game()
    current = RED

    for step in range(max_moves):
        if game.status != STATUS_PLAYING:
            break

        moves = get_legal_moves(game.board, current)
        if not moves:
            # No legal moves = loss for current player
            if current == nnue_color:
                return 'opponent_win'
            else:
                return 'nnue_win'

        if current == nnue_color:
            move = nnue_move(nnue_model, game.board, current, device)
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
            # Invalid move = loss
            if current == nnue_color:
                return 'opponent_win'
            else:
                return 'nnue_win'

        current = -current

    # Check final status
    if game.status == STATUS_RED_WIN:
        return 'nnue_win' if nnue_color == RED else 'opponent_win'
    elif game.status == STATUS_BLACK_WIN:
        return 'nnue_win' if nnue_color == BLACK else 'opponent_win'
    else:
        return 'draw'


def evaluate(model, device, opponent_name, opponent_fn, num_games=100):
    """Run evaluation games and report results."""
    results = {'nnue_win': 0, 'opponent_win': 0, 'draw': 0}

    t0 = time.time()
    for i in range(num_games):
        # Alternate colors
        nnue_color = RED if i % 2 == 0 else BLACK
        result = play_game(model, device, opponent_fn, nnue_color=nnue_color)
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

    print(f"\n=== NNUE vs {opponent_name} ({num_games} games) ===")
    print(f"  Wins:   {results['nnue_win']:3d} ({win_rate:.1f}%)")
    print(f"  Losses: {results['opponent_win']:3d} ({loss_rate:.1f}%)")
    print(f"  Draws:  {results['draw']:3d} ({draw_rate:.1f}%)")
    print(f"  Time:   {total:.0f}s ({total/num_games:.1f}s/game)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--games', type=int, default=100)
    parser.add_argument('--opponent', type=str, default='all',
                        choices=['random', 'greedy', 'minimax', 'all'])
    parser.add_argument('--checkpoint', type=str,
                        default=os.path.join(SCRIPT_DIR, 'checkpoints', 'nnue_best.pt'))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = NNUENet().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}, "
          f"sign_acc={ckpt['sign_accuracy']:.3f}")

    # Quick sanity check: eval starting position
    game = Game()
    val = evaluate_position(model, game.board, RED, device)
    print(f"Starting position eval (Red to move): {val:.4f}")

    opponents = {
        'random': ('Random', random_move),
        'greedy': ('Greedy (material)', greedy_move),
    }

    if args.opponent == 'all':
        for name, (label, fn) in opponents.items():
            evaluate(model, device, label, fn, args.games)
        # Minimax is slow, fewer games
        print("\n--- Minimax (depth 2, fewer games due to speed) ---")
        evaluate(model, device, 'Minimax-d2',
                 lambda b, c: minimax_move(b, c, depth=2),
                 min(args.games, 20))
    elif args.opponent == 'minimax':
        evaluate(model, device, 'Minimax-d2',
                 lambda b, c: minimax_move(b, c, depth=2),
                 args.games)
    else:
        label, fn = opponents[args.opponent]
        evaluate(model, device, label, fn, args.games)


if __name__ == "__main__":
    main()
