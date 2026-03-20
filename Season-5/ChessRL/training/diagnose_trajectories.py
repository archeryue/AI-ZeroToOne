"""Diagnostic: Play games with different checkpoints and print full move trajectories."""

import os
import sys
import numpy as np
import torch

# Import C++ engine first
import engine_c as cc

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHESS_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "ChineseChess", "backend")
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, CHESS_DIR)

from agents.alphazero.network import AlphaZeroNet
from agents.alphazero.mcts import MCTS
from env.action_space import decode_action, NUM_ACTIONS

# Piece names for display
PIECE_NAMES = {
    1: "General", 2: "Advisor", 3: "Elephant",
    4: "Horse", 5: "Chariot", 6: "Cannon", 7: "Soldier"
}

def piece_str(piece):
    """Human-readable piece string."""
    if piece == 0:
        return "."
    color = "R" if piece > 0 else "B"
    name = PIECE_NAMES.get(abs(piece), "?")
    return f"{color}-{name}"

def material_count(game):
    """Return (red_material, black_material) as dicts."""
    red, black = {}, {}
    for r in range(10):
        for c in range(9):
            p = game.board.get(r, c)
            if p > 0:
                name = PIECE_NAMES.get(p, "?")
                red[name] = red.get(name, 0) + 1
            elif p < 0:
                name = PIECE_NAMES.get(-p, "?")
                black[name] = black.get(name, 0) + 1
    return red, black

def material_value(game):
    """Red perspective material value in [-1, 1]."""
    _PV = {1: 0, 2: 2, 3: 2, 4: 4, 5: 9, 6: 4.5, 7: 1}
    red_mat, black_mat = 0.0, 0.0
    for r in range(10):
        for c in range(9):
            p = game.board.get(r, c)
            if p > 0:
                red_mat += _PV.get(p, 0)
            elif p < 0:
                black_mat += _PV.get(-p, 0)
    return red_mat, black_mat, max(-1.0, min(1.0, (red_mat - black_mat) / 10.0))

def print_board(game):
    """Print the board in a readable format."""
    symbols = {
        0: ". ",
        1: "K ", 2: "A ", 3: "E ", 4: "H ", 5: "R ", 6: "C ", 7: "S ",
        -1: "k ", -2: "a ", -3: "e ", -4: "h ", -5: "r ", -6: "c ", -7: "s ",
    }
    print("   0 1 2 3 4 5 6 7 8")
    for r in range(10):
        row_str = ""
        for c in range(9):
            p = game.board.get(r, c)
            row_str += symbols.get(p, "? ")
        print(f" {r} {row_str}")
    print("(UPPER=Red, lower=Black)")
    print(f"Turn: {'Red' if game.current_turn == cc.RED else 'Black'}")

def play_diagnostic_game(network, device, label, game_idx=0, vs_random=True):
    """Play one game and print every move."""
    mcts = MCTS(network, device, num_simulations=50, c_puct=1.5)
    game = cc.Game()

    print(f"\n{'='*60}")
    print(f"Game {game_idx+1}: {label}" + (" vs Random" if vs_random else " self-play"))
    print(f"{'='*60}")

    step = 0
    pos_hist = {}
    move_log = []

    while game.status == cc.STATUS_PLAYING and step < 200:
        turn_str = "Red" if game.current_turn == cc.RED else "Black"

        if vs_random and game.current_turn == cc.BLACK:
            # Random opponent
            mask = cc.get_action_mask(game.board, game.current_turn)
            legal = np.where(mask)[0]
            action = int(np.random.choice(legal))
            method = "random"
        else:
            # Network + MCTS
            action, info = mcts.select_action(game, temperature=0.1, add_noise=False)
            method = "MCTS"

        fr, fc, tr, tc = decode_action(action)
        moved_piece = game.board.get(fr, fc)
        captured_piece = game.board.get(tr, tc)

        move_desc = f"  {step+1:3d}. {turn_str:5s} ({method:6s}): {piece_str(moved_piece)} ({fr},{fc})->({tr},{tc})"
        if captured_piece != 0:
            move_desc += f"  CAPTURES {piece_str(captured_piece)}"

        game.make_move(fr, fc, tr, tc)
        step += 1

        fen = game.board.to_fen()
        pos_hist[fen] = pos_hist.get(fen, 0) + 1

        # Print move
        red_mat, black_mat, mat_val = material_value(game)
        move_desc += f"  | Mat: R={red_mat:.0f} B={black_mat:.0f} (val={mat_val:+.2f})"

        # Flag repetitions
        if pos_hist[fen] >= 2:
            move_desc += f"  [REPEAT x{pos_hist[fen]}]"

        move_log.append(move_desc)

        # Print every move for first 30, then every 10
        if step <= 30 or step % 10 == 0 or captured_piece != 0 or pos_hist[fen] >= 2:
            print(move_desc)

    # Final state
    print(f"\n--- Game ended after {step} steps ---")
    if game.status == cc.STATUS_RED_WIN:
        result = "Red wins (checkmate)"
    elif game.status == cc.STATUS_BLACK_WIN:
        result = "Black wins (checkmate)"
    else:
        red_mat, black_mat, mat_val = material_value(game)
        if mat_val > 0.05:
            result = f"Red wins by material ({red_mat:.0f} vs {black_mat:.0f})"
        elif mat_val < -0.05:
            result = f"Black wins by material ({red_mat:.0f} vs {black_mat:.0f})"
        else:
            result = "Draw"
    print(f"Result: {result}")

    # Print final board
    print_board(game)

    # Material summary
    red_pieces, black_pieces = material_count(game)
    print(f"\nRed pieces remaining: {red_pieces}")
    print(f"Black pieces remaining: {black_pieces}")

    # Repetition stats
    repeats = sum(1 for v in pos_hist.values() if v >= 2)
    print(f"Positions repeated 2+ times: {repeats}")

    return result


def analyze_policy_entropy(network, device, label):
    """Check how spread out the policy is on the starting position."""
    game = cc.Game()
    obs = cc.board_to_observation(game.board, game.current_turn)
    mask = cc.get_action_mask(game.board, game.current_turn)

    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
    mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)

    with torch.no_grad():
        log_policy, value = network(obs_t)
        # Apply mask
        log_policy = log_policy.squeeze(0).cpu().numpy()
        mask_np = mask.astype(bool)

        # Softmax over legal moves only
        legal_logits = log_policy[mask_np]
        legal_logits -= legal_logits.max()
        probs = np.exp(legal_logits) / np.exp(legal_logits).sum()

        # Entropy
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        max_entropy = np.log(len(probs))

        # Top moves
        legal_actions = np.where(mask_np)[0]
        sorted_idx = np.argsort(-probs)

        print(f"\n{'='*60}")
        print(f"Policy Analysis: {label}")
        print(f"{'='*60}")
        print(f"Value estimate: {value.item():.4f}")
        print(f"Legal moves: {len(legal_actions)}")
        print(f"Entropy: {entropy:.3f} / {max_entropy:.3f} ({entropy/max_entropy*100:.1f}%)")
        print(f"Top 10 moves:")
        for i in range(min(10, len(sorted_idx))):
            idx = sorted_idx[i]
            action = legal_actions[idx]
            fr, fc, tr, tc = decode_action(action)
            piece = game.board.get(fr, fc)
            print(f"  {i+1}. {piece_str(piece)} ({fr},{fc})->({tr},{tc})  prob={probs[idx]:.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    v3_dir = os.path.join(script_dir, "candidate4_v3_nopretrain")

    checkpoints = {
        "v3-nopretrain Best (iter50, 70%)": os.path.join(v3_dir, "az_best.pt"),
        "v3-nopretrain Latest (iter166, 30%)": os.path.join(v3_dir, "az_checkpoint.pt"),
    }

    for label, path in checkpoints.items():
        if not os.path.exists(path):
            print(f"Skipping {label}: {path} not found")
            continue

        print(f"\n\n{'#'*70}")
        print(f"# Loading: {label}")
        print(f"# Path: {path}")
        print(f"{'#'*70}")

        net = AlphaZeroNet(num_blocks=5, channels=64)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        if 'model_state_dict' in ckpt:
            net.load_state_dict(ckpt['model_state_dict'])
        elif 'model_state' in ckpt:
            net.load_state_dict(ckpt['model_state'])
        else:
            net.load_state_dict(ckpt)
        net.to(device)
        net.eval()

        # Policy analysis on starting position
        analyze_policy_entropy(net, device, label)

        # Play 3 games vs random
        np.random.seed(42)
        for g in range(3):
            play_diagnostic_game(net, device, label, game_idx=g, vs_random=True)


if __name__ == "__main__":
    main()
