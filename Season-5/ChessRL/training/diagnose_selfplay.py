"""Diagnose: run actual self-play code and inspect the training targets it produces."""

import os
import sys
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# This imports the actual self-play function and all helpers
from train_alphazero import (
    run_self_play_batch, AlphaZeroNet, NUM_BLOCKS, CHANNELS,
    NUM_SIMULATIONS, VIRTUAL_LOSS_N, NUM_ACTIONS, _new_game, _get_obs, _get_mask,
    batch_evaluate, MCTSNode, _get_legal_actions, _simulate, _is_playing,
    _terminal_value, C_PUCT, DIRICHLET_ALPHA, DIRICHLET_EPSILON, TEMP_THRESHOLD,
    MAX_GAME_STEPS, decode_action
)
import engine_c as cc

PIECE_NAMES = {1: "General", 2: "Advisor", 3: "Elephant",
               4: "Horse", 5: "Chariot", 6: "Cannon", 7: "Soldier"}

def piece_str(piece):
    if piece == 0: return "."
    color = "R" if piece > 0 else "B"
    return f"{color}-{PIECE_NAMES.get(abs(piece), '?')}"


def analyze_training_targets(examples, label=""):
    """Analyze the action_probs stored as training targets."""
    print(f"\n{'='*70}")
    print(f"Training Target Analysis: {label}")
    print(f"{'='*70}")
    print(f"Total positions: {len(examples)}")

    entropies = []
    top_probs = []
    nonzero_counts = []

    for i, (obs, action_probs, player) in enumerate(examples):
        # Only analyze legal moves (non-zero probabilities)
        nonzero = action_probs[action_probs > 0]
        n_nonzero = len(nonzero)
        nonzero_counts.append(n_nonzero)

        if n_nonzero > 0:
            entropy = -np.sum(nonzero * np.log(nonzero + 1e-8))
            max_entropy = np.log(n_nonzero) if n_nonzero > 1 else 1.0
            entropies.append(entropy / max_entropy if max_entropy > 0 else 0)
            top_probs.append(np.max(action_probs))

        # Print first 5 positions in detail
        if i < 5:
            sorted_idx = np.argsort(-action_probs)
            print(f"\n  Position {i+1} (player={'Red' if player == cc.RED else 'Black'}):")
            print(f"    Non-zero actions: {n_nonzero}")
            if n_nonzero > 0:
                print(f"    Entropy: {entropies[-1]*100:.1f}% of max")
                print(f"    Top 5 actions:")
                for j in range(min(5, n_nonzero)):
                    a = sorted_idx[j]
                    p = action_probs[a]
                    if p > 0:
                        fr, fc, tr, tc = decode_action(a)
                        print(f"      {piece_str(0)} action={a} ({fr},{fc})->({tr},{tc})  prob={p:.4f}")

    entropies = np.array(entropies)
    top_probs = np.array(top_probs)
    nonzero_counts = np.array(nonzero_counts)

    print(f"\n--- Summary Statistics ---")
    print(f"Avg non-zero actions per position: {nonzero_counts.mean():.1f}")
    print(f"Entropy (% of max): mean={entropies.mean()*100:.1f}%, "
          f"std={entropies.std()*100:.1f}%, "
          f"min={entropies.min()*100:.1f}%, max={entropies.max()*100:.1f}%")
    print(f"Top action prob: mean={top_probs.mean():.3f}, "
          f"std={top_probs.std():.3f}, "
          f"min={top_probs.min():.3f}, max={top_probs.max():.3f}")

    # Histogram of top-action probability
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    hist, _ = np.histogram(top_probs, bins=bins)
    print(f"\nTop-action probability distribution:")
    for i in range(len(hist)):
        bar = '#' * (hist[i] * 50 // max(len(examples), 1))
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]:4d} ({hist[i]/len(examples)*100:5.1f}%) {bar}")


def run_one_self_play_and_inspect(network, device, label, num_sims=200):
    """Run one self-play iteration (16 games) and inspect the targets."""
    print(f"\n\n{'#'*70}")
    print(f"# Self-Play Inspection: {label}")
    print(f"# num_sims={num_sims}, virtual_loss_n={VIRTUAL_LOSS_N}")
    print(f"{'#'*70}")

    np.random.seed(42)
    completed, total_evals, total_batches = run_self_play_batch(
        network, device, n_parallel=4, num_sims=num_sims)

    print(f"\nCompleted {len(completed)} games")
    print(f"Total NN evals: {total_evals}, batches: {total_batches}")

    all_examples = []
    for i, (examples, result, steps, draw_reason, final_game) in enumerate(completed):
        print(f"\n  Game {i+1}: result={result}, steps={steps}, "
              f"draw_reason={draw_reason}, examples={len(examples)}")
        all_examples.extend(examples)

    analyze_training_targets(all_examples, label)

    # Also analyze just the early-game vs late-game positions
    n = len(all_examples)
    if n > 20:
        early = all_examples[:n//4]
        late = all_examples[3*n//4:]
        analyze_training_targets(early, f"{label} — Early game (first 25%)")
        analyze_training_targets(late, f"{label} — Late game (last 25%)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(SCRIPT_DIR, "candidate4")

    checkpoints = {
        "Pretrained": os.path.join(ckpt_dir, "az_pretrained.pt"),
        "Best (iter 25)": os.path.join(ckpt_dir, "az_best.pt"),
    }

    for label, path in checkpoints.items():
        if not os.path.exists(path):
            continue

        net = AlphaZeroNet(num_blocks=NUM_BLOCKS, channels=CHANNELS)
        ckpt = torch.load(path, map_location=device, weights_only=True)
        if 'model_state_dict' in ckpt:
            net.load_state_dict(ckpt['model_state_dict'])
        elif 'model_state' in ckpt:
            net.load_state_dict(ckpt['model_state'])
        else:
            net.load_state_dict(ckpt)
        net.to(device)
        net.eval()

        run_one_self_play_and_inspect(net, device, label, num_sims=NUM_SIMULATIONS)


if __name__ == "__main__":
    main()
