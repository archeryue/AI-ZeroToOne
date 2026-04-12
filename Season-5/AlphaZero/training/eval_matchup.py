"""Head-to-head evaluator: two AlphaZero checkpoints play each other.

Use this to measure *actual* self-improvement between two training
snapshots — beating random is meaningless once the model is strong
(~96% against random at iter 9 already).

Fair-eval design (v4 — PAIRED matchup):

  At these weights and 100 sims/move, BLACK wins nearly 100% of games
  regardless of which model plays black — first-move advantage beats
  komi for both networks. That means a single-game evaluation just
  measures "who plays black this round."

  So we play games in *pairs* that share the same random opening:

      Pair P, opening O:
        Game A — NEW plays BLACK, OLD plays WHITE from position after O.
        Game B — NEW plays WHITE, OLD plays BLACK from position after O.

  Pair score for NEW ∈ {0, 0.5, 1.0}:
    - 1.0 : NEW won both A and B (clearly stronger)
    - 0.5 : NEW won exactly one (tied, likely the color advantage one)
    - 0.0 : NEW lost both (clearly weaker)

  Aggregate score = mean over pairs ∈ [0, 1]. 0.5 = matched, >0.5 = stronger.

Variance source:
  - First K ply of each pair's opening are random legal moves.
  - Both games in a pair see the IDENTICAL opening (we save the
    sequence and replay it).
  - After move K, MCTS plays deterministically (argmax, no Dirichlet).

Usage:
    python -m training.eval_matchup \
        --board-size 9 \
        --new checkpoints/9x9_run1/preserved_iter0012.pt \
        --old checkpoints/9x9_run1/eval_opponent_iter0009.pt \
        --pairs 50

    (100 games total = 50 pairs)
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import go_engine  # noqa: E402
from model.config import CONFIGS  # noqa: E402
from model.network import AlphaZeroNet  # noqa: E402
from training.self_play import GAME_CLASS, MCTS_CLASS, make_evaluator  # noqa: E402


def load_net(path: str, model_cfg, device: torch.device) -> torch.nn.Module:
    state = torch.load(path, map_location=device, weights_only=False)
    net = AlphaZeroNet(model_cfg).to(device)
    net.load_state_dict(state["model_state_dict"])
    net.eval()
    return net


def binomial_ci_95(k: float, n: int) -> tuple[float, float]:
    """Wilson 95% CI for a binomial proportion. k may be fractional (pair scores)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    z = 1.96
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def generate_opening(rng, game_cls, train_cfg, N: int, k_random: int):
    """Generate a random-opening move list and the resulting game state."""
    g = game_cls(train_cfg.komi)
    moves = []
    for _ in range(k_random):
        legal = g.get_legal_moves()
        if not legal:
            break
        r, c = legal[rng.integers(0, len(legal))]
        g.make_move(r, c)
        moves.append((r, c))
    return moves


def play_from_opening(
    new_net,
    old_net,
    device,
    model_cfg,
    train_cfg,
    opening_moves: list,
    new_color: int,
    sims_per_move: int,
    seed_base: int,
) -> int:
    """Play one full game from a given opening sequence. Returns:
    +1 if NEW wins, 0 if OLD wins, -1 for draw.
    """
    N = model_cfg.board_size
    GameClass = GAME_CLASS[N]
    MCTSClass = MCTS_CLASS[N]

    new_eval = make_evaluator(new_net, device)
    old_eval = make_evaluator(old_net, device)

    game = GameClass(train_cfg.komi)
    new_tree = MCTSClass(
        game, train_cfg.c_puct,
        train_cfg.dirichlet_alpha, 0.0)
    old_tree = MCTSClass(
        game, train_cfg.c_puct,
        train_cfg.dirichlet_alpha, 0.0)

    # Apply the shared opening to both trees and the game
    for r, c in opening_moves:
        action = r * N + c
        game.make_move(r, c)
        new_tree.advance(action)
        old_tree.advance(action)

    # Play the rest with full-strength deterministic MCTS
    start_move = len(opening_moves)
    for move_num in range(start_move, train_cfg.max_game_moves):
        if game.status != go_engine.PLAYING:
            break

        is_new_turn = (game.current_turn == new_color)
        tree = new_tree if is_new_turn else old_tree
        ev = new_eval if is_new_turn else old_eval

        tree.run_simulations(
            sims_per_move,
            train_cfg.virtual_loss_batch,
            ev, add_noise=False,
            seed=seed_base + move_num,
        )
        action = tree.best_action()

        if action == N * N:
            game.pass_move()
        else:
            game.make_move(action // N, action % N)

        new_tree.advance(action)
        old_tree.advance(action)

    while game.status == go_engine.PLAYING:
        game.pass_move()

    if game.status == go_engine.BLACK_WIN:
        winner = go_engine.BLACK
    elif game.status == go_engine.WHITE_WIN:
        winner = go_engine.WHITE
    else:
        return -1

    return 1 if winner == new_color else 0


def main():
    parser = argparse.ArgumentParser(
        description="Paired head-to-head match between two AlphaZero checkpoints.")
    parser.add_argument("--board-size", type=int, default=9, choices=[9, 13, 19])
    parser.add_argument("--new", required=True, help="Path to NEW checkpoint (.pt)")
    parser.add_argument("--old", required=True, help="Path to OLD checkpoint (.pt)")
    parser.add_argument("--pairs", type=int, default=50,
                        help="Number of paired matchups (each pair = 2 games)")
    parser.add_argument("--sims", type=int, default=100,
                        help="MCTS simulations per move (default 100)")
    parser.add_argument("--random-moves", type=int, default=4,
                        help="Opening ply played by uniform random legal move "
                             "(shared across both games of a pair). Default 4.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_cfg, train_cfg = CONFIGS[args.board_size]
    sims_per_move = min(args.sims, train_cfg.num_simulations)

    print(f"Paired matchup eval — {args.board_size}x{args.board_size} Go")
    print(f"  NEW: {args.new}")
    print(f"  OLD: {args.old}")
    print(f"  Pairs: {args.pairs} ({args.pairs * 2} games)  "
          f"Sims/move: {sims_per_move}  "
          f"Random opening ply: {args.random_moves}  Device: {device}")
    print()

    new_net = load_net(args.new, model_cfg, device)
    old_net = load_net(args.old, model_cfg, device)

    rng = np.random.default_rng(args.seed)

    pair_scores = []  # 0.0, 0.5, or 1.0 per pair
    both_wins = 0
    one_wins = 0
    zero_wins = 0
    t0 = time.time()
    N = model_cfg.board_size
    GameClass = GAME_CLASS[N]

    for pair_i in range(args.pairs):
        opening = generate_opening(rng, GameClass, train_cfg, N, args.random_moves)

        # Game A: NEW = BLACK
        rA = play_from_opening(
            new_net, old_net, device, model_cfg, train_cfg,
            opening, go_engine.BLACK,
            sims_per_move, seed_base=pair_i * 200003)

        # Game B: NEW = WHITE
        rB = play_from_opening(
            new_net, old_net, device, model_cfg, train_cfg,
            opening, go_engine.WHITE,
            sims_per_move, seed_base=pair_i * 200003 + 100001)

        new_wins_in_pair = (rA == 1) + (rB == 1)
        if new_wins_in_pair == 2:
            score = 1.0
            both_wins += 1
        elif new_wins_in_pair == 1:
            score = 0.5
            one_wins += 1
        else:
            score = 0.0
            zero_wins += 1
        pair_scores.append(score)

        if (pair_i + 1) % 10 == 0 or pair_i + 1 == args.pairs:
            done = pair_i + 1
            avg = sum(pair_scores) / done
            elapsed = time.time() - t0
            print(f"  [{done}/{args.pairs} pairs]  "
                  f"NEW both-wins={both_wins}  one-wins={one_wins}  "
                  f"zero-wins={zero_wins}  score={avg:.3f}  "
                  f"({elapsed:.0f}s)")

    total_score = sum(pair_scores)
    n = len(pair_scores)
    avg = total_score / n
    lo, hi = binomial_ci_95(total_score, n)

    print()
    print(f"Final: NEW pair-wins {both_wins}–{zero_wins} OLD "
          f"({one_wins} tied pairs)")
    print(f"  Aggregate score: {avg:.3f}  (95% CI: {lo:.3f} – {hi:.3f})")
    print(f"  Elapsed: {time.time() - t0:.1f}s")

    if avg >= 0.60:
        verdict = "STRONG improvement"
    elif avg >= 0.55:
        verdict = "meaningful improvement"
    elif avg >= 0.45:
        verdict = "roughly matched — unclear if improved"
    else:
        verdict = "REGRESSION"
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
