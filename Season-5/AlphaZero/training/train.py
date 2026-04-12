"""AlphaZero training entry point.

Orchestrates the self-play → train → eval loop:
  1. Self-play: parallel games with C++ MCTS + batched GPU inference
  2. Train: sample from replay buffer, update network
  3. Eval: periodically evaluate vs baselines
  4. Repeat

Usage:
    # Full training (GPU, 5 workers × ~52 games = 256 parallel)
    python -m training.train --board-size 9 --iterations 100

    # Smoke test (CPU, tiny model, 3 iterations)
    python -m training.train --board-size 9 --smoke-test
"""

import argparse
import os
import sys
import time
import json

# Limit CPU thread pools BEFORE importing torch/numpy. The orchestrator's
# CPU work is tiny (host↔device copies, augmentation) — torch's default
# 128-thread intra-op pool causes severe contention with the C++ self-play
# workers and slows self-play by ~7x. See worker contention diagnosis.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import torch
import numpy as np

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # Another module already set interop threads; ignore silently.
    pass
# Let fp32 matmuls use TF32 on Ampere/Ada — no quality cost for this model.
torch.set_float32_matmul_precision("high")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import ModelConfig, TrainingConfig, CONFIGS
from model.network import AlphaZeroNet
from training.replay_buffer import ReplayBuffer
from training.parallel_self_play import ParallelSelfPlay
from training.trainer import Trainer


def evaluate_vs_random(net, device, model_cfg, train_cfg, num_games=100):
    """Quick evaluation: AlphaZero (with MCTS) vs random player."""
    import go_engine
    from training.self_play import GAME_CLASS, MCTS_CLASS, make_evaluator

    N = model_cfg.board_size
    GameClass = GAME_CLASS[N]
    MCTSClass = MCTS_CLASS[N]
    evaluator = make_evaluator(net, device)

    net.eval()
    wins = 0

    for game_i in range(num_games):
        game = GameClass(train_cfg.komi)
        tree = MCTSClass(game, train_cfg.c_puct,
                         train_cfg.dirichlet_alpha, train_cfg.dirichlet_epsilon)

        az_color = go_engine.BLACK
        seed = game_i * 1000

        for move_num in range(train_cfg.max_game_moves):
            if game.status != go_engine.PLAYING:
                break

            if game.current_turn == az_color:
                tree.run_simulations(
                    min(train_cfg.num_simulations, 100),
                    train_cfg.virtual_loss_batch,
                    evaluator, add_noise=False, seed=seed + move_num)
                action = tree.best_action()
                if action == N * N:
                    game.pass_move()
                else:
                    game.make_move(action // N, action % N)
                tree.advance(action)
            else:
                legal = game.get_legal_moves()
                if legal and np.random.random() > 0.05:
                    r, c = legal[np.random.randint(len(legal))]
                    action = r * N + c
                    game.make_move(r, c)
                else:
                    action = N * N
                    game.pass_move()
                tree.advance(action)

        while game.status == go_engine.PLAYING:
            game.pass_move()

        if game.status == go_engine.BLACK_WIN:
            wins += 1

    return wins / num_games


def main():
    parser = argparse.ArgumentParser(description="AlphaZero Training")
    parser.add_argument("--board-size", type=int, default=9, choices=[9, 13, 19])
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of self-play → train iterations")
    parser.add_argument("--games-per-iter", type=int, default=None,
                        help="Override games per iteration")
    parser.add_argument("--num-workers", type=int, default=5,
                        help="Number of self-play worker threads")
    parser.add_argument("--parallel-games", type=int, default=None,
                        help="Override total parallel games (default: 256)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Directory for checkpoints and logs")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu/mps)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick test with minimal settings")
    args = parser.parse_args()

    # Select config
    model_cfg, train_cfg = CONFIGS[args.board_size]

    if args.smoke_test:
        model_cfg = ModelConfig(
            board_size=args.board_size, num_blocks=2, channels=32)
        train_cfg.num_simulations = 16
        train_cfg.num_games_per_iter = 32
        train_cfg.num_parallel_games = 8
        train_cfg.train_steps_per_iter = 5
        train_cfg.buffer_size = 10_000
        train_cfg.batch_size = 32
        args.iterations = 3
        args.num_workers = 2

    if args.games_per_iter:
        train_cfg.num_games_per_iter = args.games_per_iter
    if args.parallel_games:
        train_cfg.num_parallel_games = args.parallel_games

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    num_workers = args.num_workers

    print(f"AlphaZero Training — {args.board_size}x{args.board_size} Go")
    print(f"  Device: {device}")
    print(f"  Model: {model_cfg.num_blocks}b x {model_cfg.channels}ch")
    print(f"  Workers: {num_workers}, parallel games: {train_cfg.num_parallel_games}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Games/iter: {train_cfg.num_games_per_iter}")
    print(f"  Sims/move: {train_cfg.num_simulations}")
    print(f"  Buffer: {train_cfg.buffer_size}")
    print()

    # Create model
    net = AlphaZeroNet(model_cfg).to(device)
    print(f"  Parameters: {net.param_count():,}")

    # Create trainer and buffer
    trainer = Trainer(net, train_cfg, device)
    buffer = ReplayBuffer(train_cfg.buffer_size, model_cfg.board_size)

    # Resume from checkpoint
    start_iter = 0
    if args.checkpoint and os.path.exists(args.checkpoint):
        state = trainer.load_checkpoint(args.checkpoint)
        start_iter = state.get("iteration", 0) + 1
        print(f"  Resumed from iteration {start_iter - 1}")

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "training_log.jsonl")

    print("\n--- Training Loop ---\n")

    for iteration in range(start_iter, args.iterations):
        iter_start = time.time()

        # 1. Parallel self-play (new workers each iteration to use updated net)
        sp_start = time.time()
        sp = ParallelSelfPlay(net, device, model_cfg, train_cfg,
                              num_workers=num_workers)
        sp_stats = sp.run_games(train_cfg.num_games_per_iter, buffer)
        sp_time = time.time() - sp_start

        avg_moves = sp_stats['positions'] / max(sp_stats['games'], 1)
        print(f"Iter {iteration:4d} | Self-play: {sp_stats['games']} games, "
              f"{sp_stats['positions']} pos ({avg_moves:.0f} avg moves), "
              f"{sp_stats['games_per_sec']:.1f} games/s, "
              f"buf={sp_stats['buffer_size']}, {sp_time:.1f}s")

        # 2. Train
        if len(buffer) >= train_cfg.batch_size:
            tr_start = time.time()
            tr_stats = trainer.train_epoch(buffer, args.iterations)
            tr_time = time.time() - tr_start

            print(f"         | Train: loss={tr_stats['loss']:.4f} "
                  f"(pi={tr_stats['policy_loss']:.4f}, v={tr_stats['value_loss']:.4f}), "
                  f"lr={tr_stats['lr']:.6f}, {tr_time:.1f}s")
        else:
            tr_stats = {}
            print(f"         | Train: skipped (buffer {len(buffer)} < "
                  f"batch {train_cfg.batch_size})")

        # 3. Evaluate periodically
        eval_stats = {}
        if (iteration + 1) % train_cfg.eval_interval == 0:
            ev_start = time.time()
            num_eval = 20 if args.smoke_test else 100
            win_rate = evaluate_vs_random(net, device, model_cfg, train_cfg,
                                          num_games=num_eval)
            ev_time = time.time() - ev_start
            eval_stats = {"vs_random_winrate": win_rate}
            print(f"         | Eval vs random: {win_rate:.1%} "
                  f"({num_eval} games, {ev_time:.1f}s)")

        # 4. Checkpoint
        if (iteration + 1) % train_cfg.checkpoint_interval == 0:
            ckpt_path = os.path.join(args.output_dir,
                                     f"checkpoint_{iteration:04d}.pt")
            trainer.save_checkpoint(ckpt_path, iteration, extra=eval_stats)
            print(f"         | Checkpoint saved: {ckpt_path}")

        # Log
        iter_time = time.time() - iter_start
        log_entry = {
            "iteration": iteration,
            "time": iter_time,
            "self_play": sp_stats,
            "train": tr_stats,
            "eval": eval_stats,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(f"         | Total: {iter_time:.1f}s")
        print()

    # Final checkpoint
    final_path = os.path.join(args.output_dir, "model_final.pt")
    trainer.save_checkpoint(final_path, args.iterations - 1)
    print(f"Training complete. Final model: {final_path}")


if __name__ == "__main__":
    main()
