"""Reproduce the run4/run4b eval crash in isolation.

Both runs died during evaluate_vs_random with the main thread in
conv/batchnorm forward, no traceback, no OOM. This script loads
checkpoint_0001.pt (the last-known-healthy state from run4b) and
runs evaluate_vs_random with increasing game counts to see where
it fails — and with explicit memory-stats logging between games.
"""

import gc
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_float32_matmul_precision("high")

import go_engine
from model.config import CONFIGS
from model.network import AlphaZeroNet
from training.trainer import Trainer
from training.self_play import GAME_CLASS, MCTS_CLASS, make_evaluator


def gpu_stats():
    if not torch.cuda.is_available():
        return "no-cuda"
    alloc = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    return f"alloc={alloc:.0f}MB reserved={reserved:.0f}MB"


def main():
    device = torch.device("cuda")
    model_cfg, train_cfg = CONFIGS[13]

    print(f"[dbg] loading cold net...", flush=True)
    net = AlphaZeroNet(model_cfg).to(device)
    print(f"[dbg] params={net.param_count():,}, gpu={gpu_stats()}", flush=True)

    ckpt = "checkpoints/13x13_run4b/checkpoint_0001.pt"
    if os.path.exists(ckpt):
        print(f"[dbg] loading {ckpt}", flush=True)
        trainer = Trainer(net, train_cfg, device)
        trainer.load_checkpoint(ckpt)
    else:
        print(f"[dbg] no checkpoint at {ckpt}, using cold init", flush=True)

    net.eval()
    print(f"[dbg] pre-eval gpu={gpu_stats()}", flush=True)

    N = model_cfg.board_size
    GameClass = GAME_CLASS[N]
    MCTSClass = MCTS_CLASS[N]
    evaluator = make_evaluator(net, device)

    num_games = int(os.environ.get("NUM_GAMES", "50"))
    print(f"[dbg] running {num_games} games vs random, "
          f"sims={min(train_cfg.num_simulations, 100)}", flush=True)

    wins = 0
    t_start = time.time()
    for game_i in range(num_games):
        game = GameClass(train_cfg.komi)
        tree = MCTSClass(
            game, train_cfg.c_puct,
            train_cfg.dirichlet_alpha, train_cfg.dirichlet_epsilon)

        az_color = go_engine.BLACK
        seed = game_i * 1000
        fwd_count = 0

        for move_num in range(train_cfg.max_game_moves):
            if game.status != go_engine.PLAYING:
                break
            if game.current_turn == az_color:
                try:
                    tree.run_simulations(
                        min(train_cfg.num_simulations, 100),
                        train_cfg.virtual_loss_batch,
                        evaluator, add_noise=False, seed=seed + move_num)
                except BaseException as e:
                    print(f"[dbg] CRASH in run_simulations at game {game_i} "
                          f"move {move_num}: {type(e).__name__}: {e}",
                          flush=True)
                    traceback.print_exc()
                    raise
                action = tree.best_action()
                fwd_count += 1
                if action == N * N:
                    game.pass_move()
                else:
                    game.make_move(action // N, action % N)
                tree.advance(action)
            else:
                legal = game.get_legal_moves()
                if legal and np.random.random() > 0.05:
                    r, c = legal[np.random.randint(len(legal))]
                    game.make_move(r, c)
                    tree.advance(r * N + c)
                else:
                    game.pass_move()
                    tree.advance(N * N)

        while game.status == go_engine.PLAYING:
            game.pass_move()

        if game.status == go_engine.BLACK_WIN:
            wins += 1

        # Log every game so we see how far we got if it crashes
        elapsed = time.time() - t_start
        gr = gpu_stats()
        print(f"[dbg] game {game_i+1}/{num_games}: "
              f"{'W' if game.status == go_engine.BLACK_WIN else 'L'} "
              f"{move_num} moves {fwd_count} forwards | "
              f"wins={wins} elapsed={elapsed:.0f}s | {gr}",
              flush=True)

        # Drop tree explicitly so C++ memory is freed before next game
        del tree
        del game
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[dbg] DONE. {wins}/{num_games} = {wins/num_games:.1%}",
          flush=True)
    print(f"[dbg] final {gpu_stats()}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        print(f"\n[dbg] EXCEPTION: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise
