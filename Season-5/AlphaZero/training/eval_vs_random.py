"""Standalone vs-random evaluator — runs the same in-loop eval used
by train.py, but against an arbitrary checkpoint path.

Usage:
    python -m training.eval_vs_random \
        --checkpoint checkpoints/9x9_run1/preserved_iter0016.pt \
        --board-size 9 \
        --games 100
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.config import CONFIGS  # noqa: E402
from model.network import AlphaZeroNet  # noqa: E402
from training.train import evaluate_vs_random  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_cfg, train_cfg = CONFIGS[args.board_size]

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    net = AlphaZeroNet(model_cfg).to(device)
    net.load_state_dict(state["model_state_dict"])
    net.eval()

    print(f"Eval — {args.checkpoint} vs random")
    print(f"  Games: {args.games}  Device: {device}")
    t0 = time.time()
    winrate = evaluate_vs_random(net, device, model_cfg, train_cfg, num_games=args.games)
    elapsed = time.time() - t0
    print(f"  Win rate: {winrate:.1%}  ({args.games} games, {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
