"""Evaluate PPO agent vs other AIs (Random, Greedy, Minimax).

Plays N games with PPO as Red and N games with PPO as Black.

Usage:
    python evaluate_vs_minimax.py --opponent random --games 10
    python evaluate_vs_minimax.py --opponent greedy --games 10
    python evaluate_vs_minimax.py --opponent minimax --depth 3 --games 5
"""

import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ENGINE_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "ChineseChess", "backend")
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, ENGINE_DIR)

import numpy as np

from engine.board import RED, BLACK
from engine.game import Game, GameStatus
from engine.move import Move
from env.chess_env import ChineseChessEnv
from env.action_space import encode_move
from agents.ppo_agent import PPOAgent


def move_to_action(move: Move) -> int:
    """Convert a Move object to our flat action index."""
    return encode_move(move.from_row, move.from_col, move.to_row, move.to_col)


def make_opponent(name: str, depth: int = 3):
    """Create an AI opponent by name."""
    if name == "random":
        from ai.random_ai import RandomAI
        return RandomAI()
    elif name == "greedy":
        from ai.greedy_ai import GreedyAI
        return GreedyAI()
    elif name == "minimax":
        from ai.minimax_ai import MinimaxAI
        return MinimaxAI(depth=depth)
    else:
        raise ValueError(f"Unknown opponent: {name}")


def play_match(agent, opponent, env, ppo_color, max_steps=300):
    """Play one game: PPO vs opponent AI.

    Returns: (winner_str, steps, game_status)
    """
    obs, info = env.reset()
    steps = 0

    while True:
        current = env.current_turn

        if current == ppo_color:
            action = agent.select_action_greedy(obs, info["action_mask"])
        else:
            move = opponent.choose_move(env.game)
            action = move_to_action(move)

        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

        if terminated or truncated or steps >= max_steps:
            break

    status = env.game.status
    if status == GameStatus.RED_WIN:
        winner = "ppo" if ppo_color == RED else "opponent"
    elif status == GameStatus.BLACK_WIN:
        winner = "ppo" if ppo_color == BLACK else "opponent"
    else:
        winner = "draw"

    return winner, steps, status


def run_evaluation(agent, opponent, env, num_games, max_steps):
    """Run full evaluation: PPO as Red then as Black."""
    results = {"ppo": 0, "opponent": 0, "draw": 0}

    print(f"--- PPO (Red) vs {opponent.name} (Black) ---")
    for i in range(num_games):
        start = time.time()
        winner, steps, status = play_match(agent, opponent, env, RED, max_steps)
        elapsed = time.time() - start
        results[winner] += 1
        print(f"  Game {i+1}: {winner:>8s} in {steps:3d} steps ({elapsed:.1f}s) [{status.name}]")

    print(f"\n--- PPO (Black) vs {opponent.name} (Red) ---")
    for i in range(num_games):
        start = time.time()
        winner, steps, status = play_match(agent, opponent, env, BLACK, max_steps)
        elapsed = time.time() - start
        results[winner] += 1
        print(f"  Game {i+1}: {winner:>8s} in {steps:3d} steps ({elapsed:.1f}s) [{status.name}]")

    total = num_games * 2
    print(f"\n{'='*45}")
    print(f"PPO vs {opponent.name}: PPO {results['ppo']}W / {opponent.name} {results['opponent']}W / {results['draw']}D")
    print(f"PPO win rate: {results['ppo'] / total:.0%}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join(SCRIPT_DIR, "ppo_best.pt"))
    parser.add_argument("--opponent", default="random", choices=["random", "greedy", "minimax"])
    parser.add_argument("--depth", type=int, default=3, help="Minimax depth (only for minimax)")
    parser.add_argument("--games", type=int, default=10, help="Games per side")
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args()

    env = ChineseChessEnv(max_steps=args.max_steps)
    agent = PPOAgent(device="auto")
    agent.load(args.model)
    agent.network.eval()
    print(f"Loaded model: {args.model}")

    opponent = make_opponent(args.opponent, args.depth)
    print(f"Opponent: {opponent.name}")
    print(f"Playing {args.games} games per side ({args.games * 2} total)\n")

    run_evaluation(agent, opponent, env, args.games, args.max_steps)


if __name__ == "__main__":
    main()
