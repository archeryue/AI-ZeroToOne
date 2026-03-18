"""Candidate 3: PPO + Reward Shaping + Opponent Curriculum for Chinese Chess.

Builds on Candidate 2 (material-based reward shaping) by adding curriculum
training against increasingly strong opponents:

  Phase A (0-1M steps):   50% self-play + 50% vs Random
  Phase B (1M-2M steps):  50% self-play + 25% vs Random + 25% vs Greedy
  Phase C (2M-3M steps):  50% self-play + 25% vs Greedy + 25% vs Minimax

This forces the agent to learn from stronger play patterns instead of
only reinforcing its own (possibly bad) habits.
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ENGINE_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "ChineseChess", "backend")
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, ENGINE_DIR)

import torch

from engine.board import RED, BLACK
from engine.game import GameStatus
from env.chess_env import ChineseChessEnv
from env.reward_shaping import RewardShapingWrapper
from env.action_space import encode_move
from agents.ppo_agent import PPOAgent
from ai.random_ai import RandomAI
from ai.greedy_ai import GreedyAI
from ai.minimax_ai import MinimaxAI

# ---------- Hyperparameters ----------
TOTAL_STEPS = 3_000_000
GAMES_PER_BATCH = 64
MAX_STEPS_PER_GAME = 200
PRINT_EVERY = 5
EVAL_EVERY = 20
EVAL_GAMES = 20
SAVE_EVERY = 50
REWARD_SCALE = 0.01
SAVE_DIR = os.path.join(SCRIPT_DIR, "candidate3")
# --------------------------------------

# Curriculum schedule: (step_threshold, [(opponent, weight), ...])
# self-play weight is implicit (1 - sum of weights)
CURRICULUM = [
    (0,         [("random", 0.50)]),                          # Phase A
    (1_000_000, [("random", 0.25), ("greedy", 0.25)]),        # Phase B
    (2_000_000, [("greedy", 0.25), ("minimax", 0.25)]),       # Phase C
]


def get_curriculum_mix(total_steps):
    """Return the opponent mix for the current training phase."""
    mix = CURRICULUM[0][1]
    for threshold, m in CURRICULUM:
        if total_steps >= threshold:
            mix = m
    return mix


def move_to_action(move):
    return encode_move(move.from_row, move.from_col, move.to_row, move.to_col)


def play_self_play_game(env, agent):
    """Self-play game (same as Candidates 1/2)."""
    obs, info = env.reset()

    red = {"obs": [], "actions": [], "masks": [], "log_probs": [], "values": [], "rewards": [], "dones": []}
    black = {"obs": [], "actions": [], "masks": [], "log_probs": [], "values": [], "rewards": [], "dones": []}

    steps = 0
    while True:
        side = env.current_turn
        traj = red if side == RED else black

        mask = info["action_mask"]
        action, log_prob, value = agent.select_action(obs, mask)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

        traj["obs"].append(obs)
        traj["actions"].append(action)
        traj["masks"].append(mask)
        traj["log_probs"].append(log_prob)
        traj["values"].append(value)
        traj["rewards"].append(reward)
        traj["dones"].append(done)

        if done:
            if env.game.status == GameStatus.RED_WIN and len(black["rewards"]) > 0:
                black["rewards"][-1] = -1.0
                black["dones"][-1] = True
            elif env.game.status == GameStatus.BLACK_WIN and len(red["rewards"]) > 0:
                red["rewards"][-1] = -1.0
                red["dones"][-1] = True
            else:
                if len(red["rewards"]) > 0:
                    red["dones"][-1] = True
                if len(black["rewards"]) > 0:
                    black["dones"][-1] = True

            result = (
                "red_win" if env.game.status == GameStatus.RED_WIN else
                "black_win" if env.game.status == GameStatus.BLACK_WIN else
                "draw" if env.game.status == GameStatus.DRAW else
                "truncated"
            )
            break

        obs = next_obs

    return [red, black], steps, result


def play_vs_opponent_game(env, agent, opponent):
    """Play a game: PPO vs fixed opponent. PPO plays a random side.

    Returns trajectories only for the PPO side (the opponent's moves
    are not used for training).
    """
    ppo_color = RED if np.random.random() < 0.5 else BLACK
    obs, info = env.reset()

    traj = {"obs": [], "actions": [], "masks": [], "log_probs": [], "values": [], "rewards": [], "dones": []}

    steps = 0
    while True:
        current = env.current_turn

        if current == ppo_color:
            # PPO's turn
            mask = info["action_mask"]
            action, log_prob, value = agent.select_action(obs, mask)

            traj["obs"].append(obs)
            traj["actions"].append(action)
            traj["masks"].append(mask)
            traj["log_probs"].append(log_prob)
            traj["values"].append(value)
        else:
            # Opponent's turn
            move = opponent.choose_move(env.game)
            action = move_to_action(move)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1

        # Only record reward for PPO's moves
        if current == ppo_color:
            traj["rewards"].append(reward)
            traj["dones"].append(done)

        if done:
            # Assign terminal reward if PPO didn't get it directly
            if len(traj["rewards"]) > 0:
                status = env.game.status
                if status == GameStatus.RED_WIN:
                    traj["rewards"][-1] = 1.0 if ppo_color == RED else -1.0
                elif status == GameStatus.BLACK_WIN:
                    traj["rewards"][-1] = 1.0 if ppo_color == BLACK else -1.0
                traj["dones"][-1] = True

            ppo_won = (
                (status == GameStatus.RED_WIN and ppo_color == RED) or
                (status == GameStatus.BLACK_WIN and ppo_color == BLACK)
            )
            result = "ppo_win" if ppo_won else (
                "ppo_loss" if status in (GameStatus.RED_WIN, GameStatus.BLACK_WIN) else "draw"
            )
            break

        obs = next_obs

    return [traj] if len(traj["obs"]) > 0 else [], steps, result


def evaluate_vs_random(agent, eval_env, num_games=20):
    """Evaluate agent (as Red) vs random opponent."""
    wins, losses, draws = 0, 0, 0

    for _ in range(num_games):
        obs, info = eval_env.reset()
        while True:
            if eval_env.current_turn == RED:
                action = agent.select_action_greedy(obs, info["action_mask"])
            else:
                legal = np.where(info["action_mask"])[0]
                action = np.random.choice(legal)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            if terminated or truncated:
                if eval_env.game.status == GameStatus.RED_WIN:
                    wins += 1
                elif eval_env.game.status == GameStatus.BLACK_WIN:
                    losses += 1
                else:
                    draws += 1
                break

    return wins, losses, draws


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    base_env = ChineseChessEnv(max_steps=MAX_STEPS_PER_GAME)
    env = RewardShapingWrapper(base_env, scale=REWARD_SCALE)
    eval_env = ChineseChessEnv(max_steps=300)

    agent = PPOAgent()

    # Fixed opponents
    opponents = {
        "random": RandomAI(),
        "greedy": GreedyAI(),
        "minimax": MinimaxAI(depth=2),  # depth=2 for faster curriculum games
    }

    param_count = sum(p.numel() for p in agent.network.parameters())
    print(f"=== Candidate 3: PPO + Reward Shaping + Curriculum ===")
    print(f"Device: {agent.device}")
    print(f"Model parameters: {param_count:,}")
    print(f"Reward shaping scale: {REWARD_SCALE}")
    print(f"Training: {TOTAL_STEPS:,} total steps, {GAMES_PER_BATCH} games/batch")
    print(f"Curriculum: Phase A (vs Random) → Phase B (+Greedy) → Phase C (+Minimax)")
    print()

    checkpoint_path = os.path.join(SAVE_DIR, "ppo_checkpoint.pt")
    if os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        print(f"Resumed from {checkpoint_path}")

    total_steps = 0
    batch_count = 0
    game_results = {"red_win": 0, "black_win": 0, "draw": 0, "truncated": 0,
                    "ppo_win": 0, "ppo_loss": 0}
    game_lengths = []
    eval_history = []
    metric_history = []
    best_win_rate = -1.0

    start_time = time.time()

    while total_steps < TOTAL_STEPS:
        # Determine curriculum mix
        mix = get_curriculum_mix(total_steps)
        self_play_weight = 1.0 - sum(w for _, w in mix)
        num_self_play = int(GAMES_PER_BATCH * self_play_weight)
        opponent_games = []
        for opp_name, weight in mix:
            n = int(GAMES_PER_BATCH * weight)
            opponent_games.extend([(opp_name, opponents[opp_name])] * n)
        # Fill remainder with self-play
        num_self_play = GAMES_PER_BATCH - len(opponent_games)

        batch_obs, batch_actions, batch_masks = [], [], []
        batch_log_probs, batch_advantages, batch_returns = [], [], []
        batch_steps = 0

        # Self-play games
        for _ in range(num_self_play):
            trajs, steps, result = play_self_play_game(env, agent)
            batch_steps += steps
            game_results[result] = game_results.get(result, 0) + 1
            game_lengths.append(steps)

            for traj in trajs:
                if len(traj["obs"]) == 0:
                    continue
                adv, ret = PPOAgent.compute_gae(
                    traj["rewards"], traj["values"], traj["dones"],
                    agent.gamma, agent.gae_lambda,
                )
                batch_obs.extend(traj["obs"])
                batch_actions.extend(traj["actions"])
                batch_masks.extend(traj["masks"])
                batch_log_probs.extend(traj["log_probs"])
                batch_advantages.extend(adv.tolist())
                batch_returns.extend(ret.tolist())

        # Opponent games
        for opp_name, opp in opponent_games:
            trajs, steps, result = play_vs_opponent_game(env, agent, opp)
            batch_steps += steps
            game_results[result] = game_results.get(result, 0) + 1
            game_lengths.append(steps)

            for traj in trajs:
                if len(traj["obs"]) == 0:
                    continue
                adv, ret = PPOAgent.compute_gae(
                    traj["rewards"], traj["values"], traj["dones"],
                    agent.gamma, agent.gae_lambda,
                )
                batch_obs.extend(traj["obs"])
                batch_actions.extend(traj["actions"])
                batch_masks.extend(traj["masks"])
                batch_log_probs.extend(traj["log_probs"])
                batch_advantages.extend(adv.tolist())
                batch_returns.extend(ret.tolist())

        total_steps += batch_steps

        if len(batch_obs) == 0:
            continue

        obs_arr = np.array(batch_obs, dtype=np.float32)
        actions_arr = np.array(batch_actions, dtype=np.int64)
        masks_arr = np.array(batch_masks, dtype=np.bool_)
        log_probs_arr = np.array(batch_log_probs, dtype=np.float32)
        advantages_arr = np.array(batch_advantages, dtype=np.float32)
        returns_arr = np.array(batch_returns, dtype=np.float32)

        metrics = agent.update(obs_arr, actions_arr, masks_arr, log_probs_arr, advantages_arr, returns_arr)
        batch_count += 1

        phase = "A" if total_steps < 1_000_000 else "B" if total_steps < 2_000_000 else "C"
        metric_history.append({
            "steps": total_steps,
            "phase": phase,
            "avg_game_len": np.mean(game_lengths[-GAMES_PER_BATCH:]),
            **metrics,
        })

        if batch_count % PRINT_EVERY == 0:
            elapsed = time.time() - start_time
            fps = total_steps / elapsed
            recent_len = np.mean(game_lengths[-GAMES_PER_BATCH * PRINT_EVERY:])
            pw = game_results.get("ppo_win", 0)
            pl = game_results.get("ppo_loss", 0)
            rw = game_results.get("red_win", 0)
            bw = game_results.get("black_win", 0)
            total_games = sum(game_results.values())
            print(
                f"Batch {batch_count:4d} [Phase {phase}] | "
                f"Steps: {total_steps:>9,d} | "
                f"Games: {total_games:5d} | "
                f"Self R/B: {rw}/{bw} | "
                f"vsAI W/L: {pw}/{pl} | "
                f"Len: {recent_len:5.1f} | "
                f"PL: {metrics['policy_loss']:.4f} | "
                f"VL: {metrics['value_loss']:.4f} | "
                f"Ent: {metrics['entropy']:.3f} | "
                f"FPS: {fps:.0f}"
            )

        if batch_count % EVAL_EVERY == 0:
            wins, losses, draws = evaluate_vs_random(agent, eval_env, EVAL_GAMES)
            win_rate = wins / EVAL_GAMES
            eval_history.append({"steps": total_steps, "wins": wins, "losses": losses, "draws": draws})
            print(f"  >> Eval vs Random: {wins}W / {losses}L / {draws}D  (win rate: {win_rate:.0%})")

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                agent.save(os.path.join(SAVE_DIR, "ppo_best.pt"))

        if batch_count % SAVE_EVERY == 0:
            agent.save(checkpoint_path)

    agent.save(os.path.join(SAVE_DIR, "ppo_final.pt"))

    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    steps_list = [m["steps"] for m in metric_history]

    axes[0, 0].plot(steps_list, [m["policy_loss"] for m in metric_history])
    axes[0, 0].set_title("Policy Loss")
    axes[0, 0].set_xlabel("Steps")
    for thresh in [1_000_000, 2_000_000]:
        axes[0, 0].axvline(x=thresh, color="r", linestyle="--", alpha=0.3)

    axes[0, 1].plot(steps_list, [m["value_loss"] for m in metric_history])
    axes[0, 1].set_title("Value Loss")
    axes[0, 1].set_xlabel("Steps")

    axes[1, 0].plot(steps_list, [m["entropy"] for m in metric_history])
    axes[1, 0].set_title("Entropy")
    axes[1, 0].set_xlabel("Steps")

    axes[1, 1].plot(steps_list, [m["avg_game_len"] for m in metric_history])
    axes[1, 1].set_title("Avg Game Length")
    axes[1, 1].set_xlabel("Steps")

    plt.suptitle("Candidate 3: PPO + Reward Shaping + Curriculum")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"), dpi=150)
    plt.close()

    if eval_history:
        fig, ax = plt.subplots(figsize=(10, 4))
        eval_steps = [e["steps"] for e in eval_history]
        ax.plot(eval_steps, [e["wins"] / EVAL_GAMES for e in eval_history], label="Win rate vs Random")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Win Rate")
        ax.set_title("Candidate 3: Eval vs Random")
        ax.set_ylim(0, 1.05)
        for thresh in [1_000_000, 2_000_000]:
            ax.axvline(x=thresh, color="r", linestyle="--", alpha=0.3, label=f"Phase transition" if thresh == 1_000_000 else "")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "eval_curve.png"), dpi=150)
        plt.close()

    elapsed = time.time() - start_time
    total_games = sum(game_results.values())
    print(f"\nTraining complete!")
    print(f"Total steps: {total_steps:,}")
    print(f"Total games: {total_games:,}")
    print(f"Results: Self-play R={game_results.get('red_win',0)} B={game_results.get('black_win',0)} "
          f"T={game_results.get('truncated',0)} | "
          f"vs AI W={game_results.get('ppo_win',0)} L={game_results.get('ppo_loss',0)}")
    print(f"Best win rate vs random: {best_win_rate:.0%}")
    print(f"Time: {elapsed / 60:.1f} min ({elapsed / 3600:.1f} hrs)")
    print(f"Avg FPS: {total_steps / elapsed:.0f}")


if __name__ == "__main__":
    train()
