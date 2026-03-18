"""Self-play PPO training for Chinese Chess.

The same network plays both Red and Black. Each game produces two
trajectories (one per side). Terminal rewards are assigned to both
the winner (+1) and loser (-1).

Periodically evaluates against a random opponent to track progress.
"""

import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# Setup paths — engine is in ChineseChess/backend/, env/agents in ChessRL/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
ENGINE_DIR = os.path.join(os.path.dirname(PROJECT_DIR), "ChineseChess", "backend")
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, ENGINE_DIR)

import torch

from engine.board import RED, BLACK
from engine.game import GameStatus
from env.chess_env import ChineseChessEnv
from agents.ppo_agent import PPOAgent

# ---------- Hyperparameters ----------
TOTAL_STEPS = 3_000_000
GAMES_PER_BATCH = 64
MAX_STEPS_PER_GAME = 200
PRINT_EVERY = 5
EVAL_EVERY = 20       # batches between eval vs random
EVAL_GAMES = 20       # games per eval round
SAVE_EVERY = 50       # batches between checkpoints
SAVE_DIR = SCRIPT_DIR
# --------------------------------------


def play_game(env, agent):
    """Play one self-play game, return separate Red/Black trajectories."""
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
            # The env gives +1 to the winner's last action, but the loser's
            # last action got reward=0. Fix it: assign -1 to the loser.
            if env.game.status == GameStatus.RED_WIN and len(black["rewards"]) > 0:
                black["rewards"][-1] = -1.0
                black["dones"][-1] = True
            elif env.game.status == GameStatus.BLACK_WIN and len(red["rewards"]) > 0:
                red["rewards"][-1] = -1.0
                red["dones"][-1] = True
            else:
                # Draw or truncation — mark both sides as terminal
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

    return red, black, steps, result


def evaluate_vs_random(agent, env, num_games=20):
    """Evaluate agent (as Red) vs random opponent (as Black).

    Returns (wins, losses, draws).
    """
    wins, losses, draws = 0, 0, 0

    for _ in range(num_games):
        obs, info = env.reset()

        while True:
            if env.current_turn == RED:
                # Agent plays Red (greedy)
                action = agent.select_action_greedy(obs, info["action_mask"])
            else:
                # Random plays Black
                legal = np.where(info["action_mask"])[0]
                action = np.random.choice(legal)

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                if env.game.status == GameStatus.RED_WIN:
                    wins += 1
                elif env.game.status == GameStatus.BLACK_WIN:
                    losses += 1
                else:
                    draws += 1
                break

    return wins, losses, draws


def train():
    env = ChineseChessEnv(max_steps=MAX_STEPS_PER_GAME)
    agent = PPOAgent()

    param_count = sum(p.numel() for p in agent.network.parameters())
    print(f"Device: {agent.device}")
    print(f"Model parameters: {param_count:,} ({param_count * 4 / 1e6:.1f} MB)")
    print(f"Training: {TOTAL_STEPS:,} total steps, {GAMES_PER_BATCH} games/batch")
    print(f"Max steps/game: {MAX_STEPS_PER_GAME}")
    print()

    # Check for resume
    checkpoint_path = os.path.join(SAVE_DIR, "ppo_checkpoint.pt")
    if os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        print(f"Resumed from {checkpoint_path}")

    # Tracking
    total_steps = 0
    batch_count = 0
    game_results = {"red_win": 0, "black_win": 0, "draw": 0, "truncated": 0}
    game_lengths = []
    eval_history = []
    metric_history = []
    best_win_rate = -1.0

    start_time = time.time()

    while total_steps < TOTAL_STEPS:
        # --- Collect games ---
        batch_obs, batch_actions, batch_masks = [], [], []
        batch_log_probs, batch_advantages, batch_returns = [], [], []
        batch_steps = 0

        for _ in range(GAMES_PER_BATCH):
            red, black, steps, result = play_game(env, agent)
            batch_steps += steps
            game_results[result] += 1
            game_lengths.append(steps)

            # Compute GAE for each side's trajectory
            for traj in [red, black]:
                if len(traj["obs"]) == 0:
                    continue
                advantages, returns = PPOAgent.compute_gae(
                    traj["rewards"], traj["values"], traj["dones"],
                    agent.gamma, agent.gae_lambda,
                )
                batch_obs.extend(traj["obs"])
                batch_actions.extend(traj["actions"])
                batch_masks.extend(traj["masks"])
                batch_log_probs.extend(traj["log_probs"])
                batch_advantages.extend(advantages.tolist())
                batch_returns.extend(returns.tolist())

        total_steps += batch_steps

        # --- PPO update ---
        obs_arr = np.array(batch_obs, dtype=np.float32)
        actions_arr = np.array(batch_actions, dtype=np.int64)
        masks_arr = np.array(batch_masks, dtype=np.bool_)
        log_probs_arr = np.array(batch_log_probs, dtype=np.float32)
        advantages_arr = np.array(batch_advantages, dtype=np.float32)
        returns_arr = np.array(batch_returns, dtype=np.float32)

        metrics = agent.update(obs_arr, actions_arr, masks_arr, log_probs_arr, advantages_arr, returns_arr)
        batch_count += 1

        metric_history.append({
            "steps": total_steps,
            "avg_game_len": np.mean(game_lengths[-GAMES_PER_BATCH:]),
            **metrics,
        })

        # --- Print progress ---
        if batch_count % PRINT_EVERY == 0:
            elapsed = time.time() - start_time
            fps = total_steps / elapsed
            recent_len = np.mean(game_lengths[-GAMES_PER_BATCH * PRINT_EVERY:])
            total_games = sum(game_results.values())
            rw = game_results["red_win"]
            bw = game_results["black_win"]
            dr = game_results["draw"] + game_results["truncated"]
            print(
                f"Batch {batch_count:4d} | "
                f"Steps: {total_steps:>9,d} | "
                f"Games: {total_games:5d} | "
                f"R/B/D: {rw}/{bw}/{dr} | "
                f"Len: {recent_len:5.1f} | "
                f"PL: {metrics['policy_loss']:.4f} | "
                f"VL: {metrics['value_loss']:.4f} | "
                f"Ent: {metrics['entropy']:.3f} | "
                f"FPS: {fps:.0f}"
            )

        # --- Evaluate vs random ---
        if batch_count % EVAL_EVERY == 0:
            wins, losses, draws = evaluate_vs_random(agent, env, EVAL_GAMES)
            win_rate = wins / EVAL_GAMES
            eval_history.append({"steps": total_steps, "wins": wins, "losses": losses, "draws": draws})
            print(f"  >> Eval vs Random: {wins}W / {losses}L / {draws}D  (win rate: {win_rate:.0%})")

            if win_rate > best_win_rate:
                best_win_rate = win_rate
                agent.save(os.path.join(SAVE_DIR, "ppo_best.pt"))

        # --- Checkpoint ---
        if batch_count % SAVE_EVERY == 0:
            agent.save(checkpoint_path)

    # Save final
    agent.save(os.path.join(SAVE_DIR, "ppo_final.pt"))

    # --- Plot training curves ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    steps_list = [m["steps"] for m in metric_history]

    axes[0, 0].plot(steps_list, [m["policy_loss"] for m in metric_history])
    axes[0, 0].set_title("Policy Loss")
    axes[0, 0].set_xlabel("Steps")

    axes[0, 1].plot(steps_list, [m["value_loss"] for m in metric_history])
    axes[0, 1].set_title("Value Loss")
    axes[0, 1].set_xlabel("Steps")

    axes[1, 0].plot(steps_list, [m["entropy"] for m in metric_history])
    axes[1, 0].set_title("Entropy")
    axes[1, 0].set_xlabel("Steps")

    axes[1, 1].plot(steps_list, [m["avg_game_len"] for m in metric_history])
    axes[1, 1].set_title("Avg Game Length")
    axes[1, 1].set_xlabel("Steps")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_curves.png"), dpi=150)
    plt.close()

    # Plot eval results
    if eval_history:
        fig, ax = plt.subplots(figsize=(10, 4))
        eval_steps = [e["steps"] for e in eval_history]
        ax.plot(eval_steps, [e["wins"] / EVAL_GAMES for e in eval_history], label="Win rate vs Random")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Win Rate")
        ax.set_title("Evaluation: Agent (Red) vs Random (Black)")
        ax.set_ylim(0, 1.05)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "eval_curve.png"), dpi=150)
        plt.close()

    elapsed = time.time() - start_time
    total_games = sum(game_results.values())
    print(f"\nTraining complete!")
    print(f"Total steps: {total_steps:,}")
    print(f"Total games: {total_games:,}")
    print(f"Results: R={game_results['red_win']} B={game_results['black_win']} "
          f"D={game_results['draw']} T={game_results['truncated']}")
    print(f"Best win rate vs random: {best_win_rate:.0%}")
    print(f"Time: {elapsed / 60:.1f} min ({elapsed / 3600:.1f} hrs)")
    print(f"Avg FPS: {total_steps / elapsed:.0f}")


if __name__ == "__main__":
    train()
