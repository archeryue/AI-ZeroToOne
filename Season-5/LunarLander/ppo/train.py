"""Train PPO on LunarLander-v3."""

import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import torch

from agent import PPOAgent

# --------------- Hyperparameters ---------------
TOTAL_TIMESTEPS = 500000
MAX_STEPS = 1000
PRINT_EVERY = 10
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
# -----------------------------------------------


def train():
    env = gym.make("LunarLander-v3")
    agent = PPOAgent()

    print(f"Device: {agent.device}")
    print(f"Training PPO on LunarLander-v3 for {TOTAL_TIMESTEPS} timesteps...")
    print(f"Rollout: {agent.rollout_steps} steps, Update: {agent.update_epochs} epochs\n")

    reward_history = []
    episode_reward = 0.0
    episode_count = 0
    best_avg_reward = -float("inf")

    state, _ = env.reset()
    total_steps = 0

    while total_steps < TOTAL_TIMESTEPS:
        # Collect rollout
        for _ in range(agent.rollout_steps):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.buffer.store(state, action, log_prob, reward, float(done), value)
            state = next_state
            episode_reward += reward
            total_steps += 1

            if done:
                reward_history.append(episode_reward)
                episode_count += 1
                episode_reward = 0.0
                state, _ = env.reset()

                if episode_count % PRINT_EVERY == 0:
                    avg_reward = np.mean(reward_history[-100:])
                    print(
                        f"Episode {episode_count:4d} | "
                        f"Steps: {total_steps:7d} | "
                        f"Reward: {reward_history[-1]:7.1f} | "
                        f"Avg(100): {avg_reward:7.1f}"
                    )

        # Compute last value for GAE bootstrap
        with torch.no_grad():
            _, last_value = agent.network(
                torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            )
            last_value = last_value.item()

        # PPO update
        metrics = agent.update(last_value)

        # Save best model
        if len(reward_history) >= 100:
            avg_reward = np.mean(reward_history[-100:])
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(os.path.join(SAVE_DIR, "ppo_best.pt"))

    # Save final model
    agent.save(os.path.join(SAVE_DIR, "ppo_final.pt"))
    env.close()

    # Plot reward curve
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, alpha=0.3, label="Episode reward")
    if len(reward_history) >= 100:
        smoothed = np.convolve(reward_history, np.ones(100) / 100, mode="valid")
        plt.plot(range(99, len(reward_history)), smoothed, label="Avg(100)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO on LunarLander-v3")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "ppo_training_curve.png"), dpi=150)
    plt.close()
    print(f"\nTraining curve saved to ppo_training_curve.png")
    print(f"Best avg reward: {best_avg_reward:.1f}")
    print(f"Total episodes: {episode_count}")


if __name__ == "__main__":
    train()
