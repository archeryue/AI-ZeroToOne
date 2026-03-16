"""Train DQN on LunarLander-v3."""

import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from agent import DQNAgent

# --------------- Hyperparameters ---------------
NUM_EPISODES = 1000
MAX_STEPS = 1000
PRINT_EVERY = 20
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
# -----------------------------------------------


def train():
    env = gym.make("LunarLander-v3")
    agent = DQNAgent()

    print(f"Device: {agent.device}")
    print(f"Training DQN on LunarLander-v3 for {NUM_EPISODES} episodes...\n")

    reward_history = []
    best_avg_reward = -float("inf")

    for ep in range(1, NUM_EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0.0

        for step in range(MAX_STEPS):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, float(done))
            agent.update()

            state = next_state
            total_reward += reward

            if done:
                break

        reward_history.append(total_reward)
        avg_reward = np.mean(reward_history[-100:])

        if ep % PRINT_EVERY == 0:
            print(
                f"Episode {ep:4d} | "
                f"Reward: {total_reward:7.1f} | "
                f"Avg(100): {avg_reward:7.1f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

        # Save best model
        if avg_reward > best_avg_reward and ep >= 100:
            best_avg_reward = avg_reward
            agent.save(os.path.join(SAVE_DIR, "dqn_best.pt"))

    # Save final model
    agent.save(os.path.join(SAVE_DIR, "dqn_final.pt"))
    env.close()

    # Plot reward curve
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, alpha=0.3, label="Episode reward")
    # Smoothed curve (100-episode moving average)
    if len(reward_history) >= 100:
        smoothed = np.convolve(reward_history, np.ones(100) / 100, mode="valid")
        plt.plot(range(99, len(reward_history)), smoothed, label="Avg(100)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN on LunarLander-v3")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_curve.png"), dpi=150)
    plt.close()
    print(f"\nTraining curve saved to training_curve.png")
    print(f"Best avg reward: {best_avg_reward:.1f}")


if __name__ == "__main__":
    train()
