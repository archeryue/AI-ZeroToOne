"""Train DQN on Atari Pong."""

import os
import time

import matplotlib.pyplot as plt
import numpy as np

from agent import DQNAgent
from wrappers import make_atari_env

# --------------- Hyperparameters ---------------
ENV_ID = "ALE/Pong-v5"
TOTAL_FRAMES = 2000000
PRINT_EVERY = 5
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
# -----------------------------------------------


def train(resume: bool = False):
    env = make_atari_env(ENV_ID)
    agent = DQNAgent(action_dim=env.action_space.n)

    if resume:
        ckpt_path = os.path.join(SAVE_DIR, "dqn_best.pt")
        if os.path.exists(ckpt_path):
            agent.load(ckpt_path)
            print(f"Resumed from {ckpt_path} (steps_done={agent.steps_done})")

    print(f"Device: {agent.device}")
    print(f"Training DQN on {ENV_ID} for {TOTAL_FRAMES} frames...")
    print(f"Actions: {env.action_space.n}, Obs: {env.observation_space.shape}\n")

    reward_history = []
    episode_reward = 0.0
    episode_count = 0
    best_avg_reward = -float("inf")

    state, _ = env.reset()
    start_time = time.time()
    total_frames = 0

    while total_frames < TOTAL_FRAMES:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.store_transition(state, action, reward, next_state, float(done))
        agent.update()

        state = next_state
        episode_reward += reward
        total_frames += 1

        if done:
            reward_history.append(episode_reward)
            episode_count += 1
            episode_reward = 0.0
            state, _ = env.reset()

            if episode_count % PRINT_EVERY == 0:
                avg_reward = np.mean(reward_history[-50:])
                elapsed = time.time() - start_time
                fps = total_frames / elapsed
                print(
                    f"Episode {episode_count:4d} | "
                    f"Frames: {total_frames:8d} | "
                    f"Reward: {reward_history[-1]:6.1f} | "
                    f"Avg(50): {avg_reward:6.1f} | "
                    f"Eps: {agent.epsilon:.3f} | "
                    f"FPS: {fps:.0f}"
                )

            # Save best model
            if len(reward_history) >= 50:
                avg_reward = np.mean(reward_history[-50:])
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    agent.save(os.path.join(SAVE_DIR, "dqn_best.pt"))

    # Save final model
    agent.save(os.path.join(SAVE_DIR, "dqn_final.pt"))
    env.close()

    # Plot reward curve
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, alpha=0.3, label="Episode reward")
    if len(reward_history) >= 50:
        smoothed = np.convolve(reward_history, np.ones(50) / 50, mode="valid")
        plt.plot(range(49, len(reward_history)), smoothed, label="Avg(50)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"DQN on {ENV_ID}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_curve.png"), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\nTraining curve saved to training_curve.png")
    print(f"Best avg reward: {best_avg_reward:.1f}")
    print(f"Total time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    import sys
    train(resume="--resume" in sys.argv)
