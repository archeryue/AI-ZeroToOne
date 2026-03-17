"""Train PPO on Atari Pong with vectorized environments."""

import os
import sys
import time

import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from agent import PPOAgent

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dqn"))
from wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, WarpFrame, ClipRewardEnv, FrameStack

# --------------- Hyperparameters ---------------
ENV_ID = "ALE/Pong-v5"
TOTAL_FRAMES = 10000000
NUM_ENVS = 8
PRINT_EVERY = 10
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
# -----------------------------------------------


def make_single_env(env_id: str, seed: int):
    """Create a wrapped Atari env (for use in vectorized env)."""
    def _init():
        env = gym.make(env_id, frameskip=1)
        env = NoopResetEnv(env)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        env = FireResetEnv(env)
        env = WarpFrame(env)
        env = ClipRewardEnv(env)
        env = FrameStack(env, k=4)
        env.reset(seed=seed)
        return env
    return _init


def train():
    # Create vectorized environments
    envs = gym.vector.AsyncVectorEnv(
        [make_single_env(ENV_ID, seed=i) for i in range(NUM_ENVS)]
    )

    agent = PPOAgent(
        action_dim=envs.single_action_space.n,
        num_envs=NUM_ENVS,
    )

    print(f"Device: {agent.device}")
    print(f"Training PPO on {ENV_ID} for {TOTAL_FRAMES} frames ({NUM_ENVS} parallel envs)...")
    print(f"Rollout: {agent.rollout_steps} steps x {NUM_ENVS} envs = {agent.batch_size} batch\n")

    # Rollout storage
    obs_shape = (4, 84, 84)
    all_states = np.zeros((agent.rollout_steps, NUM_ENVS, *obs_shape), dtype=np.uint8)
    all_actions = np.zeros((agent.rollout_steps, NUM_ENVS), dtype=np.int64)
    all_log_probs = np.zeros((agent.rollout_steps, NUM_ENVS), dtype=np.float32)
    all_rewards = np.zeros((agent.rollout_steps, NUM_ENVS), dtype=np.float32)
    all_dones = np.zeros((agent.rollout_steps, NUM_ENVS), dtype=np.float32)
    all_values = np.zeros((agent.rollout_steps, NUM_ENVS), dtype=np.float32)

    # Tracking
    episode_rewards = []
    episode_count = 0
    running_rewards = np.zeros(NUM_ENVS)
    best_avg_reward = -float("inf")

    states, _ = envs.reset()
    start_time = time.time()
    total_frames = 0
    num_updates = 0

    while total_frames < TOTAL_FRAMES:
        # Collect rollout
        for step in range(agent.rollout_steps):
            states_t = torch.from_numpy(states).to(agent.device)
            actions, log_probs, values = agent.select_actions_batch(states_t)

            next_states, rewards, terminateds, truncateds, infos = envs.step(actions)
            dones = np.logical_or(terminateds, truncateds).astype(np.float32)

            all_states[step] = states
            all_actions[step] = actions
            all_log_probs[step] = log_probs
            all_rewards[step] = rewards
            all_dones[step] = dones
            all_values[step] = values

            running_rewards += rewards
            for i in range(NUM_ENVS):
                if dones[i]:
                    episode_rewards.append(running_rewards[i])
                    episode_count += 1
                    running_rewards[i] = 0.0

            states = next_states
            total_frames += NUM_ENVS

        # Bootstrap value for GAE
        with torch.no_grad():
            _, next_value = agent.network(torch.from_numpy(states).to(agent.device))
            next_value = next_value.squeeze(-1).cpu().numpy()

        # Compute GAE
        advantages, returns = agent.compute_gae(all_rewards, all_dones, all_values, next_value)

        # Flatten (steps, envs, ...) -> (steps*envs, ...)
        flat_states = all_states.reshape(-1, *obs_shape)
        flat_actions = all_actions.reshape(-1)
        flat_log_probs = all_log_probs.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)

        # PPO update
        metrics = agent.update(flat_states, flat_actions, flat_log_probs, flat_advantages, flat_returns)
        num_updates += 1

        # Print progress
        if num_updates % PRINT_EVERY == 0 and len(episode_rewards) > 0:
            avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
            elapsed = time.time() - start_time
            fps = total_frames / elapsed
            print(
                f"Update {num_updates:4d} | "
                f"Frames: {total_frames:8d} | "
                f"Episodes: {episode_count:4d} | "
                f"Avg(50): {avg_reward:6.1f} | "
                f"FPS: {fps:.0f}"
            )

        # Save best model
        if len(episode_rewards) >= 50:
            avg_reward = np.mean(episode_rewards[-50:])
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(os.path.join(SAVE_DIR, "ppo_best.pt"))

    # Save final model
    agent.save(os.path.join(SAVE_DIR, "ppo_final.pt"))
    envs.close()

    # Plot reward curve
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.3, label="Episode reward")
    if len(episode_rewards) >= 50:
        smoothed = np.convolve(episode_rewards, np.ones(50) / 50, mode="valid")
        plt.plot(range(49, len(episode_rewards)), smoothed, label="Avg(50)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"PPO on {ENV_ID}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_curve.png"), dpi=150)
    plt.close()

    elapsed = time.time() - start_time
    print(f"\nTraining curve saved to training_curve.png")
    print(f"Best avg reward: {best_avg_reward:.1f}")
    print(f"Total episodes: {episode_count}")
    print(f"Total time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    train()
