"""Evaluate a trained DQN agent and record videos at different training stages.

Usage:
    # Record videos from a saved model
    python evaluate.py --model dqn_best.pt --episodes 3

    # Record a random agent (before training) for comparison
    python evaluate.py --random --episodes 1
"""

import argparse
import os

import gymnasium as gym
import numpy as np

from agent import DQNAgent

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(SAVE_DIR, "videos")


def evaluate(model_path: str | None, num_episodes: int = 3, record: bool = True):
    """Run the agent and optionally record videos."""
    render_mode = "rgb_array" if record else "human"
    env = gym.make("LunarLander-v3", render_mode=render_mode)

    if record:
        video_name = "random" if model_path is None else os.path.splitext(os.path.basename(model_path))[0]
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=VIDEO_DIR,
            name_prefix=video_name,
            episode_trigger=lambda ep: True,  # record every episode
        )

    agent = DQNAgent()
    if model_path is not None:
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print("Running random agent (no model loaded)")

    rewards = []
    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            if model_path is None:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, greedy=True)

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)
        print(f"  Episode {ep + 1}: reward = {total_reward:.1f}")

    env.close()
    print(f"\nAvg reward over {num_episodes} episodes: {np.mean(rewards):.1f}")

    if record:
        print(f"Videos saved to {VIDEO_DIR}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate DQN on LunarLander-v3")
    parser.add_argument("--model", type=str, default=None, help="Path to saved model (.pt)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--random", action="store_true", help="Run random agent (no model)")
    parser.add_argument("--no-record", action="store_true", help="Disable video recording")
    args = parser.parse_args()

    if args.random:
        model_path = None
    elif args.model:
        model_path = args.model
    else:
        # Default: try to load best model
        default_path = os.path.join(SAVE_DIR, "dqn_best.pt")
        if os.path.exists(default_path):
            model_path = default_path
        else:
            print("No model found. Use --model <path> or --random")
            return

    evaluate(model_path, num_episodes=args.episodes, record=not args.no_record)


if __name__ == "__main__":
    main()
