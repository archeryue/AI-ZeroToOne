"""Evaluate a trained agent (DQN or PPO) on Atari and record videos.

Usage:
    python evaluate.py --algo dqn --episodes 3
    python evaluate.py --algo ppo --episodes 3
    python evaluate.py --random --episodes 1
"""

import argparse
import os
import sys

import ale_py
import gymnasium as gym
import numpy as np

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(SAVE_DIR, "videos")
ENV_ID = "ALE/Pong-v5"


def load_agent(algo: str, model_path: str, action_dim: int):
    if algo == "ppo":
        sys.path.insert(0, os.path.join(SAVE_DIR, "ppo"))
        from agent import PPOAgent
        agent = PPOAgent(action_dim=action_dim)
    else:
        sys.path.insert(0, os.path.join(SAVE_DIR, "dqn"))
        from agent import DQNAgent
        agent = DQNAgent(action_dim=action_dim)
    agent.load(model_path)
    return agent


def evaluate(model_path: str | None, algo: str = "dqn", num_episodes: int = 3, record: bool = True):
    """Run the agent on raw Atari env (no episodic life, no reward clipping) for proper evaluation."""
    render_mode = "rgb_array" if record else "human"
    env = gym.make(ENV_ID, render_mode=render_mode, frameskip=1)

    if record:
        if model_path is None:
            video_subdir = os.path.join(VIDEO_DIR, "random")
            video_name = "random"
        else:
            video_subdir = os.path.join(VIDEO_DIR, algo)
            video_name = algo
        os.makedirs(video_subdir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_subdir,
            name_prefix=video_name,
            episode_trigger=lambda ep: True,
        )

    # Apply preprocessing wrappers after RecordVideo (so video captures raw frames)
    sys.path.insert(0, os.path.join(SAVE_DIR, "dqn"))
    from wrappers import NoopResetEnv, MaxAndSkipEnv, FireResetEnv, WarpFrame, FrameStack

    env = NoopResetEnv(env)
    env = MaxAndSkipEnv(env, skip=4)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = FrameStack(env, k=4)

    agent = None
    if model_path is not None:
        agent = load_agent(algo, model_path, env.action_space.n)
        print(f"Loaded {algo.upper()} model from {model_path}")
    else:
        print("Running random agent")

    rewards = []
    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            if agent is None:
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
        print(f"Videos saved to {video_subdir}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate agent on Atari")
    parser.add_argument("--model", type=str, default=None, help="Path to saved model (.pt)")
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "ppo"], help="Algorithm")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--random", action="store_true", help="Run random agent")
    parser.add_argument("--no-record", action="store_true", help="Disable video recording")
    args = parser.parse_args()

    if args.random:
        model_path = None
    elif args.model:
        model_path = args.model
    else:
        default_path = os.path.join(SAVE_DIR, args.algo, f"{args.algo}_best.pt")
        if os.path.exists(default_path):
            model_path = default_path
        else:
            print(f"No model found at {default_path}. Use --model <path> or --random")
            return

    evaluate(model_path, algo=args.algo, num_episodes=args.episodes, record=not args.no_record)


if __name__ == "__main__":
    main()
