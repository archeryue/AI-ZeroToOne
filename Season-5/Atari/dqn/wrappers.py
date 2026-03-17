"""Atari preprocessing wrappers following DeepMind's DQN setup."""

import ale_py
import gymnasium as gym
import numpy as np


class NoopResetEnv(gym.Wrapper):
    """Execute random number of no-ops on reset to add stochasticity."""

    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class FireResetEnv(gym.Wrapper):
    """Press FIRE on reset (required for some Atari games like Pong)."""

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)  # FIRE
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    """Treat loss of life as end of episode (but only reset on true game over)."""

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = info.get("lives", 0)
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, _, _, info = self.env.step(0)  # no-op to advance past life loss
        self.lives = info.get("lives", 0)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """Return max of last 2 frames over skip frames (frame skipping)."""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info


class WarpFrame(gym.ObservationWrapper):
    """Resize frame to 84x84 grayscale."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )

    def observation(self, obs):
        import cv2
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized


class FrameStack(gym.Wrapper):
    """Stack last k frames as channels. Output shape: (k, 84, 84)."""

    def __init__(self, env, k=4):
        super().__init__(env)
        self.k = k
        self.frames = np.zeros((k, 84, 84), dtype=np.uint8)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(k, 84, 84), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for i in range(self.k):
            self.frames[i] = obs
        return self.frames.copy(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames = np.roll(self.frames, shift=-1, axis=0)
        self.frames[-1] = obs
        return self.frames.copy(), reward, terminated, truncated, info


class ClipRewardEnv(gym.RewardWrapper):
    """Clip reward to {-1, 0, +1}."""

    def reward(self, reward):
        return float(np.sign(reward))


def make_atari_env(env_id: str = "ALE/Pong-v5", render_mode=None):
    """Create a fully wrapped Atari environment."""
    env = gym.make(env_id, render_mode=render_mode, frameskip=1)
    env = NoopResetEnv(env)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = FireResetEnv(env)
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    env = FrameStack(env, k=4)
    return env
