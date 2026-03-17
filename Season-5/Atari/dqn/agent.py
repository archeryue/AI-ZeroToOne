import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import AtariDQN
from replay_buffer import ReplayBuffer


class DQNAgent:
    """DQN Agent for Atari games with CNN feature extraction."""

    def __init__(
        self,
        action_dim: int = 6,
        in_channels: int = 4,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.02,
        epsilon_decay_steps: int = 100000,
        buffer_capacity: int = 100000,
        batch_size: int = 32,
        target_update_freq: int = 1000,
        learning_starts: int = 10000,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learning_starts = learning_starts

        # Epsilon schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.steps_done = 0

        # Networks
        self.q_net = AtariDQN(in_channels, action_dim).to(self.device)
        self.target_net = AtariDQN(in_channels, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss, more stable for Atari

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_capacity)

    @property
    def epsilon(self) -> float:
        fraction = min(1.0, self.steps_done / self.epsilon_decay_steps)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> float | None:
        if len(self.buffer) < self.learning_starts:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(rewards).unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).unsqueeze(1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.steps_done += 1

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path: str):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint["steps_done"]
