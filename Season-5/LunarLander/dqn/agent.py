import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import DQN
from replay_buffer import ReplayBuffer


class DQNAgent:
    """DQN Agent with epsilon-greedy exploration and target network."""

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 4,
        hidden_dim: int = 128,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 10000,
        buffer_capacity: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
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

        # Epsilon schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        # Networks
        self.q_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_capacity)

    @property
    def epsilon(self) -> float:
        """Current epsilon value (linearly decayed)."""
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * max(
            0, 1 - self.steps_done / self.epsilon_decay
        )

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Select action using epsilon-greedy policy."""
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> float | None:
        """Perform one gradient step. Returns loss value, or None if buffer too small."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q values: Q(s, a)
        q_values = self.q_net(states).gather(1, actions)

        # Target Q values: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update step counter (for epsilon decay)
        self.steps_done += 1

        # Sync target network periodically
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
