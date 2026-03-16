import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import ActorCritic


class RolloutBuffer:
    """Stores a batch of rollout data for PPO updates."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def store(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)


class PPOAgent:
    """PPO Agent with clipped surrogate objective and GAE."""

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 4,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64,
        rollout_steps: int = 2048,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps
        self.action_dim = action_dim

        self.network = ActorCritic(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.buffer = RolloutBuffer()

    def select_action(self, state: np.ndarray, greedy: bool = False):
        """Select action. Returns (action, log_prob, value) for training, or just action if greedy."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if greedy:
                logits, _ = self.network(state_t)
                return logits.argmax(dim=1).item()
            else:
                action, log_prob, value = self.network.get_action_and_value(state_t)
                return action.item(), log_prob.item(), value.item()

    def compute_gae(self, last_value: float):
        """Compute Generalized Advantage Estimation."""
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = self.buffer.values

        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    def update(self, last_value: float) -> dict:
        """Run PPO update. Returns dict of loss metrics."""
        advantages, returns = self.compute_gae(last_value)

        # Convert buffer to tensors
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.LongTensor(self.buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.update_epochs):
            # Random mini-batch indices
            indices = np.random.permutation(len(self.buffer.states))

            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log_probs = old_log_probs[batch_idx]
                b_advantages = advantages_t[batch_idx]
                b_returns = returns_t[batch_idx]

                # Evaluate current policy on batch
                new_log_probs, entropy, new_values = self.network.evaluate_actions(b_states, b_actions)

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(new_values, b_returns)

                # Entropy bonus (encourage exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                num_updates += 1

        self.buffer.clear()

        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def save(self, path: str):
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(checkpoint["network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
