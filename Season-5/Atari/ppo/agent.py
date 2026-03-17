import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import AtariActorCritic


class PPOAgent:
    """PPO Agent for Atari with vectorized environment support."""

    def __init__(
        self,
        action_dim: int = 6,
        in_channels: int = 4,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.1,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        num_minibatches: int = 4,
        rollout_steps: int = 128,
        num_envs: int = 8,
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
        self.num_minibatches = num_minibatches
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.action_dim = action_dim

        self.batch_size = num_envs * rollout_steps
        self.minibatch_size = self.batch_size // num_minibatches

        self.network = AtariActorCritic(in_channels, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def select_action(self, state: np.ndarray, greedy: bool = False):
        """Select action for a single state (used in evaluation)."""
        state_t = torch.from_numpy(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if greedy:
                logits, _ = self.network(state_t)
                return logits.argmax(dim=1).item()
            else:
                action, log_prob, value = self.network.get_action_and_value(state_t)
                return action.item(), log_prob.item(), value.item()

    def select_actions_batch(self, states: torch.Tensor):
        """Select actions for a batch of states from vectorized envs."""
        with torch.no_grad():
            actions, log_probs, values = self.network.get_action_and_value(states)
        return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()

    def compute_gae(self, rewards, dones, values, next_value):
        """Compute GAE for vectorized rollout. All inputs shape: (steps, num_envs)."""
        advantages = np.zeros_like(rewards)
        last_gae = np.zeros(self.num_envs)

        for t in reversed(range(self.rollout_steps)):
            if t == self.rollout_steps - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, states, actions, log_probs, advantages, returns) -> dict:
        """Run PPO update on flattened rollout data."""
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        old_log_probs = torch.from_numpy(log_probs).to(self.device)
        advantages_t = torch.from_numpy(advantages).float().to(self.device)
        returns_t = torch.from_numpy(returns).float().to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.update_epochs):
            indices = np.random.permutation(self.batch_size)

            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                batch_idx = indices[start:end]

                new_log_probs, entropy, new_values = self.network.evaluate_actions(
                    states[batch_idx], actions[batch_idx]
                )

                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])
                surr1 = ratio * advantages_t[batch_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_t[batch_idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(new_values, returns_t[batch_idx])

                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                num_updates += 1

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
