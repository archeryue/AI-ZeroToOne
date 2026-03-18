"""PPO Agent for Chinese Chess self-play.

Key differences from Atari PPO:
- Action masking: illegal actions masked to -inf before softmax
- Self-play: same network plays both Red and Black
- Variable-length games: GAE computed per-game per-side, not fixed rollout
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.model import ChessActorCritic


class PPOAgent:
    """PPO Agent with action masking for Chinese Chess."""

    def __init__(
        self,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        num_minibatches: int = 4,
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

        self.network = ChessActorCritic().to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

    def select_action(self, obs: np.ndarray, action_mask: np.ndarray):
        """Select action for a single state with action masking."""
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(action_mask).unsqueeze(0).bool().to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.network.get_action_and_value(obs_t, mask_t)
        return action.item(), log_prob.item(), value.item()

    def select_action_greedy(self, obs: np.ndarray, action_mask: np.ndarray) -> int:
        """Select best action greedily (for evaluation)."""
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(action_mask).unsqueeze(0).bool().to(self.device)
        with torch.no_grad():
            logits, _ = self.network(obs_t, mask_t)
        return logits.argmax(dim=1).item()

    @staticmethod
    def compute_gae(rewards, values, dones, gamma, gae_lambda):
        """Compute GAE for a single trajectory (one side of one game).

        Each trajectory ends with done=True (win, loss, draw, or truncation).
        Terminal states have next_value=0.
        """
        T = len(rewards)
        values_arr = np.array(values, dtype=np.float32)
        rewards_arr = np.array(rewards, dtype=np.float32)
        dones_arr = np.array(dones, dtype=np.float32)

        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(T)):
            if t == T - 1 or dones_arr[t]:
                next_value = 0.0
            else:
                next_value = values_arr[t + 1]

            not_done = 1.0 - dones_arr[t]
            delta = rewards_arr[t] + gamma * next_value - values_arr[t]
            last_gae = delta + gamma * gae_lambda * not_done * last_gae
            advantages[t] = last_gae

        returns = advantages + values_arr
        return advantages, returns

    def update(self, obs, actions, action_masks, log_probs, advantages, returns) -> dict:
        """Run PPO update on collected self-play data."""
        obs_t = torch.from_numpy(obs).to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        masks_t = torch.from_numpy(action_masks).bool().to(self.device)
        old_log_probs_t = torch.from_numpy(log_probs).to(self.device)
        advantages_t = torch.from_numpy(advantages).float().to(self.device)
        returns_t = torch.from_numpy(returns).float().to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        batch_size = len(obs)
        minibatch_size = max(batch_size // self.num_minibatches, 1)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(self.update_epochs):
            indices = np.random.permutation(batch_size)

            for start in range(0, batch_size, minibatch_size):
                end = min(start + minibatch_size, batch_size)
                idx = indices[start:end]

                new_log_probs, entropy, new_values = self.network.evaluate_actions(
                    obs_t[idx], actions_t[idx], masks_t[idx]
                )

                ratio = torch.exp(new_log_probs - old_log_probs_t[idx])
                surr1 = ratio * advantages_t[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(new_values, returns_t[idx])
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
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
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
