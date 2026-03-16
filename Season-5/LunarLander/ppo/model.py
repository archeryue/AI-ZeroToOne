import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO.

    Actor:  state -> action probabilities (policy)
    Critic: state -> state value V(s)
    """

    def __init__(self, state_dim: int = 8, action_dim: int = 4, hidden_dim: int = 128):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head: outputs action logits
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic head: outputs state value
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action_and_value(self, x: torch.Tensor):
        """Sample an action and return (action, log_prob, value)."""
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor):
        """Evaluate given actions: return (log_probs, entropy, values)."""
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value.squeeze(-1)
