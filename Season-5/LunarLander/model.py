import torch
import torch.nn as nn


class DQN(nn.Module):
    """Simple feedforward Q-network for LunarLander.

    Input:  state vector (8 dimensions)
    Output: Q-values for each action (4 actions)
    """

    def __init__(self, state_dim: int = 8, action_dim: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
