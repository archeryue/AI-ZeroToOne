import numpy as np
import torch
import torch.nn as nn


class AtariActorCritic(nn.Module):
    """CNN Actor-Critic for Atari games.

    Shared CNN backbone (same as DeepMind DQN), separate actor/critic heads.
    Input:  stacked grayscale frames (4, 84, 84)
    Output: action logits + state value
    """

    def __init__(self, in_channels: int = 4, action_dim: int = 6):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Conv output: 64 * 7 * 7 = 3136
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
        )

        self.actor = nn.Linear(512, action_dim)
        self.critic = nn.Linear(512, 1)

        # Orthogonal initialization (helps PPO stability)
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x: torch.Tensor):
        x = x.float() / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_action_and_value(self, x: torch.Tensor):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value.squeeze(-1)
