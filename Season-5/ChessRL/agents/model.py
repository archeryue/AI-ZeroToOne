"""CNN Actor-Critic network for Chinese Chess.

Input:  (batch, 15, 10, 9) float32 observation
Output: action logits (batch, 8100) + state value (batch,)

Architecture:
- 3-layer CNN (15→64→128→128), kernel=3, padding=1 (preserves spatial dims)
- Flatten: 128 * 10 * 9 = 11520
- FC: 11520 → 512
- Actor head: 512 → 8100 (masked before softmax)
- Critic head: 512 → 1
"""

import numpy as np
import torch
import torch.nn as nn


class ChessActorCritic(nn.Module):
    """CNN Actor-Critic for Chinese Chess with action masking."""

    OBS_CHANNELS = 15
    BOARD_H = 10
    BOARD_W = 9
    NUM_ACTIONS = 8100

    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(self.OBS_CHANNELS, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        conv_out = 128 * self.BOARD_H * self.BOARD_W  # 11520

        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
        )

        self.actor = nn.Linear(512, self.NUM_ACTIONS)
        self.critic = nn.Linear(512, 1)

        # Orthogonal initialization (helps PPO stability)
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x, action_mask=None):
        """Forward pass.

        Args:
            x: (batch, 15, 10, 9) float32 observation
            action_mask: (batch, 8100) bool tensor, True = legal action
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        features = self.fc(x)

        logits = self.actor(features)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))

        value = self.critic(features)
        return logits, value

    def get_action_and_value(self, x, action_mask=None):
        logits, value = self.forward(x, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)

    def evaluate_actions(self, x, actions, action_mask=None):
        logits, value = self.forward(x, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value.squeeze(-1)
