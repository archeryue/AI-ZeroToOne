import torch
import torch.nn as nn


class AtariDQN(nn.Module):
    """CNN-based DQN for Atari games.

    Input:  stacked grayscale frames (4, 84, 84)
    Output: Q-values for each action
    Architecture follows the original DeepMind DQN paper.
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

        # Conv output size: 64 * 7 * 7 = 3136 (for 84x84 input)
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize pixel values to [0, 1]
        x = x.float() / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
