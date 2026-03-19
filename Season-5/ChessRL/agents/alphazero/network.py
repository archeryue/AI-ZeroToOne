"""AlphaZero-style ResNet policy-value network for Chinese Chess.

Input:  (batch, 15, 10, 9) float32 observation
Output: policy logits (batch, 8100) + value scalar (batch,)

Architecture (Mini version for Candidate 4):
- Initial conv: 15 -> 64 channels, k=3, pad=1, BN, ReLU
- 5 Residual Blocks (64 channels each)
- Policy head: Conv1x1(64->2) + BN + ReLU + FC(180->8100)
- Value head:  Conv1x1(64->1) + BN + ReLU + FC(90->256) + ReLU + FC(256->1) + tanh
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block: conv-BN-ReLU-conv-BN + skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class AlphaZeroNet(nn.Module):
    """ResNet dual-head network for AlphaZero."""

    OBS_CHANNELS = 15
    BOARD_H = 10
    BOARD_W = 9
    NUM_ACTIONS = 8100

    def __init__(self, num_blocks: int = 5, channels: int = 64):
        super().__init__()
        self.num_blocks = num_blocks
        self.channels = channels

        # Initial convolution
        self.conv_init = nn.Conv2d(self.OBS_CHANNELS, channels, kernel_size=3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(channels)

        # Residual tower
        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * self.BOARD_H * self.BOARD_W, self.NUM_ACTIONS)

        # Value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(self.BOARD_H * self.BOARD_W, 256)
        self.value_fc2 = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
        # Small init for policy output (encourages exploration early)
        nn.init.zeros_(self.policy_fc.bias)
        nn.init.normal_(self.policy_fc.weight, std=0.01)

    def forward(self, x, action_mask=None):
        """Forward pass.

        Args:
            x: (batch, 15, 10, 9) observation
            action_mask: (batch, 8100) bool, True = legal

        Returns:
            log_policy: (batch, 8100) log-probabilities (masked)
            value: (batch,) in [-1, 1]
        """
        # Shared trunk
        x = F.relu(self.bn_init(self.conv_init(x)))
        x = self.res_blocks(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        if action_mask is not None:
            p = p.masked_fill(~action_mask, float("-inf"))
        log_policy = F.log_softmax(p, dim=-1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return log_policy, v

    def predict(self, obs: np.ndarray, action_mask: np.ndarray, device: torch.device):
        """Single-state inference for MCTS.

        Args:
            obs: (15, 10, 9) numpy array
            action_mask: (8100,) bool numpy array

        Returns:
            policy: (8100,) numpy array of probabilities (masked, sums to 1)
            value: float scalar in [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            x = torch.from_numpy(obs).unsqueeze(0).to(device)
            mask = torch.from_numpy(action_mask).unsqueeze(0).to(device)
            log_p, v = self.forward(x, mask)
            policy = torch.exp(log_p).squeeze(0).cpu().numpy()
            value = v.item()
        return policy, value
