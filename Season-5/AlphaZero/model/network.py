"""AlphaZero ResNet dual-head network (policy + value).

Architecture follows AlphaGo Zero paper:
  Input → Conv3x3+BN+ReLU → N×ResBlock → PolicyHead + ValueHead

Policy head: Conv1x1(ch→2) + BN + ReLU + FC → softmax over actions
Value head:  Conv1x1(ch→1) + BN + ReLU + FC(→256) + ReLU + FC(→1) + tanh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class AlphaZeroNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        N = config.board_size
        ch = config.channels

        # Input block
        self.input_conv = nn.Conv2d(config.input_planes, ch, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(ch)

        # Residual tower
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(ch) for _ in range(config.num_blocks)]
        )

        # Policy head: Conv1x1(ch→2) + BN + ReLU + FC(2*N*N → N*N+1)
        self.policy_conv = nn.Conv2d(ch, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * N * N, config.actions)

        # Value head: Conv1x1(ch→1) + BN + ReLU + FC(N*N → 256) + ReLU + FC(256 → 1) + tanh
        self.value_conv = nn.Conv2d(ch, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(N * N, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Input block
        out = F.relu(self.input_bn(self.input_conv(x)))

        # Residual tower
        out = self.residual_tower(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        # Return log-softmax for cross-entropy loss, raw logits for MCTS
        policy_logits = p

        # Value head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        value = v.squeeze(-1)

        return policy_logits, value

    def predict(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference for MCTS. Returns (policy_probs, value)."""
        with torch.no_grad():
            logits, value = self(obs)
            policy = F.softmax(logits, dim=-1)
        return policy, value

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
