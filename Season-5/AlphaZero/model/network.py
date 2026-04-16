"""AlphaZero ResNet (policy + value + ownership auxiliary).

Standard AlphaGo Zero / KataGo architecture:

  Input → Conv3x3+LN+ReLU → N×ResBlock → PolicyHead + ValueHead + OwnershipHead

Policy head:    Conv1x1(ch→2) + LN + ReLU + FC → softmax over actions
Value head:     Conv1x1(ch→1) + LN + ReLU + FC(N*N→256) + ReLU + FC(256→1) + tanh
Ownership head: Conv1x1(ch→1) → (N, N) logits  (BCE-with-logits in trainer)

Uses LayerNorm everywhere instead of BatchNorm. BatchNorm's running
statistics caused recurring train/eval distribution drift throughout
Phase 1 (Problem 3: BN specialization) and Phase 2 (strength drift
candidate). LayerNorm normalizes per-sample with no running stats,
so train and eval behavior are identical.

Ownership head is a KataGo-style auxiliary that provides dense per-cell
supervision to regularize the trunk. Default ownership_loss_weight=0.0
preserves the 9x9 recipe; 13x13 preset turns it on.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, board_size: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.ln1 = nn.LayerNorm([channels, board_size, board_size])
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.ln2 = nn.LayerNorm([channels, board_size, board_size])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))
        return F.relu(out + residual)


class AlphaZeroNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        N = config.board_size
        ch = config.channels

        # Input block
        self.input_conv = nn.Conv2d(config.input_planes, ch, 3, padding=1, bias=False)
        self.input_ln = nn.LayerNorm([ch, N, N])

        # Residual tower
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(ch, N) for _ in range(config.num_blocks)]
        )

        # Policy head: Conv1x1(ch→2) + LN + ReLU + FC(2*N*N → N*N+1)
        self.policy_conv = nn.Conv2d(ch, 2, 1, bias=False)
        self.policy_ln = nn.LayerNorm([2, N, N])
        self.policy_fc = nn.Linear(2 * N * N, config.actions)

        # Value head: Conv1x1(ch→1) + LN + ReLU + FC(N*N→256) + ReLU + FC(256→1) + tanh
        # Standard AlphaGo Zero architecture.
        self.value_conv = nn.Conv2d(ch, 1, 1, bias=False)
        self.value_ln = nn.LayerNorm([1, N, N])
        self.value_fc1 = nn.Linear(N * N, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Ownership head: Conv1x1(ch→1) → (B, 1, N, N) logits.
        # KataGo-style auxiliary — dense per-cell supervision regularizes
        # the trunk. No normalization, no nonlinearity — output is fed
        # straight to binary_cross_entropy_with_logits in the trainer.
        self.ownership_conv = nn.Conv2d(ch, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Input block
        out = F.relu(self.input_ln(self.input_conv(x)))

        # Residual tower
        out = self.residual_tower(out)

        # Policy head
        p = F.relu(self.policy_ln(self.policy_conv(out)))
        p = p.flatten(1)
        p = self.policy_fc(p)
        policy_logits = p

        # Value head — standard MLP
        v = F.relu(self.value_ln(self.value_conv(out)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)

        # Ownership head — per-cell logit, (B, N, N)
        own_logits = self.ownership_conv(out).squeeze(1)

        return policy_logits, v, own_logits

    def predict(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference for MCTS. Returns (policy_probs, value)."""
        with torch.no_grad():
            logits, value, _own = self(obs)
            policy = F.softmax(logits, dim=-1)
        return policy, value

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
