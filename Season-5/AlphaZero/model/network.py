"""AlphaZero ResNet (policy + score + ownership), KataGo-style.

Architecture:

  Input → Conv3x3+LN+ReLU → N×ResBlock → PolicyHead + ScoreHead + OwnershipHead

Policy head:    Conv1x1(ch→2) + LN + ReLU + FC → softmax over actions
Score head:     Conv1x1(ch→1) + LN + ReLU + FC(N*N→32) + ReLU + FC(32→1)
                Predicts final score margin from current player's perspective.
                ~5.5k params (vs 44k for the old value MLP).
Ownership head: Conv1x1(ch→1) → (N, N) logits  (BCE-with-logits in trainer)

Value is DERIVED from score: value = tanh(score_pred * scale + bias)
where scale and bias are two learnable scalars. This means:
- The value head has ~5.5k + 2 = ~5.5k params total
- It cannot memorize 165k noisy ±1 labels (the failure mode that
  killed every prior 13x13 run with the 44k-param value MLP)
- Score supervision is denser than binary win/loss: the target is
  the actual territory margin, not ±1

The score target is computed from ownership labels already in the
replay buffer: score = sum(ownership_per_cell) from the current
player's perspective. No C++ or buffer changes needed.

Uses LayerNorm everywhere instead of BatchNorm.
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

        # Score head: Conv1x1(ch→1) + LN + ReLU + FC(N*N→32) + ReLU + FC(32→1)
        # Predicts final score margin (territory difference) from
        # current player's perspective. Much smaller than the old
        # value MLP (5.5k vs 44k params) — can't memorize cold data.
        self.score_conv = nn.Conv2d(ch, 1, 1, bias=False)
        self.score_ln = nn.LayerNorm([1, N, N])
        self.score_fc1 = nn.Linear(N * N, 32)
        self.score_fc2 = nn.Linear(32, 1)

        # Value derived from score prediction. Two learnable scalars
        # that map raw score margin → win probability:
        #   value = tanh(value_scale * score_pred + value_bias)
        # Score is raw territory margin (range ±50 typical on 13x13).
        # Scale init 0.05: tanh(0.05 * ±20) ≈ ±0.76 (a 20-point
        # lead). Bias init 0: no prior.
        self.value_scale = nn.Parameter(torch.tensor(0.05))
        self.value_bias = nn.Parameter(torch.tensor(0.0))

        # Ownership head: Conv1x1(ch→1) → (B, 1, N, N) logits.
        # KataGo-style auxiliary — dense per-cell supervision regularizes
        # the trunk. No normalization, no nonlinearity — output is fed
        # straight to binary_cross_entropy_with_logits in the trainer.
        self.ownership_conv = nn.Conv2d(ch, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Input block
        out = F.relu(self.input_ln(self.input_conv(x)))

        # Residual tower
        out = self.residual_tower(out)

        # Policy head
        p = F.relu(self.policy_ln(self.policy_conv(out)))
        p = p.flatten(1)
        p = self.policy_fc(p)
        policy_logits = p

        # Score head — predicts score margin
        s = F.relu(self.score_ln(self.score_conv(out)))
        s = s.flatten(1)
        s = F.relu(self.score_fc1(s))
        score = self.score_fc2(s).squeeze(-1)  # (B,)

        # Derived value from score
        value = torch.tanh(self.value_scale * score + self.value_bias)

        # Ownership head — per-cell logit, (B, N, N)
        own_logits = self.ownership_conv(out).squeeze(1)

        return policy_logits, value, own_logits, score

    def predict(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference for MCTS. Returns (policy_probs, value)."""
        with torch.no_grad():
            logits, value, _own, _score = self(obs)
            policy = F.softmax(logits, dim=-1)
        return policy, value

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
