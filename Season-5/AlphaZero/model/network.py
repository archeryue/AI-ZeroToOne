"""AlphaZero ResNet (policy + ownership + derived value).

Architecture follows AlphaGo Zero paper plus a KataGo-style ownership
head and a *derived* value head added in run4 (see
PHASE_TWO_TRAINING.md):

  Input → Conv3x3+BN+ReLU → N×ResBlock → PolicyHead + OwnershipHead
                                        → (value derived from ownership)

Policy head:    Conv1x1(ch→2) + BN + ReLU + FC → softmax over actions
Ownership head: Conv1x1(ch→1) → (N, N) logits  (no BN, no activation —
                fed directly to BCE-with-logits in the trainer)
Value head:     **NO independent MLP.** Value is computed deterministically
                from the ownership head's predictions:
                    p   = sigmoid(own_logits)   # per-cell P(mine)
                    v   = tanh(k * Σ(2p − 1) + b)
                where (k, b) are two learnable scalars. The full value
                head has 2 parameters instead of the ~44k of the
                original MLP.

Why: run1/run2/run3 and run4-pass1/pass2 offline A/Bs all showed that
the original value MLP (FC 169→256→1 tanh) overfits ~28k noisy cold
labels in under 1 epoch — post_v_mse always jumped from 1.00 (cold
floor) to 1.2–1.6 on held-out positions. The A6 recipe (vlw=0, no
value-head training) stayed at the cold floor, proving the value MLP
itself — not the trunk, not the loss function — was the memorization
culprit.

Tying value to ownership fixes this mechanistically: the 2-parameter
"value head" cannot memorize 28k labels, and its output is now bound
to a quantity (expected net territory, summed from per-cell ownership
probabilities) that the ownership head is supervised on with **169×
the density per position**. The value prediction inherits ownership's
generalization automatically. Value loss gradients still flow back
into the ownership head and the trunk, so value supervision is not
lost — it's just rerouted through a structure that can't overfit.

This is the KataGo "value-from-score" idea in its simplest form. The
scale k and bias b absorb the komi-related offset between "net
territory" and "winrate".

Default `ownership_loss_weight` is 0.0 to preserve the 9x9 recipe;
the 13x13 preset turns it on alongside this architectural change.
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

        # Ownership head: Conv1x1(ch→1) → (B, 1, N, N) logits.
        # No BN, no nonlinearity — output is fed straight to
        # binary_cross_entropy_with_logits in the trainer. Bias=True
        # because there's no BN to absorb a learned offset. This is
        # now the only "real" spatial head on the value-side of the
        # network — the value head below reads off this one.
        self.ownership_conv = nn.Conv2d(ch, 1, 1, bias=True)

        # Derived-value "head" — two learnable scalars. The entire
        # value pipeline is:
        #   own_probs   = sigmoid(ownership_logits)
        #   own_signed  = 2*own_probs - 1           # in [-1, 1]
        #   margin      = own_signed.sum((1,2))     # expected net cells
        #   value       = tanh(value_scale * margin + value_bias)
        # Initialize scale to ~0.02 (so tanh input for a decisive board
        # state of ~±50 net cells lands at ~±1, unsaturated at ±100),
        # and bias to 0. Both are learnable so the network can absorb
        # komi-related offsets and calibration drift during training.
        self.value_scale = nn.Parameter(torch.tensor(0.02))
        self.value_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Input block
        out = F.relu(self.input_bn(self.input_conv(x)))

        # Residual tower
        out = self.residual_tower(out)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.flatten(1)
        p = self.policy_fc(p)
        policy_logits = p

        # Ownership head — per-cell logit, (B, N, N).
        own_logits = self.ownership_conv(out).squeeze(1)

        # Derived value: value is a deterministic function of the
        # ownership head's predictions. No independent value MLP, so
        # nothing to memorize 28k cold labels with. Value gradients
        # still backprop into ownership_conv + trunk via sigmoid/sum/
        # tanh, so value supervision is not discarded — it's rerouted
        # through a structure that can't overfit.
        own_probs = torch.sigmoid(own_logits)
        own_signed = 2.0 * own_probs - 1.0
        margin = own_signed.sum(dim=(1, 2))
        value = torch.tanh(self.value_scale * margin + self.value_bias)

        return policy_logits, value, own_logits

    def predict(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run inference for MCTS. Returns (policy_probs, value).

        Ownership is computed by the forward pass but discarded here —
        MCTS only needs policy and value at inference time.
        """
        with torch.no_grad():
            logits, value, _own = self(obs)
            policy = F.softmax(logits, dim=-1)
        return policy, value

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
