"""Network training loop for AlphaZero.

Loss = policy_loss + score_loss_weight * score_loss
                  + ownership_loss_weight * ownership_loss + L2_reg
  policy_loss    = -pi . log(p)                   (cross-entropy)
  score_loss     = (score_target - score_pred)^2   (MSE with territory margin)
  ownership_loss = mean BCE-with-logits per cell   (KataGo-style aux target)
  L2_reg         = c * ||theta||^2                 (via optimizer weight_decay)

No direct value loss — value is derived from score via
  value = tanh(scale * score_pred + bias)
The derived value is used by MCTS but not directly supervised.
Score supervision is denser than binary ±1: the target is the actual
territory margin from current player's perspective.
"""

import math
import torch
import torch.nn.functional as F
import numpy as np

from model.config import TrainingConfig
from model.network import AlphaZeroNet
from training.replay_buffer import ReplayBuffer


class Trainer:
    def __init__(self, net: AlphaZeroNet, train_cfg: TrainingConfig,
                 device: torch.device):
        self.net = net
        self.cfg = train_cfg
        self.device = device

        self.optimizer = torch.optim.SGD(
            net.parameters(),
            lr=train_cfg.lr_init,
            momentum=train_cfg.momentum,
            weight_decay=train_cfg.weight_decay,
        )

        self.total_steps = 0
        self._nan_skips = 0  # running count of skipped-non-finite steps

    def get_lr(self, total_iterations: int) -> float:
        """Cosine decay from lr_init to lr_final."""
        max_steps = total_iterations * self.cfg.train_steps_per_iter
        if max_steps == 0:
            return self.cfg.lr_init
        progress = min(self.total_steps / max_steps, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.cfg.lr_final + (self.cfg.lr_init - self.cfg.lr_final) * cosine

    def train_step(self, buffer: ReplayBuffer,
                   anchor: ReplayBuffer = None,
                   anchor_frac: float = 0.0) -> dict:
        """One gradient step. Returns loss components."""
        self.net.train()

        use_anchor = (
            anchor is not None and anchor_frac > 0.0 and len(anchor) > 0)
        if use_anchor:
            n_anchor = int(self.cfg.batch_size * anchor_frac)
            n_main = self.cfg.batch_size - n_anchor
            obs_m, pol_m, val_m, own_m = buffer.sample(n_main)
            obs_a, pol_a, val_a, own_a = anchor.sample(n_anchor)
            obs_np = np.concatenate([obs_m, obs_a], axis=0)
            policy_np = np.concatenate([pol_m, pol_a], axis=0)
            value_np = np.concatenate([val_m, val_a], axis=0)
            ownership_np = np.concatenate([own_m, own_a], axis=0)
        else:
            obs_np, policy_np, value_np, ownership_np = buffer.sample(
                self.cfg.batch_size)

        obs = torch.from_numpy(obs_np).to(self.device).float()
        target_policy = torch.from_numpy(policy_np).to(self.device)
        # Ownership: int8 in {-1, 0, 1} from current player's perspective.
        target_ownership = torch.from_numpy(ownership_np).to(
            self.device).float()
        # BCE target in [0, 1]. Dame cells (0) → 0.5.
        target_ownership_01 = (target_ownership + 1.0) / 2.0
        # Score target: territory margin normalized by board side length.
        # Raw sum is ±50 typical on 13x13; dividing by N gives std≈1.5,
        # near unit scale for stable MSE gradients with weight=1.0.
        N = target_ownership.shape[1]  # board side length
        target_score = target_ownership.sum(dim=(1, 2)) / N  # (B,)

        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            logits, value, ownership_logits, score_pred = self.net(obs)

            # Policy loss: cross-entropy with MCTS visit distribution
            log_probs = F.log_softmax(logits, dim=-1)
            policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()

            # Score loss: MSE between predicted and actual score margin.
            score_w = getattr(self.cfg, "score_loss_weight", 0.0)
            if score_w > 0.0:
                score_loss = F.mse_loss(score_pred, target_score)
            else:
                score_loss = torch.zeros((), device=self.device)

            # Value loss: MSE with game outcome (reported for monitoring;
            # value is derived from score, typically vlw=0).
            target_value = torch.from_numpy(value_np).to(self.device)
            value_loss = F.mse_loss(value, target_value)

            # Ownership loss: per-cell BCE-with-logits.
            own_w = getattr(self.cfg, "ownership_loss_weight", 0.0)
            if own_w > 0.0:
                ownership_loss = F.binary_cross_entropy_with_logits(
                    ownership_logits, target_ownership_01)
            else:
                ownership_loss = torch.zeros((), device=self.device)

            # Score bias regularization: penalize deviation of batch
            # mean prediction from batch mean target. Prevents the
            # score FC bias from oscillating across iters.
            sbr_w = getattr(self.cfg, "score_bias_reg_weight", 0.0)
            if sbr_w > 0.0 and score_w > 0.0:
                score_bias_penalty = (score_pred.mean() - target_score.mean()).pow(2)
            else:
                score_bias_penalty = torch.zeros((), device=self.device)

            loss = (policy_loss
                    + self.cfg.value_loss_weight * value_loss
                    + score_w * score_loss
                    + sbr_w * score_bias_penalty
                    + own_w * ownership_loss)

        # NaN / Inf guard
        if not torch.isfinite(loss):
            self._nan_skips += 1
            return {
                "loss": float("nan"),
                "policy_loss": float("nan"),
                "value_loss": float("nan"),
                "score_loss": float("nan"),
                "ownership_loss": float("nan"),
                "skipped": True,
            }

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.total_steps += 1

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "score_loss": float(score_loss.item()),
            "ownership_loss": float(ownership_loss.item()),
            "skipped": False,
        }

    def train_epoch(self, buffer: ReplayBuffer, total_iterations: int,
                    anchor: ReplayBuffer = None,
                    anchor_frac: float = 0.0) -> dict:
        """Run train_steps_per_iter gradient steps. Returns average losses."""
        lr = self.get_lr(total_iterations)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_score = 0.0
        total_ownership = 0.0
        steps = self.cfg.train_steps_per_iter
        applied = 0
        skipped_this_epoch = 0

        for _ in range(steps):
            stats = self.train_step(
                buffer, anchor=anchor, anchor_frac=anchor_frac)
            if stats.get("skipped"):
                skipped_this_epoch += 1
                continue
            total_loss += stats["loss"]
            total_policy += stats["policy_loss"]
            total_value += stats["value_loss"]
            total_score += stats["score_loss"]
            total_ownership += stats["ownership_loss"]
            applied += 1

        denom = max(applied, 1)
        return {
            "loss": total_loss / denom,
            "policy_loss": total_policy / denom,
            "value_loss": total_value / denom,
            "score_loss": total_score / denom,
            "ownership_loss": total_ownership / denom,
            "lr": lr,
            "total_steps": self.total_steps,
            "skipped": skipped_this_epoch,
            "skipped_total": self._nan_skips,
        }

    def save_checkpoint(self, path: str, iteration: int, extra: dict = None):
        state = {
            "iteration": iteration,
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }
        if extra:
            state.update(extra)
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> dict:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.total_steps = state.get("total_steps", 0)
        return state
