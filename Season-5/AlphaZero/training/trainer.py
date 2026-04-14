"""Network training loop for AlphaZero.

Loss = policy_loss + value_loss + L2_reg
  policy_loss = -pi . log(p)          (cross-entropy with MCTS targets)
  value_loss  = (z - v)^2             (MSE with game outcome)
  L2_reg      = c * ||theta||^2       (via optimizer weight_decay)
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
        """One gradient step. Returns loss components.

        Skipped (no weight update) if the loss is non-finite — we hit a
        NaN at iter 13 of the first run because a single gradient spike
        with no clipping sent the weights to inf. The guard here plus
        `clip_grad_norm_` below prevents a recurrence.

        If `anchor` is given, `anchor_frac` of each batch is drawn from
        it instead of the main buffer. This is a regularizer against
        BatchNorm specializing to the current narrow self-play
        distribution (see iter 4→19 regression handover).
        """
        self.net.train()

        use_anchor = (
            anchor is not None and anchor_frac > 0.0 and len(anchor) > 0)
        if use_anchor:
            n_anchor = int(self.cfg.batch_size * anchor_frac)
            n_main = self.cfg.batch_size - n_anchor
            obs_m, pol_m, val_m = buffer.sample(n_main)
            obs_a, pol_a, val_a = anchor.sample(n_anchor)
            obs_np = np.concatenate([obs_m, obs_a], axis=0)
            policy_np = np.concatenate([pol_m, pol_a], axis=0)
            value_np = np.concatenate([val_m, val_a], axis=0)
        else:
            obs_np, policy_np, value_np = buffer.sample(self.cfg.batch_size)
        # Buffer stores obs as uint8 (Phase 2 memory optimization).
        # Cast to float AFTER the H2D copy so PCIe sees 1 byte/cell
        # instead of 4 — CUDA fuses the dtype promotion into the load.
        obs = torch.from_numpy(obs_np).to(self.device).float()
        target_policy = torch.from_numpy(policy_np).to(self.device)
        target_value = torch.from_numpy(value_np).to(self.device)

        with torch.amp.autocast("cuda", enabled=self.device.type == "cuda"):
            logits, value = self.net(obs)

            # Policy loss: cross-entropy with MCTS visit distribution
            log_probs = F.log_softmax(logits, dim=-1)
            policy_loss = -(target_policy * log_probs).sum(dim=-1).mean()

            # Value loss: MSE
            value_loss = F.mse_loss(value, target_value)

            loss = policy_loss + value_loss

        # NaN / Inf guard — skip the step entirely rather than poisoning
        # the weights. Returns the current (unchanged) values so the
        # epoch average still accumulates something sensible.
        if not torch.isfinite(loss):
            self._nan_skips += 1
            return {
                "loss": float("nan"),
                "policy_loss": float("nan"),
                "value_loss": float("nan"),
                "skipped": True,
            }

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — cap the L2 norm of all parameter grads
        # at 5.0. Without this, occasional SGD spikes can send weights
        # to ∞ in one step (learned the hard way at iter 13).
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.total_steps += 1

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
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
            applied += 1

        denom = max(applied, 1)
        return {
            "loss": total_loss / denom,
            "policy_loss": total_policy / denom,
            "value_loss": total_value / denom,
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
