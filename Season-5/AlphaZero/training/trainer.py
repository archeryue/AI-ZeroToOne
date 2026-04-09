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

    def get_lr(self, total_iterations: int) -> float:
        """Cosine decay from lr_init to lr_final."""
        max_steps = total_iterations * self.cfg.train_steps_per_iter
        if max_steps == 0:
            return self.cfg.lr_init
        progress = min(self.total_steps / max_steps, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.cfg.lr_final + (self.cfg.lr_init - self.cfg.lr_final) * cosine

    def train_step(self, buffer: ReplayBuffer) -> dict:
        """One gradient step. Returns loss components."""
        self.net.train()

        obs_np, policy_np, value_np = buffer.sample(self.cfg.batch_size)
        obs = torch.from_numpy(obs_np).to(self.device)
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

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.total_steps += 1

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }

    def train_epoch(self, buffer: ReplayBuffer, total_iterations: int) -> dict:
        """Run train_steps_per_iter gradient steps. Returns average losses."""
        lr = self.get_lr(total_iterations)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        steps = self.cfg.train_steps_per_iter

        for _ in range(steps):
            stats = self.train_step(buffer)
            total_loss += stats["loss"]
            total_policy += stats["policy_loss"]
            total_value += stats["value_loss"]

        return {
            "loss": total_loss / steps,
            "policy_loss": total_policy / steps,
            "value_loss": total_value / steps,
            "lr": lr,
            "total_steps": self.total_steps,
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
