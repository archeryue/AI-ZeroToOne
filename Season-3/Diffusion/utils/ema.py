"""
Exponential Moving Average (EMA) for model parameters.
"""
import torch
import torch.nn as nn
from copy import deepcopy


class EMA:
    """
    Exponential Moving Average of model parameters.

    Maintains a shadow copy of model parameters that is updated with exponential decay.
    This often leads to better sample quality for generative models.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Args:
            model: Model to track
            decay: Decay rate for EMA (closer to 1 = slower update)
        """
        self.model = model
        self.decay = decay
        self.ema_model = deepcopy(model).eval()
        self.ema_model.requires_grad_(False)

        # Copy initial parameters
        self.update(step=0)

    @torch.no_grad()
    def update(self, step: int = None):
        """
        Update EMA parameters.

        Args:
            step: Current training step (for warmup if needed)
        """
        # Optionally use a warmup phase where decay starts lower
        if step is not None and step < 2000:
            decay = min(self.decay, (1 + step) / (10 + step))
        else:
            decay = self.decay

        # Update EMA parameters
        model_params = dict(self.model.named_parameters())
        ema_params = dict(self.ema_model.named_parameters())

        for name in model_params:
            ema_params[name].mul_(decay).add_(
                model_params[name].data,
                alpha=1 - decay
            )

    def state_dict(self):
        """Get EMA model state dict."""
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        """Load EMA model state dict."""
        self.ema_model.load_state_dict(state_dict)
