"""
Flow Matching model for generative modeling.

Implements conditional flow matching as described in:
"Flow Matching for Generative Modeling" (Lipman et al., 2022)
"""
import torch
import torch.nn as nn
from typing import Optional, Callable
from tqdm import tqdm


class FlowMatching(nn.Module):
    """
    Flow Matching model for image generation.

    This model learns to generate images by learning a conditional flow from noise to data.
    The flow is parameterized by a velocity field v_t(x_t) predicted by a neural network.
    """

    def __init__(
        self,
        model: nn.Module,
        sigma_min: float = 1e-4
    ):
        """
        Args:
            model: Neural network that predicts velocity field v_t(x_t)
            sigma_min: Minimum noise level for numerical stability
        """
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min

    def forward(self, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute flow matching loss.

        Args:
            x1: Target data samples (B, C, H, W), normalized to [-1, 1]

        Returns:
            loss: Flow matching loss (scalar)
            info: Dictionary with additional information
        """
        batch_size = x1.shape[0]
        device = x1.device

        # Sample random time t ~ Uniform(0, 1)
        t = torch.rand(batch_size, device=device)

        # Sample noise x0 ~ N(0, I)
        x0 = torch.randn_like(x1)

        # Conditional flow: x_t = t * x1 + (1 - t) * x0 + sigma_min * epsilon
        # This is the optimal transport (OT) conditional flow
        epsilon = torch.randn_like(x1) * self.sigma_min
        x_t = t[:, None, None, None] * x1 + (1 - t[:, None, None, None]) * x0 + epsilon

        # Target velocity: u_t = x1 - x0
        # This is the conditional velocity field for the OT path
        u_t = x1 - x0

        # Predict velocity using the model
        v_t = self.model(x_t, t)

        # Flow matching loss: E[||v_t(x_t) - u_t||^2]
        loss = F.mse_loss(v_t, u_t)

        info = {
            'loss': loss.item(),
            'mean_t': t.mean().item()
        }

        return loss, info

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        image_shape: tuple,
        num_steps: int = 50,
        device: str = 'cuda',
        solver: str = 'euler',
        verbose: bool = True
    ) -> torch.Tensor:
        """
        Generate samples using the learned flow.

        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of images (C, H, W)
            num_steps: Number of ODE solver steps
            device: Device to use
            solver: ODE solver ('euler' or 'midpoint')
            verbose: Whether to show progress bar

        Returns:
            Generated samples (B, C, H, W)
        """
        # Start from noise x_0 ~ N(0, I)
        x = torch.randn(batch_size, *image_shape, device=device)

        # Time points for integration
        dt = 1.0 / num_steps
        timesteps = torch.linspace(0, 1, num_steps + 1, device=device)

        # Integrate the ODE: dx/dt = v_t(x)
        iterator = enumerate(timesteps[:-1])
        if verbose:
            iterator = tqdm(iterator, total=num_steps, desc="Sampling")

        for i, t in iterator:
            t_batch = torch.full((batch_size,), t, device=device)

            if solver == 'euler':
                # Euler method: x_{t+dt} = x_t + dt * v_t(x_t)
                v = self.model(x, t_batch)
                x = x + dt * v

            elif solver == 'midpoint':
                # Midpoint method (RK2): more accurate
                v1 = self.model(x, t_batch)
                x_mid = x + 0.5 * dt * v1
                t_mid = torch.full((batch_size,), t + 0.5 * dt, device=device)
                v2 = self.model(x_mid, t_mid)
                x = x + dt * v2

            else:
                raise ValueError(f"Unknown solver: {solver}")

        return x

    @torch.no_grad()
    def sample_ode(
        self,
        batch_size: int,
        image_shape: tuple,
        num_steps: int = 50,
        device: str = 'cuda',
        verbose: bool = True,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Generate samples using adaptive ODE solver (more accurate but slower).

        Args:
            batch_size: Number of samples to generate
            image_shape: Shape of images (C, H, W)
            num_steps: Number of evaluation points
            device: Device to use
            verbose: Whether to show progress bar
            return_trajectory: Whether to return full trajectory

        Returns:
            Generated samples (B, C, H, W) or trajectory if return_trajectory=True
        """
        # Use torchdiffeq for adaptive ODE solving
        try:
            from torchdiffeq import odeint
        except ImportError:
            print("Warning: torchdiffeq not installed. Falling back to Euler method.")
            return self.sample(batch_size, image_shape, num_steps, device, solver='euler', verbose=verbose)

        # Start from noise
        x0 = torch.randn(batch_size, *image_shape, device=device)

        # Define ODE function
        def ode_func(t, x):
            t_batch = torch.full((batch_size,), t.item(), device=device)
            return self.model(x, t_batch)

        # Time points
        t = torch.linspace(0, 1, num_steps + 1, device=device)

        # Solve ODE
        if verbose:
            print(f"Solving ODE with {num_steps} steps...")

        trajectory = odeint(
            ode_func,
            x0,
            t,
            method='dopri5',  # Adaptive Runge-Kutta
            rtol=1e-5,
            atol=1e-5
        )

        if return_trajectory:
            return trajectory
        else:
            return trajectory[-1]


# For compatibility
import torch.nn.functional as F
