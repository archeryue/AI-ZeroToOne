"""
U-Net architecture for flow matching.
Based on the DDPM U-Net with time embeddings adapted for flow matching.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: 1D tensor of N timesteps
        dim: Dimension of the embedding
        max_period: Controls the minimum frequency of the embeddings

    Returns:
        Embedding tensor of shape (N, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResBlock(nn.Module):
    """Residual block with time embedding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # First convolution
        # Use adaptive group count: min(32, channels) but ensure it divides evenly
        num_groups_in = min(32, in_channels)
        while in_channels % num_groups_in != 0:
            num_groups_in -= 1

        self.conv1 = nn.Sequential(
            nn.GroupNorm(num_groups_in, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )

        # Time embedding projection
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        # Second convolution
        num_groups_out = min(32, out_channels)
        while out_channels % num_groups_out != 0:
            num_groups_out -= 1

        self.conv2 = nn.Sequential(
            nn.GroupNorm(num_groups_out, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        # Residual connection
        if in_channels != out_channels:
            self.residual_proj = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
            time_emb: (B, time_emb_dim)

        Returns:
            Output tensor (B, out_channels, H, W)
        """
        h = self.conv1(x)

        # Add time embedding
        time_emb = self.time_emb_proj(time_emb)
        h = h + time_emb[:, :, None, None]

        h = self.conv2(h)

        return h + self.residual_proj(x)


class AttentionBlock(nn.Module):
    """Self-attention block."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        # Use adaptive group count
        num_groups = min(32, channels)
        while channels % num_groups != 0:
            num_groups -= 1
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            Output tensor (B, C, H, W)
        """
        B, C, H, W = x.shape
        residual = x

        x = self.norm(x)
        qkv = self.qkv(x)  # (B, C*3, H, W)

        # Reshape for multi-head attention
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, num_heads, H*W, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        out = self.proj_out(out)
        return out + residual


class Downsample(nn.Module):
    """Downsampling layer."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net model for flow matching.

    This network predicts the velocity field v_t(x_t) for the flow matching ODE.
    """

    def __init__(
        self,
        image_channels: int = 3,
        model_channels: int = 128,
        channel_mult: tuple = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (16, 8),
        dropout: float = 0.1,
        num_heads: int = 4
    ):
        """
        Args:
            image_channels: Number of input/output image channels
            model_channels: Base channel count
            channel_mult: Channel multiplier for each resolution level
            num_res_blocks: Number of residual blocks per resolution
            attention_resolutions: Resolutions at which to apply attention
            dropout: Dropout rate
            num_heads: Number of attention heads
        """
        super().__init__()
        self.image_channels = image_channels
        self.model_channels = model_channels

        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        # Input convolution
        self.input_conv = nn.Conv2d(image_channels, model_channels, 3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        input_channels = [ch]
        current_resolution = 64  # Assuming input size is 64x64

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult

            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, out_ch, time_emb_dim, dropout)]

                if current_resolution in attention_resolutions:
                    layers.append(AttentionBlock(out_ch, num_heads))

                self.down_blocks.append(nn.ModuleList(layers))
                ch = out_ch
                input_channels.append(ch)

            # Downsample (except for last level)
            if level != len(channel_mult) - 1:
                self.down_blocks.append(nn.ModuleList([Downsample(ch)]))
                input_channels.append(ch)
                current_resolution //= 2

        # Middle blocks
        self.middle_blocks = nn.ModuleList([
            ResBlock(ch, ch, time_emb_dim, dropout),
            AttentionBlock(ch, num_heads),
            ResBlock(ch, ch, time_emb_dim, dropout)
        ])

        # Upsampling path
        self.up_blocks = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult

            for i in range(num_res_blocks + 1):
                # Skip connection from downsampling path
                skip_ch = input_channels.pop()
                layers = [ResBlock(ch + skip_ch, out_ch, time_emb_dim, dropout)]

                if current_resolution in attention_resolutions:
                    layers.append(AttentionBlock(out_ch, num_heads))

                ch = out_ch
                self.up_blocks.append(nn.ModuleList(layers))

            # Upsample (except for last level)
            if level != 0:
                self.up_blocks.append(nn.ModuleList([Upsample(ch)]))
                current_resolution *= 2

        # Output convolution
        num_groups_out = min(32, ch)
        while ch % num_groups_out != 0:
            num_groups_out -= 1
        self.output_conv = nn.Sequential(
            nn.GroupNorm(num_groups_out, ch),
            nn.SiLU(),
            nn.Conv2d(ch, image_channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity field for flow matching.

        Args:
            x: Input tensor (B, C, H, W), images at time t
            t: Time tensor (B,), values in [0, 1]

        Returns:
            Predicted velocity field (B, C, H, W)
        """
        # Time embedding
        t_emb = timestep_embedding(t, self.model_channels)
        t_emb = self.time_embed(t_emb)

        # Input
        h = self.input_conv(x)

        # Downsampling with skip connections
        # Save h after input and after each down block
        # This matches how input_channels was built during construction
        skip_connections = [h]
        for modules in self.down_blocks:
            for module in modules:
                if isinstance(module, ResBlock):
                    h = module(h, t_emb)
                elif isinstance(module, AttentionBlock):
                    h = module(h)
                else:  # Downsample
                    h = module(h)
            # Save output of each down block (ResBlock group or Downsample)
            skip_connections.append(h)

        # Middle
        for module in self.middle_blocks:
            if isinstance(module, ResBlock):
                h = module(h, t_emb)
            else:
                h = module(h)

        # Upsampling with skip connections
        # Process each up block, handling upsample layers separately
        for modules in self.up_blocks:
            # Check if this is just an upsample layer
            if len(modules) == 1 and isinstance(modules[0], Upsample):
                h = modules[0](h)
            else:
                # This is a ResBlock group (possibly with attention)
                for module in modules:
                    if isinstance(module, ResBlock):
                        skip = skip_connections.pop()
                        h = torch.cat([h, skip], dim=1)
                        h = module(h, t_emb)
                    elif isinstance(module, AttentionBlock):
                        h = module(h)

        # Output
        return self.output_conv(h)
