"""Compact conditional U-Net with residual blocks and sinusoidal time embeddings.

Independent reconstruction inspired by diffusion architectures practiced in NVIDIA DLI.
No original DLI code is embedded in this module.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, width: int) -> torch.Tensor:
    """Sinusoidal time features with shape [batch, width]."""
    half = width // 2
    frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    args = t.float().reshape(-1, 1) * frequencies[None]
    embedding = torch.cat([args.sin(), args.cos()], dim=-1)
    return F.pad(embedding, (0, width - embedding.shape[-1]))


def gn(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_width: int):
        super().__init__()
        self.first = nn.Sequential(gn(in_channels), nn.SiLU(), nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.second = nn.Sequential(gn(out_channels), nn.SiLU(), nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.emb = nn.Linear(emb_width, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.first(x)
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        return self.skip(x) + self.second(h)


class ConditionalUNet(nn.Module):
    """Predict epsilon for x_t. Context is a class integer or -1 (dropped context).

    For CFG, class=-1 produces the unconditional branch. Input dimensions must
    be divisible by four (MNIST is zero-padded from 28 to 32).
    """
    def __init__(self, channels: int = 1, classes: int = 10, base: int = 32, time_width: int = 128):
        super().__init__()
        self.channels, self.classes, self.base, self.time_width = channels, classes, base, time_width
        self.time_mlp = nn.Sequential(nn.Linear(time_width, time_width), nn.SiLU(), nn.Linear(time_width, time_width))
        self.class_embed = nn.Embedding(classes + 1, time_width)
        self.stem = nn.Conv2d(channels, base, 3, padding=1)
        self.down1 = ResBlock(base, base, time_width)
        self.reduce1 = nn.Conv2d(base, base, 3, stride=2, padding=1)
        self.down2 = ResBlock(base, base * 2, time_width)
        self.reduce2 = nn.Conv2d(base * 2, base * 2, 3, stride=2, padding=1)
        self.middle = ResBlock(base * 2, base * 2, time_width)
        self.up2 = ResBlock(base * 4, base, time_width)
        self.up1 = ResBlock(base * 2, base, time_width)
        self.head = nn.Sequential(gn(base), nn.SiLU(), nn.Conv2d(base, channels, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if x.shape[-1] % 4 or x.shape[-2] % 4:
            raise ValueError('Spatial dimensions must be divisible by 4')
        if labels is None:
            labels = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
        emb = self.time_mlp(timestep_embedding(t, self.time_width)) + self.class_embed(labels.long() + 1)
        x0 = self.stem(x)
        d1 = self.down1(x0, emb)
        d2 = self.down2(self.reduce1(d1), emb)
        middle = self.middle(self.reduce2(d2), emb)
        u2 = F.interpolate(middle, size=d2.shape[-2:], mode='nearest')
        u2 = self.up2(torch.cat([u2, d2], dim=1), emb)
        u1 = F.interpolate(u2, size=d1.shape[-2:], mode='nearest')
        u1 = self.up1(torch.cat([u1, d1], dim=1), emb)
        return self.head(u1)
