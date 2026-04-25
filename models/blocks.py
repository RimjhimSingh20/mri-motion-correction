import torch
import torch.nn as nn
from typing import Optional, Tuple


class ConvBlock(nn.Module):
    """Conv3d → InstanceNorm3d → LeakyReLU (+ optional Dropout3d)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list = [
            nn.Conv3d(in_ch, out_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout3d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    """Two sequential ConvBlocks."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        mid_ch: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block = nn.Sequential(
            ConvBlock(in_ch, mid_ch, dropout=0.0),
            ConvBlock(mid_ch, out_ch, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """DoubleConv → MaxPool3d(2).  Returns (pooled, skip)."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    """Trilinear upsample → 1×1 conv → concat skip → DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(in_ch, in_ch // 2, kernel_size=1),
        )
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch, dropout=dropout)

    def _center_crop(self, skip: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Crop skip to match target spatial dims (handles odd-size volumes)."""
        diff = [s - t for s, t in zip(skip.shape[2:], target.shape[2:])]
        slices = (slice(None), slice(None)) + tuple(
            slice(d // 2, d // 2 + t) for d, t in zip(diff, target.shape[2:])
        )
        return skip[slices]

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if skip.shape[2:] != x.shape[2:]:
            skip = self._center_crop(skip, x)
        return self.conv(torch.cat([x, skip], dim=1))
