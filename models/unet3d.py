import torch
import torch.nn as nn
from .blocks import DoubleConv, EncoderBlock, DecoderBlock


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric motion correction.

    Architecture:
      - Encoder: `depth` levels, feature maps doubling each level.
      - Bottleneck: DoubleConv at the deepest resolution.
      - Decoder: mirrors encoder with skip connections.
      - InstanceNorm + LeakyReLU throughout (robust for MRI patch batches).
      - Trilinear upsampling avoids checkerboard artifacts.

    Memory footprint for a 64³ patch with base_features=32, depth=4:
      ~10 M parameters, ~4 GB VRAM at batch_size=2 with AMP.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_features: int = 32,
        depth: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.depth = depth

        # Feature-channel sizes per level: [32, 64, 128, 256, 512] for depth=4
        feats = [base_features * (2 ** i) for i in range(depth + 1)]

        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for i in range(depth):
            self.encoders.append(EncoderBlock(in_ch, feats[i], dropout=0.0))
            in_ch = feats[i]

        self.bottleneck = DoubleConv(feats[depth - 1], feats[depth], dropout=dropout)

        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            self.decoders.append(
                DecoderBlock(feats[i + 1], feats[i], feats[i], dropout=dropout)
            )

        self.out_conv = nn.Conv3d(feats[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.out_conv(x)
