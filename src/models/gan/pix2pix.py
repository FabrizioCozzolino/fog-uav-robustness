"""Pix2Pix models: Generator (U-Net) and Discriminator (PatchGAN).

Reference:
    Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks"
    (CVPR 2017)  --  https://arxiv.org/abs/1611.07004

Two networks:

  Generator G:
      Takes a clean image (3 channels), outputs a foggy image (3 channels).
      Architecture: a U-Net with skip connections (we reuse smp.Unet so we
      benefit from a ResNet-34 encoder pretrained on ImageNet, just like for
      our segmentation task).
      Output activation: tanh, so values are in [-1, +1] (same range as the
      tanh-normalized targets).

  Discriminator D (PatchGAN):
      Takes a pair (clean, foggy) -- concatenated along the channel axis to
      6 channels -- and outputs a 30x30 map of real/fake scores. Each score
      corresponds to a 70x70 receptive field on the input. This forces D to
      focus on local texture quality (where fog "lives") rather than the
      overall image structure.
      Architecture: 5 strided 4x4 convolutions, channels 64->128->256->512->1.
      No sigmoid: we'll use BCEWithLogitsLoss in the training loop.
"""
from typing import Tuple

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


# ---------------------------------------------------------------- Generator


class Pix2PixGenerator(nn.Module):
    """U-Net generator for Pix2Pix.

    Wraps `smp.Unet`, replaces its output with a 3-channel RGB head, and
    applies tanh so the output is in [-1, +1] (matching the tanh-normalized
    targets).
    """

    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        out_channels: int = 3,
    ) -> None:
        super().__init__()
        # We reuse smp.Unet because it's well tested, but we use it as a
        # generic image-to-image network: 3 input channels (clean RGB), 3
        # output channels (foggy RGB).
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
            activation=None,  # raw logits; we apply tanh ourselves
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) tanh-normalized clean image in [-1, +1]
        out = self.unet(x)             # (B, 3, H, W) raw activations
        out = torch.tanh(out)           # bring to [-1, +1]
        return out


# ------------------------------------------------------------- Discriminator


def _disc_block(
    in_ch: int,
    out_ch: int,
    stride: int = 2,
    use_norm: bool = True,
) -> nn.Sequential:
    """Conv 4x4 (stride s) -> [BatchNorm] -> LeakyReLU(0.2).

    The first block of PatchGAN does NOT use normalization.
    """
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride, padding=1, bias=not use_norm),
    ]
    if use_norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator (Isola et al., 2017).

    Takes the concatenation of (clean, foggy) along the channel axis, so the
    discriminator is *conditional* (it sees the input image, not just the
    output). With 256x256 inputs this produces a 30x30 score map; each
    output element has a receptive field of about 70x70 pixels on the input.

    No sigmoid at the end: pair this with `BCEWithLogitsLoss` for numerical
    stability.
    """

    def __init__(self, in_channels: int = 6, ndf: int = 64) -> None:
        """
        Args:
            in_channels: number of input channels. For Pix2Pix conditional D,
                this is 6 (clean RGB + foggy RGB concatenated).
            ndf: number of filters in the first conv layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            _disc_block(in_channels, ndf,        stride=2, use_norm=False),  # H/2
            _disc_block(ndf,         ndf * 2,    stride=2, use_norm=True),   # H/4
            _disc_block(ndf * 2,     ndf * 4,    stride=2, use_norm=True),   # H/8
            _disc_block(ndf * 4,     ndf * 8,    stride=1, use_norm=True),   # H/8 (stride 1)
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),       # final logit map
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """DCGAN-style init: small Gaussian for conv layers, default for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                nn.init.zeros_(m.bias)

    def forward(self, clean: torch.Tensor, foggy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clean: (B, 3, H, W) the conditioning image
            foggy: (B, 3, H, W) the candidate (real or generated) foggy image
        Returns:
            (B, 1, H', W') logit map. With H=W=256 -> H'=W'=30.
        """
        x = torch.cat([clean, foggy], dim=1)  # (B, 6, H, W)
        return self.net(x)


# --------------------------------------------------------------- factories


def build_pix2pix(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    ndf: int = 64,
) -> Tuple[Pix2PixGenerator, PatchGANDiscriminator]:
    """Build a (Generator, Discriminator) pair with sensible defaults."""
    G = Pix2PixGenerator(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        out_channels=3,
    )
    D = PatchGANDiscriminator(in_channels=6, ndf=ndf)
    return G, D


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
