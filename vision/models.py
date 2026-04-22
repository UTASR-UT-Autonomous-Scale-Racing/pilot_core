"""
Candidate networks for drivable-track segmentation.

Every model maps a normalised RGB batch ``(N, 3, H, W)`` to one channel of
logits ``(N, 1, H, W)``: positive = drivable track, negative = not track.
Build one with :func:`build_model`.

| name        | Reference |
| ----------- | --------- |
| ``unet``    | O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015 |
| ``fast_scnn`` | R. P. K. Poudel, S. Liwicki, R. Cipolla, "Fast-SCNN: Fast Semantic Segmentation Network", BMVC 2019 |
| ``lraspp``  | A. Howard et al., "Searching for MobileNetV3" (LR-ASPP head), ICCV 2019 |
| ``deeplabv3`` | L.-C. Chen et al., "Rethinking Atrous Convolution for Semantic Image Segmentation", arXiv:1706.05587, 2017 (MobileNetV3 backbone) |

``lraspp`` and ``deeplabv3`` come from torchvision and can start from an
ImageNet-pretrained backbone (``pretrained_backbone=True`` downloads it).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

MODELS = ("unet", "fast_scnn", "lraspp", "deeplabv3")


# --- U-Net ---

def _double_conv(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Four-level U-Net with bilinear upsampling (compact width for real time)."""

    def __init__(self, base: int = 16) -> None:
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc = nn.ModuleList([_double_conv(3, c[0])] + [_double_conv(c[i], c[i + 1]) for i in range(4)])
        self.dec = nn.ModuleList([_double_conv(c[i + 1] + c[i], c[i]) for i in reversed(range(4))])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for block in self.dec:
            skip = skips.pop()
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return self.head(x)


# --- Fast-SCNN ---

class _ConvBNReLU(nn.Sequential):
    def __init__(self, c_in: int, c_out: int, k: int = 3, s: int = 1, groups: int = 1) -> None:
        super().__init__(nn.Conv2d(c_in, c_out, k, s, k // 2, groups=groups, bias=False),
                         nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class _DSConv(nn.Sequential):
    """Depthwise-separable convolution."""

    def __init__(self, c_in: int, c_out: int, s: int = 1) -> None:
        super().__init__(_ConvBNReLU(c_in, c_in, 3, s, groups=c_in), _ConvBNReLU(c_in, c_out, 1))


class _Bottleneck(nn.Module):
    """MobileNetV2 inverted residual."""

    def __init__(self, c_in: int, c_out: int, s: int, t: int = 6) -> None:
        super().__init__()
        self.use_res = s == 1 and c_in == c_out
        self.block = nn.Sequential(
            _ConvBNReLU(c_in, c_in * t, 1),
            _ConvBNReLU(c_in * t, c_in * t, 3, s, groups=c_in * t),
            nn.Conv2d(c_in * t, c_out, 1, bias=False), nn.BatchNorm2d(c_out),
        )

    def forward(self, x):
        return x + self.block(x) if self.use_res else self.block(x)


class _PyramidPooling(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.stages = nn.ModuleList([_ConvBNReLU(c, c // 4, 1) for _ in (1, 2, 3, 6)])
        self.bins = (1, 2, 3, 6)
        self.out = _ConvBNReLU(c * 2, c, 1)

    def forward(self, x):
        h, w = int(x.shape[-2]), int(x.shape[-1])
        feats = [x]
        for stage, b in zip(self.stages, self.bins):
            # Fixed-kernel average pooling (≈ b×b bins).  Unlike adaptive pooling it
            # exports to ONNX for any feature-map size; the result is upsampled anyway.
            k = (max(1, h // b), max(1, w // b))
            pooled = F.avg_pool2d(x, kernel_size=k, stride=k)
            feats.append(F.interpolate(stage(pooled), size=(h, w), mode="bilinear", align_corners=False))
        return self.out(torch.cat(feats, dim=1))


class FastSCNN(nn.Module):
    """Learning-to-downsample → global feature extractor → feature fusion → classifier."""

    def __init__(self) -> None:
        super().__init__()
        self.downsample = nn.Sequential(_ConvBNReLU(3, 32, 3, 2), _DSConv(32, 48, 2), _DSConv(48, 64, 2))
        layers = []
        for c_in, c_out, n, s in ((64, 64, 3, 2), (64, 96, 3, 2), (96, 128, 3, 1)):
            for i in range(n):
                layers.append(_Bottleneck(c_in if i == 0 else c_out, c_out, s if i == 0 else 1))
        self.global_features = nn.Sequential(*layers, _PyramidPooling(128))
        self.fuse_low = nn.Sequential(nn.Conv2d(64, 128, 1, bias=False), nn.BatchNorm2d(128))
        self.fuse_high = nn.Sequential(_ConvBNReLU(128, 128, 3, groups=128),
                                       nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128))
        self.classifier = nn.Sequential(_DSConv(128, 128), _DSConv(128, 128), nn.Dropout(0.1), nn.Conv2d(128, 1, 1))

    def forward(self, x):
        size = x.shape[-2:]
        low = self.downsample(x)
        high = self.global_features(low)
        high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = F.relu(self.fuse_low(low) + self.fuse_high(high))
        return F.interpolate(self.classifier(x), size=size, mode="bilinear", align_corners=False)


# --- torchvision models ---

class _TorchvisionSeg(nn.Module):
    """Adapts a torchvision segmentation model (dict output) to plain logits."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


def _lraspp(pretrained_backbone: bool) -> nn.Module:
    from torchvision.models import MobileNet_V3_Large_Weights
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large

    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    return _TorchvisionSeg(lraspp_mobilenet_v3_large(weights=None, weights_backbone=weights, num_classes=1))


def _deeplabv3(pretrained_backbone: bool) -> nn.Module:
    from torchvision.models import MobileNet_V3_Large_Weights
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    return _TorchvisionSeg(deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=weights, num_classes=1,
                                                        aux_loss=False))


def build_model(name: str, pretrained_backbone: bool = False) -> nn.Module:
    """Build a candidate network by name (see :data:`MODELS`)."""
    if name == "unet":
        return UNet()
    if name == "fast_scnn":
        return FastSCNN()
    if name == "lraspp":
        return _lraspp(pretrained_backbone)
    if name == "deeplabv3":
        return _deeplabv3(pretrained_backbone)
    raise ValueError(f"Unknown vision model {name!r}; options: {', '.join(MODELS)}")
