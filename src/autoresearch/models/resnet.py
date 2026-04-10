"""ResNet wrapper for small-image datasets like CIFAR-10."""

import torch.nn as nn
from torchvision import models


class ResNet18(nn.Module):
    """ResNet-18 wrapper with optional CIFAR-10 modifications.

    For 32x32 images (CIFAR-10), the standard ResNet conv1 (7x7, stride 2)
    and maxpool aggressively downsample. We replace conv1 with a 3x3 conv
    (stride 1) and remove maxpool, which is standard practice for CIFAR.

    Args:
        num_classes: Number of output classes
        cifar_mode: If True, modify first conv and remove maxpool for 32x32 images
        pretrained: If True, use ImageNet pretrained weights (only when cifar_mode=False)
    """

    def __init__(
        self,
        num_classes: int = 10,
        cifar_mode: bool = True,
        pretrained: bool = False,
    ):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)

        if cifar_mode:
            # Replace 7x7 conv with 3x3 conv for small images
            base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # Remove maxpool (replace with identity)
            base.maxpool = nn.Identity()

        # Replace final FC layer
        base.fc = nn.Linear(base.fc.in_features, num_classes)

        self.model = base

    def forward(self, x):
        return self.model(x)
