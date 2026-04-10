"""Models package."""

from autoresearch.models.cnn import CNN
from autoresearch.models.ffnn import FFNN
from autoresearch.models.resnet import ResNet18

__all__ = [
    "FFNN",
    "CNN",
    "ResNet18",
]
