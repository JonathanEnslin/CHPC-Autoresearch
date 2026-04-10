"""Convolutional Neural Network implementation."""

from typing import List, Optional, Type

import torch
import torch.nn as nn


class CNN(nn.Module):
    """Flexible Convolutional Neural Network.

    Args:
        input_channels: Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        input_height: Height of input images
        input_width: Width of input images
        conv_channels: List of output channels for each conv layer
        conv_kernel_sizes: List of kernel sizes for each conv layer
        conv_strides: List of strides for each conv layer
        conv_paddings: List of padding for each conv layer
        pool_kernel_size: Kernel size for max pooling
        pool_stride: Stride for max pooling
        fc_hidden_units: List of hidden layer sizes for fully connected layers
        num_classes: Number of output classes
        activation: Activation function class (e.g., nn.ReLU)
        dropout: Dropout probability. If None, no dropout is applied.
    """

    def __init__(
        self,
        input_channels: int,
        input_height: int,
        input_width: int,
        conv_channels: List[int],
        conv_kernel_sizes: List[int],
        conv_strides: List[int],
        conv_paddings: List[int],
        pool_kernel_size: int = 2,
        pool_stride: int = 2,
        fc_hidden_units: List[int] = None,
        num_classes: int = 10,
        activation: Type[nn.Module] = nn.ReLU,
        dropout: Optional[float] = None,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width

        # Build convolutional layers
        conv_layers = []
        in_channels = input_channels
        current_height = input_height
        current_width = input_width

        for out_channels, kernel_size, stride, padding in zip(
            conv_channels, conv_kernel_sizes, conv_strides, conv_paddings
        ):
            conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            )
            conv_layers.append(activation)
            conv_layers.append(nn.MaxPool2d(pool_kernel_size, pool_stride))

            if dropout is not None and dropout > 0:
                conv_layers.append(nn.Dropout2d(dropout))

            # Calculate output dimensions after conv and pooling
            current_height = (current_height + 2 * padding - kernel_size) // stride + 1
            current_width = (current_width + 2 * padding - kernel_size) // stride + 1
            current_height = current_height // pool_stride
            current_width = current_width // pool_stride

            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_layers)

        # Calculate flattened size
        self.flattened_size = conv_channels[-1] * current_height * current_width

        # Build fully connected layers
        fc_layers = []
        in_features = self.flattened_size

        if fc_hidden_units:
            for out_features in fc_hidden_units:
                fc_layers.append(nn.Linear(in_features, out_features))
                fc_layers.append(activation)

                if dropout is not None and dropout > 0:
                    fc_layers.append(nn.Dropout(dropout))

                in_features = out_features

        # Output layer
        fc_layers.append(nn.Linear(in_features, num_classes))

        self.fc_layers = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x
