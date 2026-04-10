"""Feed-Forward Neural Network implementation."""

from typing import List, Optional, Type, Union

import torch
import torch.nn as nn


class FFNN(nn.Module):
    """Flexible Feed-Forward Neural Network.

    Args:
        input_size: Size of input features (e.g., 784 for MNIST 28x28 images)
        hidden_units: List of hidden layer sizes. Last element is output size.
        activation: Activation function class or instance (e.g., nn.ReLU or nn.ReLU())
        dropout: Dropout probability. If None, no dropout is applied.
        flatten_input: Whether to flatten input (for image data)
    """

    def __init__(
        self,
        input_size: int,
        hidden_units: List[int],
        activation: Union[Type[nn.Module], nn.Module] = nn.ReLU,
        dropout: Optional[float] = None,
        flatten_input: bool = True,
    ):
        super().__init__()

        self.flatten_input = flatten_input
        self.input_size = input_size

        # Build layers
        layers = []
        in_features = input_size

        for _, out_features in enumerate(hidden_units[:-1]):
            layers.append(nn.Linear(in_features, out_features))
            # Handle both class and instance
            if isinstance(activation, type):
                layers.append(activation())
            else:
                layers.append(activation)

            if dropout is not None and dropout > 0:
                layers.append(nn.Dropout(dropout))

            in_features = out_features

        # Output layer (no activation)
        layers.append(nn.Linear(in_features, hidden_units[-1]))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        if self.flatten_input:
            x = x.view(x.size(0), -1)

        return self.network(x)
