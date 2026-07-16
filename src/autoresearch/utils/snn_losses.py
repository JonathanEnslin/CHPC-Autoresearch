"""Temporal losses for directly trained spiking neural networks."""

import torch
from torch import nn


class TemporalEfficientCrossEntropy(nn.Module):
    """TET-style interpolation of aggregate and per-timestep cross-entropy."""

    requires_temporal_logits = True

    def __init__(self, temporal_weight: float = 0.1) -> None:
        super().__init__()
        if not 0.0 <= temporal_weight <= 1.0:
            raise ValueError("temporal_weight must lie in [0, 1]")
        self.temporal_weight = temporal_weight
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        temporal_logits: torch.Tensor,
    ) -> torch.Tensor:
        aggregate_loss = self.cross_entropy(logits, target)
        temporal_loss = torch.stack(
            [self.cross_entropy(step_logits, target) for step_logits in temporal_logits]
        ).mean()
        return (1.0 - self.temporal_weight) * aggregate_loss + self.temporal_weight * temporal_loss


def compute_model_loss(
    model: nn.Module,
    loss_fn: nn.Module,
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Apply a loss, supplying temporal logits only when requested."""
    if getattr(loss_fn, "requires_temporal_logits", False):
        temporal_logits = model.get_temporal_logits()
        return loss_fn(logits, target, temporal_logits)
    return loss_fn(logits, target)
