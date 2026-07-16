"""Model evaluation utilities."""

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from autoresearch.utils.snn_losses import compute_model_loss


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> Dict[str, float]:
    """Validate the model.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        loss_fn: Loss function
        device: Device to validate on

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    activity_totals: Dict[str, float] = {}
    batches = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x)
            loss = compute_model_loss(model, loss_fn, y_pred, y)

            total_loss += loss.item()
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            if hasattr(model, "pop_activity_metrics"):
                for key, value in model.pop_activity_metrics().items():
                    activity_totals[key] = activity_totals.get(key, 0.0) + value
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break

    avg_loss = total_loss / batches
    accuracy = 100.0 * correct / total

    metrics = {"loss": avg_loss, "accuracy": accuracy}
    metrics.update({key: value / batches for key, value in activity_totals.items()})
    return metrics
