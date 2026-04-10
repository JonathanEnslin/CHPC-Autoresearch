"""Device setup utilities."""

import logging

import torch

log = logging.getLogger(__name__)


def setup_device() -> torch.device:
    """Setup and return the device for training.

    Returns:
        Device to use for training
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)
    return device
