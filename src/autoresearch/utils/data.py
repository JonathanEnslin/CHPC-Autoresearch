"""Data loading utilities."""

from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split


def create_dataloaders(
    dataset: Dataset,
    batch_size: int,
    validation_split: float = 0.2,
    num_workers: int = 4,
    seed: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders from a dataset.

    Args:
        dataset: PyTorch dataset
        batch_size: Batch size for dataloaders
        validation_split: Fraction of data to use for validation
        num_workers: Number of workers for data loading
        seed: Random seed for split reproducibility

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Calculate split sizes
    dataset_size = len(dataset)
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size

    # Split dataset
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size], generator=generator
        )
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
