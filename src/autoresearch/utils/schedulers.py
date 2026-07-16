"""Learning-rate schedulers used by AutoResearch experiments."""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineAnnealingLR(LRScheduler):
    """Linear warm-up followed by cosine decay over a fixed total epoch count."""

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        warmup_epochs: int = 5,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if not 0 < warmup_epochs < T_max:
            raise ValueError("warmup_epochs must be positive and shorter than T_max")
        self.T_max = T_max
        self.warmup_epochs = warmup_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_epochs:
            scale = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * scale for base_lr in self.base_lrs]

        progress = (self.last_epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return [self.eta_min + (base_lr - self.eta_min) * cosine for base_lr in self.base_lrs]
