"""Pooling operators and activity instrumentation for spiking CNNs."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as functional
from torch import nn


class SpikeTieMaxPool2d(nn.Module):
    """2D max pooling with explicit routing for simultaneous spikes."""

    def __init__(
        self,
        kernel_size: int = 2,
        stride: Optional[int] = None,
        mode: str = "deterministic",
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if kernel_size != 2 or (stride is not None and stride != 2):
            raise ValueError("SpikeTieMaxPool2d currently supports only 2x2, stride-2 pooling")
        if mode not in {
            "deterministic",
            "random",
            "membrane_excess",
            "membrane_least_excess",
        }:
            raise ValueError(f"Unsupported tie-break mode: {mode}")
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.mode = mode
        self.random_seed = random_seed
        self._generators: Dict[str, torch.Generator] = {}
        self.last_metrics: Dict[str, float] = {}

    def _generator(self, device: torch.device) -> Optional[torch.Generator]:
        if self.random_seed is None:
            return None
        key = str(device)
        if key not in self._generators:
            self._generators[key] = torch.Generator(device=device).manual_seed(self.random_seed)
        return self._generators[key]

    def forward(
        self,
        spikes: torch.Tensor,
        membrane: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool binary spikes, explicitly resolving only multi-spike windows."""
        pooled = functional.max_pool2d(spikes, self.kernel_size, self.stride)
        windows = functional.unfold(spikes, kernel_size=self.kernel_size, stride=self.stride)
        batch, channels, _, _ = spikes.shape
        windows = windows.view(batch, channels, self.kernel_size**2, -1)
        active = windows > 0
        multi_spike = active.sum(dim=2) >= 2
        self.last_metrics = {
            "multi_spike_windows": float(multi_spike.detach().sum().item()),
            "pool_windows": float(multi_spike.numel()),
            "multi_spike_window_rate": float(multi_spike.detach().float().mean().item()),
        }
        if self.mode == "deterministic" or not bool(multi_spike.any()):
            return pooled

        routed = pooled.flatten(start_dim=2)
        contested_spikes = windows.permute(0, 1, 3, 2)[multi_spike]
        if self.mode == "random":
            scores = torch.rand(
                contested_spikes.shape,
                device=spikes.device,
                dtype=spikes.dtype,
                generator=self._generator(spikes.device),
            )
        else:
            if membrane is None:
                raise ValueError("membrane tie-breaking requires pre-reset membrane values")
            membrane_windows = functional.unfold(
                membrane, kernel_size=self.kernel_size, stride=self.stride
            ).view(batch, channels, self.kernel_size**2, -1)
            scores = membrane_windows.permute(0, 1, 3, 2)[multi_spike]

        if self.mode == "membrane_least_excess":
            scores = scores.masked_fill(contested_spikes <= 0, float("inf"))
            winner_offset = scores.argmin(dim=1)
        else:
            scores = scores.masked_fill(contested_spikes <= 0, float("-inf"))
            winner_offset = scores.argmax(dim=1)
        selected = contested_spikes.gather(1, winner_offset.unsqueeze(1)).squeeze(1)
        routed[multi_spike] = selected
        winner_offset = winner_offset.detach()
        for offset in range(self.kernel_size**2):
            self.last_metrics[f"winner_offset_{offset}"] = float((winner_offset == offset).sum().item())
        return routed.view_as(pooled)


class SpikeActivityRecorder:
    """Collect detached per-layer, per-timestep activity summaries."""

    def __init__(self) -> None:
        self._metrics: Dict[str, float] = {}

    def record(self, layer: str, timestep: int, spikes: torch.Tensor) -> None:
        count = float(spikes.detach().sum().item())
        prefix = f"activity/{layer}/timestep_{timestep}"
        self._metrics[f"{prefix}/spike_count"] = count
        self._metrics[f"{prefix}/spikes_per_sample"] = count / spikes.shape[0]
        self._metrics[f"{prefix}/firing_rate"] = count / spikes.numel()

    def record_pool(self, layer: str, timestep: int, metrics: Dict[str, float]) -> None:
        prefix = f"pooling/{layer}/timestep_{timestep}"
        self._metrics.update({f"{prefix}/{key}": value for key, value in metrics.items()})

    def consume(self) -> Dict[str, float]:
        metrics = self._metrics
        self._metrics = {}
        return metrics
