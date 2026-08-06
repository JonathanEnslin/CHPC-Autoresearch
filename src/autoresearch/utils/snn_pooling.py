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
        track_nonwinner_spikes: bool = False,
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
        if track_nonwinner_spikes and mode != "membrane_excess":
            raise ValueError(
                "Non-winner spike tracking is defined only for membrane_excess pooling"
            )
        self.track_nonwinner_spikes = track_nonwinner_spikes
        self._generators: Dict[str, torch.Generator] = {}
        self.last_metrics: Dict[str, float] = {}
        self.last_nonwinner_spikes: Optional[torch.Tensor] = None

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
        """Pool binary spikes, explicitly resolving only multi-spike windows.

        When requested for greatest-excess routing, ``last_nonwinner_spikes``
        contains the live spikes not selected as the winner in each nonempty
        window. Its routing mask is detached, so it can be used as a surrogate-
        gradient activity regularizer without differentiating through argmax.
        """
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

        membrane_windows = None
        self.last_nonwinner_spikes = None
        if self.track_nonwinner_spikes:
            if membrane is None:
                raise ValueError("membrane tie-breaking requires pre-reset membrane values")
            membrane_windows = functional.unfold(
                membrane, kernel_size=self.kernel_size, stride=self.stride
            ).view(batch, channels, self.kernel_size**2, -1)
            active_windows = active.any(dim=2)
            winner_offset = membrane_windows.masked_fill(~active, float("-inf")).argmax(dim=2)
            winner_mask = functional.one_hot(
                winner_offset,
                num_classes=self.kernel_size**2,
            ).permute(0, 1, 3, 2).to(spikes.dtype)
            winner_mask = winner_mask * active_windows.unsqueeze(2).to(spikes.dtype)
            winner_mask = functional.fold(
                winner_mask.reshape(batch, channels * self.kernel_size**2, -1),
                output_size=spikes.shape[-2:],
                kernel_size=self.kernel_size,
                stride=self.stride,
            ).detach()
            self.last_nonwinner_spikes = spikes * (1.0 - winner_mask)
            nonwinner = self.last_nonwinner_spikes.detach()
            self.last_metrics.update(
                {
                    "nonwinner_spike_count": float(nonwinner.sum().item()),
                    "nonwinner_spike_rate": float(nonwinner.mean().item()),
                }
            )

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
            if membrane_windows is None:
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
        self._total_neuron_spike_count = 0.0
        self._total_neuron_spikes_per_sample = 0.0

    def record(
        self,
        layer: str,
        timestep: int,
        spikes: torch.Tensor,
        count_toward_network_total: bool = False,
    ) -> None:
        count = float(spikes.detach().sum().item())
        prefix = f"activity/{layer}/timestep_{timestep}"
        self._metrics[f"{prefix}/spike_count"] = count
        self._metrics[f"{prefix}/spikes_per_sample"] = count / spikes.shape[0]
        self._metrics[f"{prefix}/firing_rate"] = count / spikes.numel()
        if count_toward_network_total:
            self._total_neuron_spike_count += count
            self._total_neuron_spikes_per_sample += count / spikes.shape[0]

    def record_pool(self, layer: str, timestep: int, metrics: Dict[str, float]) -> None:
        prefix = f"pooling/{layer}/timestep_{timestep}"
        self._metrics.update({f"{prefix}/{key}": value for key, value in metrics.items()})

    def record_pool_spikes(
        self,
        layer: str,
        timestep: int,
        pre_pool_spikes: torch.Tensor,
        post_pool_spikes: torch.Tensor,
    ) -> None:
        """Record raw spike counts at both sides of a pooling operation."""
        prefix = f"pooling/{layer}/timestep_{timestep}"
        pre_count = float(pre_pool_spikes.detach().sum().item())
        post_count = float(post_pool_spikes.detach().sum().item())
        self._metrics.update(
            {
                f"{prefix}/pre_pool_spike_count": pre_count,
                f"{prefix}/post_pool_spike_count": post_count,
                f"{prefix}/pool_spike_count_reduction": pre_count - post_count,
                f"{prefix}/pre_pool_spikes_per_sample": pre_count / pre_pool_spikes.shape[0],
                f"{prefix}/post_pool_spikes_per_sample": post_count / post_pool_spikes.shape[0],
            }
        )

    def consume(self) -> Dict[str, float]:
        metrics = self._metrics
        metrics["activity/network/total_neuron_spike_count"] = self._total_neuron_spike_count
        metrics["activity/network/total_neuron_spikes_per_sample"] = (
            self._total_neuron_spikes_per_sample
        )
        self._metrics = {}
        self._total_neuron_spike_count = 0.0
        self._total_neuron_spikes_per_sample = 0.0
        return metrics
