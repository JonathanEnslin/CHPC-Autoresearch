"""Configurable convolutional SNN for pooling-placement experiments."""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from autoresearch.utils.snn_pooling import SpikeActivityRecorder, SpikeTieMaxPool2d
from autoresearch.utils.spike_encoding import scaled_log_ttfs_encode


class SNNPoolingCNN(nn.Module):
    """Convolutional SNN with controlled pre- and post-LIF pooling variants."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        channels: Sequence[int],
        convs_per_stage: Sequence[int],
        pooled_stages: Sequence[int],
        pooling_placement: str = "post_lif",
        tie_break: str = "deterministic",
        timesteps: int = 6,
        bntt: bool = False,
        encoding: str = "direct",
        decoding: str = "tet",
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_slope: float = 25.0,
        random_seed: int | None = None,
        ttfs_log_scale: float = 20.0,
    ) -> None:
        super().__init__()
        if len(channels) != len(convs_per_stage):
            raise ValueError("channels and convs_per_stage must have the same length")
        if pooling_placement not in {"pre_lif", "post_lif"}:
            raise ValueError("pooling_placement must be 'pre_lif' or 'post_lif'")
        if encoding not in {"direct", "scaled_log_ttfs"}:
            raise ValueError("encoding must be 'direct' or 'scaled_log_ttfs'")
        if decoding not in {"tet", "first_spike"}:
            raise ValueError("decoding must be 'tet' or 'first_spike'")
        if pooling_placement == "pre_lif" and tie_break != "deterministic":
            raise ValueError("random and membrane tie-breaking are defined only for post-LIF pooling")

        self.timesteps = timesteps
        self.bntt = bntt
        self.encoding = encoding
        self.decoding = decoding
        self.pooling_placement = pooling_placement
        self.pooled_stages = set(pooled_stages)
        self.threshold = threshold
        self.ttfs_log_scale = ttfs_log_scale
        spike_grad = surrogate.fast_sigmoid(slope=surrogate_slope)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.lifs = nn.ModuleList()
        previous_channels = input_channels
        for stage_channels, conv_count in zip(channels, convs_per_stage):
            stage_convs = nn.ModuleList()
            stage_norms = nn.ModuleList()
            stage_lifs = nn.ModuleList()
            for _ in range(conv_count):
                stage_convs.append(nn.Conv2d(previous_channels, stage_channels, 3, padding=1, bias=False))
                if bntt:
                    stage_norms.append(
                        nn.ModuleList(
                            [nn.BatchNorm2d(stage_channels) for _ in range(timesteps)]
                        )
                    )
                else:
                    stage_norms.append(nn.BatchNorm2d(stage_channels))
                stage_lifs.append(
                    snn.Leaky(
                        beta=beta,
                        threshold=threshold,
                        spike_grad=spike_grad,
                        reset_mechanism="subtract",
                        reset_delay=True,
                    )
                )
                previous_channels = stage_channels
            self.convs.append(stage_convs)
            self.norms.append(stage_norms)
            self.lifs.append(stage_lifs)

        self.pre_pools = nn.ModuleDict(
            {str(stage): nn.MaxPool2d(2, 2) for stage in self.pooled_stages}
        )
        self.post_pools = nn.ModuleDict(
            {
                str(stage): SpikeTieMaxPool2d(
                    mode=tie_break,
                    random_seed=None if random_seed is None else random_seed + stage,
                )
                for stage in self.pooled_stages
            }
        )
        self.classifier = nn.Linear(channels[-1], num_classes)
        self.output_lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=spike_grad,
            reset_mechanism="subtract",
            reset_delay=True,
        )
        self._last_temporal_logits: torch.Tensor | None = None
        self._last_activity: Dict[str, float] = {}
        self._record_activity = True

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoding == "direct":
            return x.unsqueeze(0).expand(self.timesteps, *x.shape)
        return scaled_log_ttfs_encode(x, self.timesteps, self.ttfs_log_scale)

    def _initial_membranes(self) -> List[List[torch.Tensor]]:
        return [[lif.init_leaky() for lif in stage] for stage in self.lifs]

    def get_temporal_logits(self) -> torch.Tensor:
        if self._last_temporal_logits is None:
            raise RuntimeError("Temporal logits are available only after a forward pass")
        return self._last_temporal_logits

    def pop_activity_metrics(self) -> Dict[str, float]:
        metrics = self._last_activity
        self._last_activity = {}
        return metrics

    def set_activity_recording(self, enabled: bool) -> None:
        """Enable or disable detached spike-activity summaries for a forward pass."""
        self._record_activity = enabled
        if not enabled:
            self._last_activity = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the SNN and return count or first-spike class scores."""
        inputs = self._encode(x)
        membranes = self._initial_membranes()
        output_membrane = self.output_lif.init_leaky()
        output_spikes = []
        recorder = SpikeActivityRecorder() if self._record_activity else None

        for timestep in range(self.timesteps):
            current = inputs[timestep]
            for stage_index, (stage_convs, stage_norms, stage_lifs) in enumerate(
                zip(self.convs, self.norms, self.lifs)
            ):
                for conv_index, (conv, norm, lif) in enumerate(
                    zip(stage_convs, stage_norms, stage_lifs)
                ):
                    normalizer = norm[timestep] if self.bntt else norm
                    current = normalizer(conv(current))
                    is_pool_site = (
                        stage_index in self.pooled_stages
                        and conv_index == len(stage_convs) - 1
                    )
                    layer_name = f"stage_{stage_index + 1}_conv_{conv_index + 1}"
                    if is_pool_site and self.pooling_placement == "pre_lif":
                        current = self.pre_pools[str(stage_index)](current)
                        current, membranes[stage_index][conv_index] = lif(
                            current, membranes[stage_index][conv_index]
                        )
                    else:
                        current, membranes[stage_index][conv_index] = lif(
                            current, membranes[stage_index][conv_index]
                        )
                        if recorder is not None:
                            recorder.record(layer_name, timestep, current)
                        if is_pool_site:
                            current = self.post_pools[str(stage_index)](
                                current, membranes[stage_index][conv_index]
                            )
                            if recorder is not None:
                                recorder.record_pool(
                                    layer_name,
                                    timestep,
                                    self.post_pools[str(stage_index)].last_metrics,
                                )
                    if recorder is not None:
                        recorder.record(f"{layer_name}_output", timestep, current)

            pooled = current.mean(dim=(-2, -1))
            output_current = self.classifier(pooled)
            output_spike, output_membrane = self.output_lif(output_current, output_membrane)
            if recorder is not None:
                recorder.record("output", timestep, output_spike)
            output_spikes.append(output_spike)

        temporal_logits = torch.stack(output_spikes)
        self._last_temporal_logits = temporal_logits
        self._last_activity = recorder.consume() if recorder is not None else {}
        if self.decoding == "tet":
            return temporal_logits.sum(dim=0)

        first_event_mask = temporal_logits.detach().cumsum(dim=0) == 1
        first_events = temporal_logits * first_event_mask.to(temporal_logits.dtype)
        temporal_weights = torch.arange(
            self.timesteps, 0, -1, device=x.device, dtype=x.dtype
        ).view(self.timesteps, 1, 1)
        return (first_events * temporal_weights).sum(dim=0)
