"""Configurable convolutional SNN for pooling-placement experiments."""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from autoresearch.utils.snn_pooling import SpikeActivityRecorder, SpikeTieMaxPool2d
from autoresearch.utils.spike_encoding import (
    scaled_log_ttfs_encode,
    thresholded_log_ttfs_encode,
)


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
        output_mode: str = "spiking",
        beta: float = 0.9,
        threshold: float = 1.0,
        surrogate_kind: str = "fast_sigmoid",
        surrogate_slope: float = 25.0,
        reset_delay: bool = True,
        random_seed: int | None = None,
        ttfs_log_scale: float = 20.0,
        temporal_batch_norm: bool = False,
        projection_features: int | None = None,
        classifier_bias: bool = True,
        ttfs_input_threshold: float = 0.01,
        first_spike_survival: bool = False,
        nonwinner_regularization_lambda: float = 0.0,
        all_spike_regularization_lambda: float = 0.0,
        track_nonwinner_activity: bool = False,
    ) -> None:
        super().__init__()
        if len(channels) != len(convs_per_stage):
            raise ValueError("channels and convs_per_stage must have the same length")
        if pooling_placement not in {"pre_lif", "post_lif"}:
            raise ValueError("pooling_placement must be 'pre_lif' or 'post_lif'")
        if encoding not in {"direct", "scaled_log_ttfs", "thresholded_log_ttfs"}:
            raise ValueError(
                "encoding must be 'direct', 'scaled_log_ttfs', or 'thresholded_log_ttfs'"
            )
        if decoding not in {"tet", "first_spike"}:
            raise ValueError("decoding must be 'tet' or 'first_spike'")
        if output_mode not in {"spiking", "linear"}:
            raise ValueError("output_mode must be 'spiking' or 'linear'")
        if decoding == "first_spike" and output_mode != "spiking":
            raise ValueError("first-spike decoding requires a spiking output mode")
        if surrogate_kind not in {"fast_sigmoid", "atan"}:
            raise ValueError("surrogate_kind must be 'fast_sigmoid' or 'atan'")
        if pooling_placement == "pre_lif" and tie_break != "deterministic":
            raise ValueError("random and membrane tie-breaking are defined only for post-LIF pooling")
        if temporal_batch_norm and bntt:
            raise ValueError("temporal_batch_norm and bntt are mutually exclusive")
        if nonwinner_regularization_lambda < 0.0:
            raise ValueError("nonwinner_regularization_lambda must be non-negative")
        if all_spike_regularization_lambda < 0.0:
            raise ValueError("all_spike_regularization_lambda must be non-negative")
        if nonwinner_regularization_lambda > 0.0 and all_spike_regularization_lambda > 0.0:
            raise ValueError("Only one pool-activity regularizer may be active at a time")
        if (
            nonwinner_regularization_lambda > 0.0
            or all_spike_regularization_lambda > 0.0
        ) and (
            pooling_placement != "post_lif" or tie_break != "membrane_excess"
        ):
            raise ValueError(
                "pool-activity regularization requires post-LIF membrane_excess pooling"
            )

        self.timesteps = timesteps
        self.bntt = bntt
        self.encoding = encoding
        self.decoding = decoding
        self.output_mode = output_mode
        self.pooling_placement = pooling_placement
        self.pooled_stages = set(pooled_stages)
        self.threshold = threshold
        self.ttfs_log_scale = ttfs_log_scale
        self.temporal_batch_norm = temporal_batch_norm
        self.ttfs_input_threshold = ttfs_input_threshold
        self.first_spike_survival = first_spike_survival
        self.nonwinner_regularization_lambda = nonwinner_regularization_lambda
        self.all_spike_regularization_lambda = all_spike_regularization_lambda
        self.pool_activity_regularization_lambda = max(
            nonwinner_regularization_lambda,
            all_spike_regularization_lambda,
        )
        self.pool_activity_regularization_mode = (
            "nonwinner"
            if nonwinner_regularization_lambda > 0.0
            else "all" if all_spike_regularization_lambda > 0.0 else "none"
        )
        self.track_nonwinner_activity = track_nonwinner_activity
        spike_grad = (
            surrogate.fast_sigmoid(slope=surrogate_slope)
            if surrogate_kind == "fast_sigmoid"
            else surrogate.atan()
        )

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
                        reset_delay=reset_delay,
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
                    track_nonwinner_spikes=(
                        nonwinner_regularization_lambda > 0.0 or track_nonwinner_activity
                    ),
                )
                for stage in self.pooled_stages
            }
        )
        self.projection = (
            nn.Linear(channels[-1], projection_features, bias=classifier_bias)
            if projection_features is not None
            else None
        )
        self.projection_lif = (
            snn.Leaky(
                beta=beta,
                threshold=threshold,
                spike_grad=spike_grad,
                reset_mechanism="subtract",
                reset_delay=reset_delay,
            )
            if projection_features is not None
            else None
        )
        classifier_features = projection_features or channels[-1]
        self.classifier = nn.Linear(classifier_features, num_classes, bias=classifier_bias)
        self.output_lif = (
            snn.Leaky(
                beta=beta,
                threshold=threshold,
                spike_grad=spike_grad,
                reset_mechanism="subtract",
                reset_delay=reset_delay,
            )
            if output_mode == "spiking"
            else None
        )
        self._last_temporal_logits: torch.Tensor | None = None
        self._last_activity: Dict[str, float] = {}
        self._last_nonwinner_regularization: torch.Tensor | None = None
        self._last_pool_activity_regularization: torch.Tensor | None = None
        self._pool_activity_regularization_terms: List[torch.Tensor] = []
        self._record_activity = True

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.encoding == "direct":
            return x.unsqueeze(0).expand(self.timesteps, *x.shape)
        if self.encoding == "scaled_log_ttfs":
            return scaled_log_ttfs_encode(x, self.timesteps, self.ttfs_log_scale)
        return thresholded_log_ttfs_encode(
            x,
            self.timesteps,
            input_threshold=self.ttfs_input_threshold,
            log_scale=self.ttfs_log_scale,
        )

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

    def get_nonwinner_regularization_loss(self) -> torch.Tensor:
        """Compatibility alias for the most recent pool-activity penalty."""
        return self.get_pool_activity_regularization_loss()

    def get_pool_activity_regularization_loss(self) -> torch.Tensor:
        """Return the unweighted mean pool-activity penalty after a forward pass."""
        if self._last_pool_activity_regularization is None:
            raise RuntimeError("Pool-activity regularization is available only after a forward pass")
        return self._last_pool_activity_regularization

    def set_activity_recording(self, enabled: bool) -> None:
        """Enable or disable detached spike-activity summaries for a forward pass."""
        self._record_activity = enabled
        if not enabled:
            self._last_activity = {}

    def _decode(self, temporal_logits: torch.Tensor) -> torch.Tensor:
        """Decode temporal class values with TET or differentiable first-spike scores."""
        self._last_temporal_logits = temporal_logits
        if self.decoding == "tet":
            return temporal_logits.sum(dim=0)

        if self.first_spike_survival:
            prior_survival = torch.cat(
                [
                    torch.ones_like(temporal_logits[:1]),
                    torch.cumprod(1.0 - temporal_logits[:-1], dim=0),
                ],
                dim=0,
            )
            first_events = temporal_logits * prior_survival
        else:
            first_event_mask = temporal_logits.detach().cumsum(dim=0) == 1
            first_events = temporal_logits * first_event_mask.to(temporal_logits.dtype)
        temporal_weights = torch.arange(
            self.timesteps,
            0,
            -1,
            device=temporal_logits.device,
            dtype=temporal_logits.dtype,
        ).view(self.timesteps, 1, 1)
        return (first_events * temporal_weights).sum(dim=0)

    def _readout(
        self,
        current: torch.Tensor,
        timestep: int,
        output_membrane: torch.Tensor | None,
        projection_membrane: torch.Tensor | None,
        recorder: SpikeActivityRecorder | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        pooled = current.mean(dim=(-2, -1))
        if self.projection is not None and self.projection_lif is not None:
            projection_current = self.projection(pooled)
            projection_value, projection_membrane = self.projection_lif(
                projection_current, projection_membrane
            )
            if recorder is not None:
                recorder.record("projection", timestep, projection_value)
            pooled = projection_value
        output_current = self.classifier(pooled)
        if self.output_lif is not None:
            output_value, output_membrane = self.output_lif(
                output_current, output_membrane
            )
            if recorder is not None:
                recorder.record("output", timestep, output_value)
        else:
            output_value = output_current
        return output_value, output_membrane, projection_membrane

    def _apply_lif_and_pool(
        self,
        current: torch.Tensor,
        membrane: torch.Tensor | None,
        lif: snn.Leaky,
        stage_index: int,
        is_pool_site: bool,
        layer_name: str,
        timestep: int,
        recorder: SpikeActivityRecorder | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if is_pool_site and self.pooling_placement == "pre_lif":
            current = self.pre_pools[str(stage_index)](current)
            return lif(current, membrane)

        current, membrane = lif(current, membrane)
        if recorder is not None:
            recorder.record(layer_name, timestep, current)
        if is_pool_site:
            if self.pool_activity_regularization_mode == "all":
                self._pool_activity_regularization_terms.append(current.mean())
            pool = self.post_pools[str(stage_index)]
            current = pool(current, membrane)
            if (
                self.pool_activity_regularization_mode == "nonwinner"
                and pool.last_nonwinner_spikes is not None
            ):
                self._pool_activity_regularization_terms.append(
                    pool.last_nonwinner_spikes.mean()
                )
            if recorder is not None:
                recorder.record_pool(
                    layer_name,
                    timestep,
                    pool.last_metrics,
                )
        return current, membrane

    def _forward_sequential(
        self,
        inputs: torch.Tensor,
        recorder: SpikeActivityRecorder | None,
    ) -> torch.Tensor:
        membranes = self._initial_membranes()
        output_membrane = self.output_lif.init_leaky() if self.output_lif is not None else None
        projection_membrane = (
            self.projection_lif.init_leaky() if self.projection_lif is not None else None
        )
        output_values = []

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
                    current, membranes[stage_index][conv_index] = self._apply_lif_and_pool(
                        current,
                        membranes[stage_index][conv_index],
                        lif,
                        stage_index,
                        is_pool_site,
                        layer_name,
                        timestep,
                        recorder,
                    )
                    if recorder is not None:
                        recorder.record(f"{layer_name}_output", timestep, current)

            output_value, output_membrane, projection_membrane = self._readout(
                current,
                timestep,
                output_membrane,
                projection_membrane,
                recorder,
            )
            output_values.append(output_value)
        return torch.stack(output_values)

    def _forward_temporal_batch_norm(
        self,
        inputs: torch.Tensor,
        recorder: SpikeActivityRecorder | None,
    ) -> torch.Tensor:
        """Process layers over all timesteps to normalise their joint time-batch axis."""
        membranes = self._initial_membranes()
        current = inputs
        for stage_index, (stage_convs, stage_norms, stage_lifs) in enumerate(
            zip(self.convs, self.norms, self.lifs)
        ):
            for conv_index, (conv, norm, lif) in enumerate(
                zip(stage_convs, stage_norms, stage_lifs)
            ):
                time_steps, batch_size = current.shape[:2]
                normalized = norm(conv(current.flatten(0, 1))).reshape(
                    time_steps, batch_size, -1, *current.shape[-2:]
                )
                is_pool_site = (
                    stage_index in self.pooled_stages
                    and conv_index == len(stage_convs) - 1
                )
                layer_name = f"stage_{stage_index + 1}_conv_{conv_index + 1}"
                outputs = []
                for timestep in range(self.timesteps):
                    output, membranes[stage_index][conv_index] = self._apply_lif_and_pool(
                        normalized[timestep],
                        membranes[stage_index][conv_index],
                        lif,
                        stage_index,
                        is_pool_site,
                        layer_name,
                        timestep,
                        recorder,
                    )
                    if recorder is not None:
                        recorder.record(f"{layer_name}_output", timestep, output)
                    outputs.append(output)
                current = torch.stack(outputs)

        output_membrane = self.output_lif.init_leaky() if self.output_lif is not None else None
        projection_membrane = (
            self.projection_lif.init_leaky() if self.projection_lif is not None else None
        )
        output_values = []
        for timestep in range(self.timesteps):
            output_value, output_membrane, projection_membrane = self._readout(
                current[timestep],
                timestep,
                output_membrane,
                projection_membrane,
                recorder,
            )
            output_values.append(output_value)
        return torch.stack(output_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the SNN and return count or first-spike class scores."""
        inputs = self._encode(x)
        recorder = SpikeActivityRecorder() if self._record_activity else None
        self._pool_activity_regularization_terms = []
        temporal_logits = (
            self._forward_temporal_batch_norm(inputs, recorder)
            if self.temporal_batch_norm
            else self._forward_sequential(inputs, recorder)
        )
        if self._pool_activity_regularization_terms:
            self._last_pool_activity_regularization = torch.stack(
                self._pool_activity_regularization_terms
            ).mean()
        else:
            self._last_pool_activity_regularization = torch.zeros(
                (), device=x.device, dtype=x.dtype
            )
        self._last_nonwinner_regularization = self._last_pool_activity_regularization
        self._last_activity = recorder.consume() if recorder is not None else {}
        if recorder is not None and self.pool_activity_regularization_lambda > 0.0:
            self._last_activity["regularization/pool_activity_penalty"] = float(
                self._last_pool_activity_regularization.detach().item()
            )
            self._last_activity["regularization/weighted_pool_activity_penalty"] = float(
                (
                    self.pool_activity_regularization_lambda
                    * self._last_pool_activity_regularization
                )
                .detach()
                .item()
            )
        if recorder is not None and self.nonwinner_regularization_lambda > 0.0:
            self._last_activity["regularization/nonwinner_spike_penalty"] = float(
                self._last_pool_activity_regularization.detach().item()
            )
            self._last_activity["regularization/weighted_nonwinner_spike_penalty"] = float(
                (
                    self.nonwinner_regularization_lambda
                    * self._last_pool_activity_regularization
                )
                .detach()
                .item()
            )
        return self._decode(temporal_logits)
