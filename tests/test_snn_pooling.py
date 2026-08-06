"""Unit tests for explicit spike-pooling routing."""

import torch

from autoresearch.utils.snn_pooling import SpikeActivityRecorder, SpikeTieMaxPool2d


def test_membrane_excess_nonwinner_mask_exempts_the_greatest_excess_spike() -> None:
    pool = SpikeTieMaxPool2d(mode="membrane_excess", track_nonwinner_spikes=True)
    spikes = torch.tensor([[[[1.0, 1.0], [0.0, 1.0]]]], requires_grad=True)
    membrane = torch.tensor([[[[1.1, 1.8], [0.4, 1.4]]]])

    pooled = pool(spikes, membrane)

    assert torch.equal(pooled, torch.ones_like(pooled))
    assert pool.last_nonwinner_spikes is not None
    expected = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    assert torch.equal(pool.last_nonwinner_spikes, expected)
    pool.last_nonwinner_spikes.mean().backward()
    assert spikes.grad is not None
    assert spikes.grad[0, 0, 0, 1].item() == 0.0
    assert spikes.grad[0, 0, 0, 0].item() > 0.0
    assert spikes.grad[0, 0, 1, 1].item() > 0.0


def test_membrane_excess_nonwinner_mask_does_not_penalize_single_spike_windows() -> None:
    pool = SpikeTieMaxPool2d(mode="membrane_excess", track_nonwinner_spikes=True)
    spikes = torch.tensor([[[[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]]])
    membrane = torch.ones_like(spikes)

    pool(spikes, membrane)

    assert pool.last_nonwinner_spikes is not None
    assert pool.last_nonwinner_spikes.sum().item() == 0.0
    assert pool.last_metrics["nonwinner_spike_rate"] == 0.0


def test_activity_recorder_reports_raw_pool_transition_and_network_total() -> None:
    recorder = SpikeActivityRecorder()
    pre_pool = torch.tensor([[[[1.0, 1.0], [0.0, 1.0]]]])
    post_pool = torch.ones(1, 1, 1, 1)

    recorder.record("stage_1_conv_2", timestep=0, spikes=pre_pool, count_toward_network_total=True)
    recorder.record_pool_spikes("stage_1_conv_2", timestep=0, pre_pool_spikes=pre_pool, post_pool_spikes=post_pool)
    metrics = recorder.consume()

    prefix = "pooling/stage_1_conv_2/timestep_0"
    assert metrics[f"{prefix}/pre_pool_spike_count"] == 3.0
    assert metrics[f"{prefix}/post_pool_spike_count"] == 1.0
    assert metrics[f"{prefix}/pool_spike_count_reduction"] == 2.0
    assert metrics["activity/network/total_neuron_spike_count"] == 3.0
    assert metrics["activity/network/total_neuron_spikes_per_sample"] == 3.0
