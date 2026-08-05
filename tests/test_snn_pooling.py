"""Unit tests for explicit spike-pooling routing."""

import torch

from autoresearch.utils.snn_pooling import SpikeTieMaxPool2d


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
