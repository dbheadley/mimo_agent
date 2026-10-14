"""
Unit tests for mimo_pack.analysis.point_process module.

Tests the application-agnostic Gaussian mixture model fit for point process
(spike train) data relative to a set of reference event times.
"""

import numpy as np
import pynapple as nap
import pytest

from mimo_pack.analysis.point_process import (
    GaussianComponent,
    PointProcessGMMFitResult,
    fit_point_process_gmm,
)


def _synthetic_latency_spikes(mu_s=0.005, sigma_s=0.0005, n_events=200, seed=0):
    """Build spike times with a single latency component after each event."""
    rng = np.random.default_rng(seed)
    ref_times = np.arange(n_events, dtype=float)
    latencies = rng.normal(mu_s, sigma_s, size=n_events)
    spike_times = np.sort(ref_times + latencies)
    return spike_times, ref_times


class TestFitPointProcessGMM:
    """Test suite for fit_point_process_gmm."""

    def test_basic_functionality_recovers_single_component(self):
        spike_times, ref_times = _synthetic_latency_spikes()

        result = fit_point_process_gmm(spike_times, ref_times, window_s=0.030)

        assert isinstance(result, PointProcessGMMFitResult)
        assert result.n_gaussians == 1
        assert len(result.components) == 1
        comp = result.components[0]
        assert isinstance(comp, GaussianComponent)
        assert comp.mu_s == pytest.approx(0.005, abs=0.001)
        assert comp.weight == pytest.approx(1.0, abs=1e-6)
        assert not hasattr(comp, "corrected_prob")
        assert result.n_events == 200
        assert result.window_s == (0.030, 0.030)
        assert isinstance(result.rel_spike_times, np.ndarray)
        assert len(result.rel_spike_time_parts) == 200
        assert np.allclose(np.concatenate(result.rel_spike_time_parts), result.rel_spike_times)

    def test_accepts_pynapple_ts_inputs(self):
        spike_times, ref_times = _synthetic_latency_spikes()
        spike_ts = nap.Ts(t=spike_times)
        ref_ts = nap.Ts(t=ref_times)

        result = fit_point_process_gmm(spike_ts, ref_ts, window_s=0.030)

        assert isinstance(result, PointProcessGMMFitResult)
        assert result.n_gaussians == 1

    def test_empty_ref_times_returns_none(self):
        spike_times = np.array([0.1, 0.2, 0.3])

        result = fit_point_process_gmm(spike_times, np.array([]))

        assert result is None

    def test_too_few_spikes_returns_empty_components(self):
        rng = np.random.default_rng(1)
        ref_times = np.arange(50, dtype=float)
        # No spikes fall in the post-event window.
        spike_times = ref_times + 0.5

        result = fit_point_process_gmm(spike_times, ref_times, window_s=0.030, min_spikes=10)

        assert isinstance(result, PointProcessGMMFitResult)
        assert result.n_gaussians == 0
        assert result.components == []
        assert np.isnan(result.bic)
        assert result.n_events == 50

    def test_two_component_mixture_is_selected_by_bic(self):
        rng = np.random.default_rng(2)
        n_events = 300
        ref_times = np.arange(n_events, dtype=float)
        latencies_a = rng.normal(0.003, 0.0003, size=n_events)
        latencies_b = rng.normal(0.015, 0.0003, size=n_events)
        spike_times = np.sort(
            np.concatenate([ref_times + latencies_a, ref_times + latencies_b])
        )

        result = fit_point_process_gmm(spike_times, ref_times, window_s=0.030)

        assert result.n_gaussians == 2
        mus = sorted(c.mu_s for c in result.components)
        assert mus[0] == pytest.approx(0.003, abs=0.001)
        assert mus[1] == pytest.approx(0.015, abs=0.001)
        assert sum(c.weight for c in result.components) == pytest.approx(1.0, abs=1e-3)

    def test_asymmetric_window_only_collects_requested_sides(self):
        # A response at +5 ms and a spontaneous spike at -20 ms per event.
        rng = np.random.default_rng(4)
        n_events = 100
        ref_times = np.arange(n_events, dtype=float)
        post_latencies = rng.normal(0.005, 0.0003, size=n_events)
        pre_latencies = rng.normal(-0.020, 0.0003, size=n_events)
        spike_times = np.sort(
            np.concatenate([ref_times + post_latencies, ref_times + pre_latencies])
        )

        # Asymmetric window: 10 ms before, 8 ms after -- excludes the -20 ms
        # spontaneous spikes but keeps the +5 ms response.
        result = fit_point_process_gmm(spike_times, ref_times, window_s=(0.010, 0.008))

        assert result.window_s == (0.010, 0.008)
        assert result.n_gaussians == 1
        assert result.components[0].mu_s == pytest.approx(0.005, abs=0.001)
        assert np.all(result.rel_spike_times > -0.010)
        assert np.all(result.rel_spike_times <= 0.008)

    def test_rel_spike_time_parts_align_with_ref_times(self):
        ref_times = np.array([0.0, 1.0, 2.0])
        spike_times = np.array([0.005, 1.010, 1.015, 2.5])  # last spike outside any window

        result = fit_point_process_gmm(
            spike_times, ref_times, window_s=0.030, min_spikes=1
        )

        assert len(result.rel_spike_time_parts) == 3
        np.testing.assert_allclose(result.rel_spike_time_parts[0], [0.005])
        np.testing.assert_allclose(result.rel_spike_time_parts[1], [0.010, 0.015])
        assert result.rel_spike_time_parts[2].size == 0
