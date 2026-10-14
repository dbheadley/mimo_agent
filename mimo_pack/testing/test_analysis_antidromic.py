"""
Unit tests for mimo_pack.analysis.antidromic module.

Confirms the antidromic-specific wrapper around
mimo_pack.analysis.point_process.fit_point_process_gmm preserves its public
API (AntidromicFitResult.n_stims) after delegating the GMM fit to the
application-agnostic implementation, and that the per-trial collision
correction behaves as expected: a collision-window spike is only counted
when the response is otherwise absent, and stims with both are flagged.
"""

import numpy as np
import pytest

from mimo_pack.analysis.antidromic import AntidromicFitResult, fit_antidromic_gaussians


def _synthetic_latency_spikes(mu_s=0.005, sigma_s=0.0005, n_stims=200, seed=0):
    rng = np.random.default_rng(seed)
    stim_times = np.arange(n_stims, dtype=float)
    latencies = rng.normal(mu_s, sigma_s, size=n_stims)
    spike_times = np.sort(stim_times + latencies)
    return spike_times, stim_times


class TestFitAntidromicGaussians:
    """Test suite for fit_antidromic_gaussians."""

    def test_basic_functionality_recovers_single_component(self):
        spike_times, stim_times = _synthetic_latency_spikes()

        result = fit_antidromic_gaussians(spike_times, stim_times, window_s=0.030)

        assert isinstance(result, AntidromicFitResult)
        assert result.n_gaussians == 1
        assert result.n_stims == 200
        assert result.window_s == 0.030
        comp = result.components[0]
        assert comp.mu_s == pytest.approx(0.005, abs=0.001)
        assert comp.n_collision_violations == 0
        # Isolated pulses (1 s apart, 30 ms window): no pre-stim collisions
        # and no other components, so the correction should be negligible.
        assert comp.corrected_prob == pytest.approx(comp.raw_prob, abs=1e-3)

    def test_corrected_prob_counts_collision_only_stims_without_violation(self):
        # 30% of stims have a spontaneous spike in the collision window that
        # blocks the antidromic response (no response spike for that stim);
        # the rest respond normally. The collision-only stims should be
        # added back into corrected_prob without being flagged as
        # violations, since the response is genuinely absent for them.
        rng = np.random.default_rng(3)
        n_stims = 300
        stim_times = np.arange(n_stims, dtype=float)  # widely spaced, isolated
        blocked = rng.random(n_stims) < 0.3

        response_times = stim_times[~blocked] + rng.normal(0.005, 0.0003, size=(~blocked).sum())
        collision_times = stim_times[blocked] + rng.normal(-0.002, 0.0003, size=blocked.sum())
        spike_times = np.sort(np.concatenate([response_times, collision_times]))

        result = fit_antidromic_gaussians(spike_times, stim_times, window_s=0.030)

        comp = max(result.components, key=lambda c: c.mu_s)
        assert comp.mu_s == pytest.approx(0.005, abs=0.001)
        assert comp.n_collision_violations == 0
        assert comp.corrected_prob > comp.raw_prob
        assert comp.corrected_prob == pytest.approx(1.0, abs=0.05)

    def test_collision_and_response_together_flags_violation(self):
        # Every stim has both a response spike and a spike in the
        # corresponding collision window: this violates the antidromic
        # collision principle and must be flagged, not silently corrected.
        rng = np.random.default_rng(4)
        n_stims = 200
        stim_times = np.arange(n_stims, dtype=float)
        response_times = stim_times + rng.normal(0.005, 0.0003, size=n_stims)
        collision_times = stim_times + rng.normal(-0.002, 0.0003, size=n_stims)
        spike_times = np.sort(np.concatenate([response_times, collision_times]))

        with pytest.warns(UserWarning, match="Collision test violated"):
            result = fit_antidromic_gaussians(spike_times, stim_times, window_s=0.030)

        comp = max(result.components, key=lambda c: c.mu_s)
        assert comp.mu_s == pytest.approx(0.005, abs=0.001)
        assert comp.n_collision_violations > 0

    def test_empty_stim_times_returns_none(self):
        result = fit_antidromic_gaussians(np.array([0.1, 0.2]), np.array([]))

        assert result is None

    def test_too_few_spikes_returns_empty_components(self):
        stim_times = np.arange(50, dtype=float)
        spike_times = stim_times + 0.5  # outside the post-stimulus window

        result = fit_antidromic_gaussians(spike_times, stim_times, window_s=0.030, min_spikes=10)

        assert result.n_gaussians == 0
        assert result.components == []
        assert result.n_stims == 50
