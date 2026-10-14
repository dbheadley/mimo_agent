"""
Unit tests for monosynaptic connection detection in mimo_pack.analysis.crosscorr.

Tests ccg_convolution_significance (moving-median Poisson baseline
significance) and find_significant_connections (excitatory/inhibitory
monosynaptic connection detection from cross-correlograms).
"""

import numpy as np
import pandas as pd
import pynapple as nap
import pytest

from mimo_pack.analysis.crosscorr import (
    ccg_convolution_significance,
    find_significant_connections,
)

# Matches the batch-analysis configuration: 0.5 ms bins, +/-20 ms window
BIN_SIZE = 0.0005
N_HALF_BINS = 40
LAG_MIN = 0.001
LAG_MAX = 0.010


def _synthetic_ccg(baseline_count=10.0, bump_bins=None, bump_counts=None, seed=0,
                   poisson_noise=False, col=(1, 2)):
    """Build a synthetic cross-correlogram counts DataFrame.

    Parameters
    ----------
    baseline_count : float
        Flat baseline spike count per bin.
    bump_bins : list of int, optional
        Bin indices (0-based into the lag axis) whose counts are replaced.
    bump_counts : list of float, optional
        Replacement counts for bump_bins.
    seed : int
        Random seed for Poisson noise.
    poisson_noise : bool
        If True, draw counts from a Poisson distribution around baseline.
    col : tuple
        Unit pair column label, (reference_unit, target_unit).
    """
    lags = np.arange(-N_HALF_BINS, N_HALF_BINS + 1) * BIN_SIZE
    rng = np.random.default_rng(seed)
    if poisson_noise:
        counts = rng.poisson(baseline_count, size=lags.size).astype(float)
    else:
        counts = np.full(lags.size, baseline_count, dtype=float)
    if bump_bins is not None:
        counts[np.asarray(bump_bins)] = np.asarray(bump_counts, dtype=float)
    return pd.DataFrame({col: counts}, index=lags)


def _detect(cc, **kwargs):
    """Run the full significance + detection pipeline on a synthetic CCG."""
    p_ex, p_in, bl_ex, bl_in = ccg_convolution_significance(cc)
    params = dict(lag_min_abs=LAG_MIN, lag_max_abs=LAG_MAX,
                  min_contiguous_bins=2, alpha=0.05)
    params.update(kwargs)
    return find_significant_connections(p_ex, p_in, bl_ex, bl_in, cc, **params)


def _lag_to_bin(lag_s):
    """Convert a lag in seconds to its 0-based bin index."""
    return N_HALF_BINS + int(round(lag_s / BIN_SIZE))


class TestCcgConvolutionSignificance:
    """Test suite for ccg_convolution_significance."""

    def test_output_types_and_shapes(self):
        cc = _synthetic_ccg()
        p_ex, p_in, bl_ex, bl_in = ccg_convolution_significance(cc)
        for df in (p_ex, p_in, bl_ex, bl_in):
            assert isinstance(df, pd.DataFrame)
            assert df.shape == cc.shape
            assert (df.index == cc.index).all()
            assert list(df.columns) == list(cc.columns)

    def test_flat_ccg_has_nonsignificant_pvalues(self):
        cc = _synthetic_ccg(baseline_count=10.0)
        p_ex, p_in, bl_ex, bl_in = ccg_convolution_significance(cc)
        col = cc.columns[0]
        # Baselines equal the flat counts, so p-values are far from 0
        assert np.allclose(bl_ex[col].values, 10.0)
        assert np.allclose(bl_in[col].values, 10.0)
        assert (p_ex[col].values > 0.4).all()
        assert (p_in[col].values > 0.4).all()

    def test_two_bins_closest_to_zero_are_excluded_from_baselines(self):
        cc = _synthetic_ccg(baseline_count=10.0)
        changed_center = cc.copy()
        excluded_bins = np.argsort(np.abs(cc.index.to_numpy()), kind="stable")[:2]
        changed_center.iloc[excluded_bins, 0] = 1_000.0

        _, _, baseline_ex, baseline_in = ccg_convolution_significance(cc)
        _, _, changed_baseline_ex, changed_baseline_in = (
            ccg_convolution_significance(changed_center)
        )

        assert np.allclose(baseline_ex.values, changed_baseline_ex.values)
        assert np.allclose(baseline_in.values, changed_baseline_in.values)

    def test_excitatory_peak_lowers_pvalue(self):
        peak_bin = _lag_to_bin(0.002)
        cc = _synthetic_ccg(bump_bins=[peak_bin, peak_bin + 1],
                            bump_counts=[40.0, 30.0])
        p_ex, _, bl_ex, _ = ccg_convolution_significance(cc)
        col = cc.columns[0]
        # Narrow bump does not shift the moving median much
        assert bl_ex[col].values[peak_bin] == pytest.approx(10.0)
        assert p_ex[col].values[peak_bin] < 1e-10

    def test_trough_lowers_inhibitory_pvalue(self):
        trough_bin = _lag_to_bin(0.002)
        cc = _synthetic_ccg(baseline_count=20.0,
                            bump_bins=[trough_bin, trough_bin + 1],
                            bump_counts=[0.0, 0.0])
        _, p_in, _, bl_in = ccg_convolution_significance(cc)
        col = cc.columns[0]
        assert bl_in[col].values[trough_bin] == pytest.approx(20.0)
        assert p_in[col].values[trough_bin] < 1e-6

    def test_rejects_non_dataframe_input(self):
        with pytest.raises(TypeError):
            ccg_convolution_significance(np.zeros(81))

    def test_empty_ccg_returns_empty_outputs(self):
        lags = np.arange(-N_HALF_BINS, N_HALF_BINS + 1) * BIN_SIZE
        cc = pd.DataFrame(index=lags)
        p_ex, p_in, bl_ex, bl_in = ccg_convolution_significance(cc)
        for df in (p_ex, p_in, bl_ex, bl_in):
            assert isinstance(df, pd.DataFrame)
            assert df.shape == (lags.size, 0)


class TestFindSignificantConnections:
    """Test suite for find_significant_connections."""

    def test_excitatory_peak_detected(self):
        peak_bin = _lag_to_bin(0.002)
        cc = _synthetic_ccg(bump_bins=[peak_bin, peak_bin + 1],
                            bump_counts=[40.0, 30.0])
        connections = _detect(cc)
        assert len(connections) == 1
        conn = connections[0]
        assert conn['direction'] == (1, 2)
        assert conn['sign'] == 1
        assert conn['lag_s'] == pytest.approx(0.002, abs=BIN_SIZE)
        # amplitude = (count - baseline) / baseline = (40 - 10) / 10
        assert conn['amplitude'] == pytest.approx(3.0, rel=0.01)

    def test_negative_lag_peak_reverses_direction(self):
        peak_bin = _lag_to_bin(-0.002)
        cc = _synthetic_ccg(bump_bins=[peak_bin, peak_bin - 1],
                            bump_counts=[40.0, 30.0])
        connections = _detect(cc)
        assert len(connections) == 1
        conn = connections[0]
        assert conn['direction'] == (2, 1)
        assert conn['sign'] == 1
        assert conn['lag_s'] == pytest.approx(0.002, abs=BIN_SIZE)

    def test_inhibitory_trough_detected(self):
        trough_bin = _lag_to_bin(0.003)
        cc = _synthetic_ccg(baseline_count=20.0,
                            bump_bins=[trough_bin, trough_bin + 1, trough_bin + 2],
                            bump_counts=[0.0, 1.0, 0.0])
        connections = _detect(cc)
        assert len(connections) == 1
        conn = connections[0]
        assert conn['direction'] == (1, 2)
        assert conn['sign'] == -1
        assert LAG_MIN < conn['lag_s'] <= LAG_MAX
        assert conn['amplitude'] < 0

    def test_flat_poisson_ccg_yields_no_detection(self):
        cc = _synthetic_ccg(baseline_count=10.0, poisson_noise=True, seed=42)
        connections = _detect(cc)
        assert connections == []

    def test_peak_outside_lag_window_rejected(self):
        # 15 ms is beyond the 10 ms monosynaptic window
        peak_bin = _lag_to_bin(0.015)
        cc = _synthetic_ccg(bump_bins=[peak_bin, peak_bin + 1],
                            bump_counts=[40.0, 30.0])
        connections = _detect(cc)
        assert connections == []

    def test_peak_in_near_zero_exclusion_rejected(self):
        # 0.5 ms is below the 1 ms minimum lag
        peak_bin = _lag_to_bin(0.0005)
        cc = _synthetic_ccg(bump_bins=[peak_bin],
                            bump_counts=[40.0])
        connections = _detect(cc, min_contiguous_bins=1)
        assert connections == []

    def test_single_significant_bin_rejected(self):
        # One significant bin fails the >=2 contiguous-bin requirement
        peak_bin = _lag_to_bin(0.002)
        cc = _synthetic_ccg(bump_bins=[peak_bin],
                            bump_counts=[40.0])
        connections = _detect(cc)
        assert connections == []

    def test_min_contiguous_bins_one_accepts_single_bin(self):
        peak_bin = _lag_to_bin(0.002)
        cc = _synthetic_ccg(bump_bins=[peak_bin],
                            bump_counts=[40.0])
        connections = _detect(cc, min_contiguous_bins=1)
        assert len(connections) == 1
        assert connections[0]['sign'] == 1

    def test_run_extending_past_lag_max_accepted_when_peak_in_window(self):
        # Peak at 9.5 ms; run continues to 11 ms, outside the 10 ms window
        start = _lag_to_bin(0.0095)
        cc = _synthetic_ccg(bump_bins=[start, start + 1, start + 2, start + 3],
                            bump_counts=[60.0, 40.0, 35.0, 30.0])
        connections = _detect(cc)
        assert len(connections) == 1
        assert connections[0]['lag_s'] == pytest.approx(0.0095, abs=BIN_SIZE / 2)

    def test_run_with_peak_outside_window_rejected(self):
        # Run spans 9.5-11 ms but its maximum is at 10.5 ms (outside window)
        start = _lag_to_bin(0.0095)
        cc = _synthetic_ccg(bump_bins=[start, start + 1, start + 2, start + 3],
                            bump_counts=[30.0, 35.0, 60.0, 40.0])
        assert _detect(cc) == []

    def test_run_spanning_exclusion_zone_accepted_when_peak_in_window(self):
        # Run covers 0.5-2 ms with its maximum at 1.5 ms
        start = _lag_to_bin(0.0005)
        cc = _synthetic_ccg(bump_bins=[start, start + 1, start + 2, start + 3],
                            bump_counts=[30.0, 35.0, 60.0, 40.0])
        connections = _detect(cc)
        assert len(connections) == 1
        assert connections[0]['lag_s'] == pytest.approx(0.0015, abs=BIN_SIZE / 2)

    def test_exclusion_zone_maximum_rejects_run(self):
        # Maximum at 0.5 ms (excluded) dominates the run reaching into 1-2 ms
        start = _lag_to_bin(0.0005)
        cc = _synthetic_ccg(bump_bins=[start, start + 1, start + 2],
                            bump_counts=[80.0, 40.0, 35.0])
        assert _detect(cc) == []

    def test_run_longer_than_max_contiguous_bins_rejected(self):
        start = _lag_to_bin(0.006)
        bins = list(range(start, start + 5))
        cc = _synthetic_ccg(bump_bins=bins, bump_counts=[40.0] * 5)
        assert _detect(cc, max_contiguous_bins=4) == []
        assert len(_detect(cc, max_contiguous_bins=10)) == 1

    def test_largest_deviation_selected_within_window(self):
        small = _lag_to_bin(0.002)
        large = _lag_to_bin(0.006)
        cc = _synthetic_ccg(bump_bins=[small, small + 1, large, large + 1],
                            bump_counts=[30.0, 30.0, 60.0, 50.0])
        connections = _detect(cc)
        assert len(connections) == 1
        assert connections[0]['lag_s'] == pytest.approx(0.006, abs=BIN_SIZE / 2)

    def test_multiple_columns_detected_independently(self):
        peak_bin = _lag_to_bin(0.002)
        lags = np.arange(-N_HALF_BINS, N_HALF_BINS + 1) * BIN_SIZE
        counts_flat = np.full(lags.size, 10.0)
        counts_peak = counts_flat.copy()
        counts_peak[peak_bin] = 40.0
        counts_peak[peak_bin + 1] = 30.0
        cc = pd.DataFrame({(1, 2): counts_peak, (1, 3): counts_flat}, index=lags)
        connections = _detect(cc)
        assert len(connections) == 1
        assert connections[0]['direction'] == (1, 2)

    def test_rejects_non_dataframe_input(self):
        cc = _synthetic_ccg()
        p_ex, p_in, bl_ex, bl_in = ccg_convolution_significance(cc)
        with pytest.raises(TypeError):
            find_significant_connections(p_ex, p_in, bl_ex, bl_in,
                                         cc.values,
                                         lag_min_abs=LAG_MIN,
                                         lag_max_abs=LAG_MAX)

    def test_empty_ccg_yields_no_connections(self):
        lags = np.arange(-N_HALF_BINS, N_HALF_BINS + 1) * BIN_SIZE
        cc = pd.DataFrame(index=lags)
        p_ex, p_in, bl_ex, bl_in = ccg_convolution_significance(cc)
        connections = find_significant_connections(
            p_ex, p_in, bl_ex, bl_in, cc,
            lag_min_abs=LAG_MIN, lag_max_abs=LAG_MAX)
        assert connections == []


class TestSpikeTrainIntegration:
    """End-to-end detection from synthetic Poisson spike trains.

    Mirrors the batch pipeline: pynapple cross-correlogram (norm=False)
    converted to raw counts, then significance testing and connection
    detection.
    """

    DURATION = 200.0  # seconds
    RATE = 20.0       # Hz

    @staticmethod
    def _poisson_train(rate, duration, rng):
        """Homogeneous Poisson spike train as sorted times in seconds."""
        n = rng.poisson(rate * duration)
        return np.sort(rng.uniform(0.0, duration, n))

    @staticmethod
    def _ccg_counts(pre, post):
        """Cross-correlogram of raw spike counts for a pre/post pair."""
        grp = nap.TsGroup({1: nap.Ts(t=pre), 2: nap.Ts(t=post)})
        cc_norm = nap.compute_crosscorrelogram(
            grp, binsize=BIN_SIZE, windowsize=0.02, norm=False)
        n_ref = np.array([len(grp[col[0]]) for col in cc_norm.columns])
        return pd.DataFrame(
            (cc_norm.values * (BIN_SIZE * n_ref)).round(),
            index=cc_norm.index, columns=cc_norm.columns)

    def test_injected_excitatory_coupling_detected(self):
        rng = np.random.default_rng(0)
        pre = self._poisson_train(self.RATE, self.DURATION, rng)
        post = self._poisson_train(self.RATE, self.DURATION, rng)
        # Inject spike transmission at ~2 ms lag with p=0.25
        transmitted = rng.random(pre.size) < 0.25
        delays = rng.uniform(0.0015, 0.0030, transmitted.sum())
        post = np.sort(np.concatenate([post, pre[transmitted] + delays]))

        cc = self._ccg_counts(pre, post)
        connections = _detect(cc)
        assert len(connections) == 1
        conn = connections[0]
        assert conn['direction'] == (1, 2)
        assert conn['sign'] == 1
        assert 0.0015 <= conn['lag_s'] <= 0.0030
        assert conn['amplitude'] > 0

    def test_independent_trains_yield_no_detection(self):
        rng = np.random.default_rng(1)
        pre = self._poisson_train(self.RATE, self.DURATION, rng)
        post = self._poisson_train(self.RATE, self.DURATION, rng)
        cc = self._ccg_counts(pre, post)
        connections = _detect(cc)
        assert connections == []

    def test_injected_inhibitory_coupling_detected(self):
        rng = np.random.default_rng(2)
        pre = self._poisson_train(self.RATE, self.DURATION, rng)
        post = self._poisson_train(2 * self.RATE, self.DURATION, rng)
        # Suppress post spikes 1-4 ms after each pre spike
        idx = np.searchsorted(pre, post) - 1
        valid = idx >= 0
        diff = np.full(post.size, np.inf)
        diff[valid] = post[valid] - pre[idx[valid]]
        post = post[~((diff > 0.001) & (diff < 0.004))]

        cc = self._ccg_counts(pre, post)
        connections = _detect(cc)
        assert len(connections) == 1
        conn = connections[0]
        assert conn['direction'] == (1, 2)
        assert conn['sign'] == -1
        assert LAG_MIN < conn['lag_s'] <= LAG_MAX
        assert conn['amplitude'] < 0
