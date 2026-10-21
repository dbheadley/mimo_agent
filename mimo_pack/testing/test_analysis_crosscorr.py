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
        # Run covers the two bumped bins (2.0 and 2.5 ms)
        assert conn['run_start_s'] == pytest.approx(0.002, abs=BIN_SIZE / 2)
        assert conn['run_end_s'] == pytest.approx(0.0025, abs=BIN_SIZE / 2)
        assert conn['run_width_bins'] == 2

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
        # Run lags are reported as absolute delays (smallest to largest)
        assert conn['run_start_s'] == pytest.approx(0.002, abs=BIN_SIZE / 2)
        assert conn['run_end_s'] == pytest.approx(0.0025, abs=BIN_SIZE / 2)
        assert conn['run_width_bins'] == 2

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
        assert conn['run_width_bins'] == 3

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

    def test_run_crossing_lag_max_is_clipped_to_window(self):
        # Run spans 9-10.5 ms; bins past the 10 ms window edge are not
        # considered, so the reported run must end inside the window.
        start = _lag_to_bin(0.009)
        cc = _synthetic_ccg(bump_bins=[start, start + 1, start + 2, start + 3],
                            bump_counts=[30.0, 40.0, 50.0, 60.0])
        connections = _detect(cc)
        assert len(connections) == 1
        conn = connections[0]
        assert conn['run_start_s'] == pytest.approx(0.009, abs=BIN_SIZE / 2)
        assert conn['run_end_s'] <= LAG_MAX + BIN_SIZE / 10
        assert conn['lag_s'] <= LAG_MAX + BIN_SIZE / 10
        assert 2 <= conn['run_width_bins'] <= 3

    def test_only_in_window_deviation_is_tested(self):
        # The global maximum (11 ms, 90 counts) is outside the window; the MAD
        # bin is chosen among in-window bins only, so the in-window run at
        # 9-9.5 ms is the one reported.
        start = _lag_to_bin(0.009)
        cc = _synthetic_ccg(bump_bins=[start, start + 1, start + 4],
                            bump_counts=[30.0, 35.0, 90.0])
        connections = _detect(cc)
        assert len(connections) == 1
        conn = connections[0]
        assert conn['lag_s'] == pytest.approx(0.0095, abs=BIN_SIZE / 2)
        assert conn['run_start_s'] == pytest.approx(0.009, abs=BIN_SIZE / 2)
        assert conn['run_end_s'] == pytest.approx(0.0095, abs=BIN_SIZE / 2)
        assert conn['run_width_bins'] == 2

    def test_run_spanning_exclusion_zone_is_clipped_to_window(self):
        # Run covers 0.5-2 ms with its maximum at 1.5 ms; bins at or below
        # 1 ms are outside the window and are not part of the run.
        start = _lag_to_bin(0.0005)
        cc = _synthetic_ccg(bump_bins=[start, start + 1, start + 2, start + 3],
                            bump_counts=[30.0, 35.0, 60.0, 40.0])
        connections = _detect(cc)
        assert len(connections) == 1
        conn = connections[0]
        assert conn['lag_s'] == pytest.approx(0.0015, abs=BIN_SIZE / 2)
        assert conn['run_start_s'] == pytest.approx(0.0015, abs=BIN_SIZE / 2)
        assert conn['run_end_s'] == pytest.approx(0.002, abs=BIN_SIZE / 2)
        assert conn['run_width_bins'] == 2

    def test_exclusion_zone_maximum_does_not_hide_in_window_run(self):
        # Larger deviation at 0.5 ms (excluded) is ignored; only the in-window
        # bin at 1.5 ms is tested, giving a run of width 1.
        start = _lag_to_bin(0.0005)
        cc = _synthetic_ccg(bump_bins=[start, start + 1, start + 2],
                            bump_counts=[80.0, 40.0, 35.0])
        assert _detect(cc) == []  # width 1 < min_contiguous_bins
        connections = _detect(cc, min_contiguous_bins=1)
        assert len(connections) == 1
        conn = connections[0]
        assert conn['lag_s'] == pytest.approx(0.0015, abs=BIN_SIZE / 2)
        assert conn['run_width_bins'] == 1

    def test_run_grows_both_directions_and_stops_at_nonsignificant_bin(self):
        # Peak (40) at 3.5 ms flanked by mildly elevated bins (17, p~0.03).
        # A non-significant bin (12) ends the run on the later side, so the
        # trailing 17 is excluded.
        start = _lag_to_bin(0.003)
        cc = _synthetic_ccg(
            bump_bins=[start, start + 1, start + 2, start + 3, start + 4],
            bump_counts=[17.0, 40.0, 17.0, 12.0, 17.0])
        connections = _detect(cc)
        assert len(connections) == 1
        conn = connections[0]
        assert conn['lag_s'] == pytest.approx(0.0035, abs=BIN_SIZE / 2)
        assert conn['run_start_s'] == pytest.approx(0.003, abs=BIN_SIZE / 2)
        assert conn['run_end_s'] == pytest.approx(0.004, abs=BIN_SIZE / 2)
        assert conn['run_width_bins'] == 3

    def test_no_bonferroni_correction_applied(self):
        # Counts of 17 against a baseline of 10 give p ~ 0.027: significant at
        # uncorrected alpha=0.05 but far above alpha / n_bins.
        start = _lag_to_bin(0.004)
        cc = _synthetic_ccg(bump_bins=[start, start + 1], bump_counts=[17.0, 17.0])
        p_ex, _, _, _ = ccg_convolution_significance(cc)
        assert 0.05 / len(cc) < p_ex[cc.columns[0]].values[start] < 0.05
        connections = _detect(cc)
        assert len(connections) == 1
        assert connections[0]['run_width_bins'] == 2
        # A stricter alpha rejects the same CCG
        assert _detect(cc, alpha=0.01) == []

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
        assert connections[0]['run_start_s'] == pytest.approx(0.006, abs=BIN_SIZE / 2)
        assert connections[0]['run_width_bins'] == 2

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


# ---------------------------------------------------------------------------
# Jitter surrogate test
# ---------------------------------------------------------------------------

from mimo_pack.analysis.crosscorr import (  # noqa: E402
    _ccg_counts,
    ccg_jitter_significance,
    jitter_spike_times,
)
from mimo_pack.testing.data import synthetic_spikes  # noqa: E402

JIT_DURATION = 300.0
JIT_RATE = 20.0


def _modulated_train(rate, duration, mod_freq, rng, mod_depth=0.8, phase=0.0):
    """Inhomogeneous Poisson train with sinusoidal rate modulation (thinning)."""
    n = rng.poisson(rate * (1 + mod_depth) * duration)
    t = np.sort(rng.uniform(0.0, duration, n))
    p = (1 + mod_depth * np.sin(2 * np.pi * mod_freq * t + phase)) / (1 + mod_depth)
    return t[rng.uniform(size=t.size) < p]


def _counts_from_group(grp, windowsize=0.02):
    cc_norm = nap.compute_crosscorrelogram(
        grp, binsize=BIN_SIZE, windowsize=windowsize, norm=False)
    n_ref = np.array([len(grp[col[0]]) for col in cc_norm.columns])
    return pd.DataFrame((cc_norm.values * (BIN_SIZE * n_ref)).round(),
                        index=cc_norm.index, columns=cc_norm.columns)


def _detect_jitter(grp, cc, n_surrogates=200, **kwargs):
    p_ex, p_in, bl_ex, bl_in = ccg_jitter_significance(
        grp, cc, n_surrogates=n_surrogates, seed=0, **kwargs)
    return find_significant_connections(
        p_ex, p_in, bl_ex, bl_in, cc, lag_min_abs=LAG_MIN, lag_max_abs=LAG_MAX,
        min_contiguous_bins=2, alpha=0.01)


class TestCCGCountsKernel:
    def test_matches_pynapple_counts(self):
        grp = synthetic_spikes(n_units=3, duration=60.0, rate_range=(5.0, 40.0), seed=3)
        cc = _counts_from_group(grp)
        for ref, tgt in cc.columns:
            out = np.zeros(len(cc.index))
            _ccg_counts(np.sort(grp[ref].t), np.sort(grp[tgt].t), BIN_SIZE,
                        len(cc.index), out)
            np.testing.assert_array_equal(out, cc[(ref, tgt)].to_numpy())


class TestJitterSpikeTimes:
    def test_preserves_window_counts_and_sorting(self):
        rng = np.random.default_rng(0)
        t = np.sort(rng.uniform(0, 10, 500))
        w = 0.004
        surr = jitter_spike_times(nap.Ts(t=t), jitter_window=w, n_surrogates=5, seed=1)
        assert surr.shape == (5, t.size)
        orig_win = np.floor(t / w)
        for row in surr:
            assert np.all(np.diff(row) >= 0)
            np.testing.assert_array_equal(np.floor(row / w), orig_win)
        assert not np.allclose(surr[0], t)

    def test_reproducible(self):
        t = np.sort(np.random.default_rng(1).uniform(0, 5, 100))
        a = jitter_spike_times(t, n_surrogates=3, seed=7)
        b = jitter_spike_times(t, n_surrogates=3, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_empty_train(self):
        assert jitter_spike_times(np.array([]), n_surrogates=2).shape == (2, 0)

    @pytest.mark.parametrize('kwargs', [{'jitter_window': 0}, {'n_surrogates': 0}])
    def test_invalid_parameters(self, kwargs):
        with pytest.raises(ValueError):
            jitter_spike_times(np.arange(10.0), **kwargs)

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            jitter_spike_times('spikes')


@pytest.fixture(scope='module')
def coupled():
    """Pre->post excitatory coupling (1.5-3 ms delay) on top of shared 5 Hz modulation."""
    rng = np.random.default_rng(10)
    pre = _modulated_train(JIT_RATE, JIT_DURATION, 5.0, rng)
    post = _modulated_train(JIT_RATE, JIT_DURATION, 5.0, rng)
    driven = pre[rng.uniform(size=pre.size) < 0.1]
    driven = driven + rng.uniform(0.0015, 0.003, driven.size)
    post = np.sort(np.concatenate([post, driven]))
    grp = nap.TsGroup({1: nap.Ts(t=pre), 2: nap.Ts(t=post)})
    return grp, _counts_from_group(grp)


class TestCCGJitterSignificance:
    def test_output_format(self, coupled):
        grp, cc = coupled
        out = ccg_jitter_significance(grp, cc, n_surrogates=20, seed=0)
        assert len(out) == 4
        for df in out:
            assert isinstance(df, pd.DataFrame)
            assert df.index.equals(cc.index)
            assert list(df.columns) == list(cc.columns)
        p_ex, p_in, bl_ex, bl_in = out
        assert ((p_ex.values >= 0) & (p_ex.values <= 1)).all()
        assert ((p_in.values >= 0) & (p_in.values <= 1)).all()
        pd.testing.assert_frame_equal(bl_ex, bl_in)
        # Jitter preserves the target count, so surrogate totals stay close to observed
        assert abs(bl_ex.values.sum() - cc.values.sum()) / cc.values.sum() < 0.05

    def test_reproducible_and_independent_of_n_jobs(self):
        grp = synthetic_spikes(n_units=3, duration=30.0, rate_range=(5.0, 20.0), seed=4)
        cc = _counts_from_group(grp)
        a = ccg_jitter_significance(grp, cc, n_surrogates=30, seed=5, n_jobs=1)
        b = ccg_jitter_significance(grp, cc, n_surrogates=30, seed=5, n_jobs=2)
        for x, y in zip(a, b):
            pd.testing.assert_frame_equal(x, y)

    def test_excitatory_coupling_detected(self, coupled):
        grp, cc = coupled
        conns = _detect_jitter(grp, cc)
        assert len(conns) == 1
        assert conns[0]['direction'] == (1, 2)
        assert conns[0]['sign'] == 1
        assert 0.0015 <= conns[0]['lag_s'] <= 0.0030

    def test_slow_comodulation_not_detected(self):
        rng = np.random.default_rng(11)
        pre = _modulated_train(JIT_RATE, JIT_DURATION, 2.0, rng)
        post = _modulated_train(JIT_RATE, JIT_DURATION, 2.0, rng)
        grp = nap.TsGroup({1: nap.Ts(t=pre), 2: nap.Ts(t=post)})
        assert _detect_jitter(grp, _counts_from_group(grp)) == []

    def test_inhibitory_coupling_detected(self):
        rng = np.random.default_rng(12)
        pre = _modulated_train(JIT_RATE, JIT_DURATION, 5.0, rng)
        post = _modulated_train(2 * JIT_RATE, JIT_DURATION, 5.0, rng)
        idx = np.searchsorted(pre, post) - 1
        diff = np.full(post.size, np.inf)
        diff[idx >= 0] = post[idx >= 0] - pre[idx[idx >= 0]]
        post = post[~((diff > 0.001) & (diff < 0.003))]
        grp = nap.TsGroup({1: nap.Ts(t=pre), 2: nap.Ts(t=post)})
        conns = _detect_jitter(grp, _counts_from_group(grp))
        assert len(conns) == 1
        assert conns[0]['direction'] == (1, 2)
        assert conns[0]['sign'] == -1

    def test_empirical_p_values_bounded(self, coupled):
        grp, cc = coupled
        n = 50
        p_ex, p_in, _, _ = ccg_jitter_significance(
            grp, cc, n_surrogates=n, seed=0, p_method='empirical')
        for p in (p_ex.values, p_in.values):
            assert (p >= 1 / (n + 1)).all() and (p <= 1).all()

    def test_input_validation(self, coupled):
        grp, cc = coupled
        with pytest.raises(TypeError):
            ccg_jitter_significance({1: grp[1]}, cc)
        with pytest.raises(TypeError):
            ccg_jitter_significance(grp, cc.values)
        with pytest.raises(ValueError):
            ccg_jitter_significance(grp, cc, p_method='bad')
        with pytest.raises(ValueError):
            ccg_jitter_significance(grp, cc, jitter_window=-1)
        bad = cc.copy()
        bad.columns = pd.MultiIndex.from_tuples([(1, 99)])
        with pytest.raises(ValueError):
            ccg_jitter_significance(grp, bad)
