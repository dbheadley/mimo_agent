"""
Tests for mimo_pack.preprocess.fiber_photometry baseline/regression/df_over_f functions.
"""

import numpy as np
import pytest
import xarray as xr

from mimo_pack.consts import ATTR_FS, ATTR_UNITS, COORD_SHANK, COORD_X, COORD_Y, DIM_CHAN, DIM_TIME
from mimo_pack.preprocess.fiber_photometry import (
    df_over_f,
    estimate_baseline,
    exponential_decay_baseline,
    isosbestic_baseline,
)
from mimo_pack.testing.data import synthetic_fiber_photometry


def _add_channel_dim(da, name):
    """Wrap a 1D (time,) DataArray with a size-1 channel dimension."""
    out = da.expand_dims({DIM_CHAN: [name]}, axis=-1)
    out = out.assign_coords({
        COORD_X: (DIM_CHAN, [0.0]),
        COORD_Y: (DIM_CHAN, [0.0]),
        COORD_SHANK: (DIM_CHAN, [0]),
    })
    out.attrs = da.attrs.copy()
    return out


class TestExponentialDecayBaseline:
    def test_basic_functionality_and_shape(self):
        signal, _ = synthetic_fiber_photometry(duration=60.0, fs=100.0, seed=0)
        baseline = exponential_decay_baseline(signal)

        assert isinstance(baseline, xr.DataArray)
        assert baseline.shape == signal.shape
        assert baseline.dims == signal.dims

    def test_monoexponential_model(self):
        signal, _ = synthetic_fiber_photometry(duration=60.0, fs=100.0, seed=1)
        baseline = exponential_decay_baseline(signal, model='monoexponential')
        assert baseline.shape == signal.shape

    def test_robust_option(self):
        signal, _ = synthetic_fiber_photometry(duration=60.0, fs=100.0, seed=2)
        baseline = exponential_decay_baseline(signal, robust=True)
        assert baseline.shape == signal.shape

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            exponential_decay_baseline(np.zeros(10))

    def test_invalid_model_raises(self):
        signal, _ = synthetic_fiber_photometry(duration=10.0, fs=100.0, seed=3)
        with pytest.raises(ValueError):
            exponential_decay_baseline(signal, model='not_a_model')

    def test_missing_time_dim_raises(self):
        bad = xr.DataArray(np.zeros(10), dims=['foo'], attrs={ATTR_FS: 100.0})
        with pytest.raises(ValueError):
            exponential_decay_baseline(bad)

    def test_coordinate_and_channel_preservation(self):
        signal, _ = synthetic_fiber_photometry(duration=30.0, fs=100.0, seed=4)
        signal = _add_channel_dim(signal, 'ch0')

        baseline = exponential_decay_baseline(signal)

        assert DIM_CHAN in baseline.dims
        assert list(baseline[DIM_CHAN].values) == ['ch0']
        assert COORD_X in baseline.coords
        assert COORD_Y in baseline.coords
        assert COORD_SHANK in baseline.coords

    def test_attrs_and_provenance(self):
        signal, _ = synthetic_fiber_photometry(duration=30.0, fs=100.0, seed=5)
        signal.attrs['description'] = 'raw signal'

        baseline = exponential_decay_baseline(signal)

        assert baseline.attrs['description'].startswith('raw signal ->')


class TestEstimateBaseline:
    def test_median_global(self):
        signal, _ = synthetic_fiber_photometry(duration=30.0, fs=100.0, seed=6)
        baseline = estimate_baseline(signal, method='median')
        assert DIM_TIME not in baseline.dims

    def test_percentile_global(self):
        signal, _ = synthetic_fiber_photometry(duration=30.0, fs=100.0, seed=7)
        baseline = estimate_baseline(signal, method='percentile', percentile=8.0)
        assert DIM_TIME not in baseline.dims

    def test_rolling_median_window(self):
        signal, _ = synthetic_fiber_photometry(duration=30.0, fs=100.0, seed=8)
        baseline = estimate_baseline(signal, method='median', window=5.0)
        assert baseline.shape == signal.shape

    def test_exponential_delegates(self):
        signal, _ = synthetic_fiber_photometry(duration=30.0, fs=100.0, seed=9)
        baseline = estimate_baseline(signal, method='exponential', exp_model='monoexponential')
        assert baseline.shape == signal.shape

    def test_invalid_method_raises(self):
        signal, _ = synthetic_fiber_photometry(duration=10.0, fs=100.0, seed=10)
        with pytest.raises(ValueError):
            estimate_baseline(signal, method='not_a_method')

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            estimate_baseline(np.zeros(10))

    def test_window_longer_than_signal_raises(self):
        signal, _ = synthetic_fiber_photometry(duration=5.0, fs=100.0, seed=11)
        with pytest.raises(ValueError):
            estimate_baseline(signal, method='median', window=100.0)

    def test_attrs_and_provenance(self):
        signal, _ = synthetic_fiber_photometry(duration=10.0, fs=100.0, seed=12)
        signal.attrs['description'] = 'raw signal'
        baseline = estimate_baseline(signal, method='median')
        assert baseline.attrs['description'].startswith('raw signal ->')

    def test_coordinate_preservation(self):
        signal, _ = synthetic_fiber_photometry(duration=10.0, fs=100.0, seed=13)
        signal = _add_channel_dim(signal, 'ch0')
        baseline = estimate_baseline(signal, method='median')
        assert list(baseline[DIM_CHAN].values) == ['ch0']
        assert COORD_X in baseline.coords


class TestIsosbesticBaseline:
    def test_basic_ols(self):
        signal, isosbestic = synthetic_fiber_photometry(duration=30.0, fs=100.0, seed=14)
        baseline = isosbestic_baseline(signal, isosbestic, method='ols')
        assert isinstance(baseline, xr.DataArray)
        assert baseline.shape == signal.shape

    def test_robust_vs_ols_with_outliers(self):
        signal, isosbestic = synthetic_fiber_photometry(duration=30.0, fs=100.0, seed=15)
        sig_vals = signal.values.copy()
        sig_vals[100] += 50.0  # inject outlier
        signal_out = signal.copy(data=sig_vals)

        baseline_ols = isosbestic_baseline(signal_out, isosbestic, method='ols')
        baseline_robust = isosbestic_baseline(signal_out, isosbestic, method='robust')

        assert baseline_ols.shape == baseline_robust.shape
        # Robust fit should be less influenced by the injected outlier
        assert not np.allclose(baseline_ols.values, baseline_robust.values)

    def test_invalid_method_raises(self):
        signal, isosbestic = synthetic_fiber_photometry(duration=10.0, fs=100.0, seed=16)
        with pytest.raises(ValueError):
            isosbestic_baseline(signal, isosbestic, method='not_a_method')

    def test_invalid_type_raises(self):
        signal, isosbestic = synthetic_fiber_photometry(duration=10.0, fs=100.0, seed=17)
        with pytest.raises(TypeError):
            isosbestic_baseline(np.zeros(10), isosbestic)
        with pytest.raises(TypeError):
            isosbestic_baseline(signal, np.zeros(10))

    def test_shape_mismatch_raises(self):
        signal, isosbestic = synthetic_fiber_photometry(duration=10.0, fs=100.0, seed=18)
        with pytest.raises(ValueError):
            isosbestic_baseline(signal, isosbestic.isel({DIM_TIME: slice(0, 10)}))

    def test_attrs_and_provenance(self):
        signal, isosbestic = synthetic_fiber_photometry(duration=10.0, fs=100.0, seed=19)
        signal.attrs['description'] = 'raw signal'
        baseline = isosbestic_baseline(signal, isosbestic)
        assert baseline.attrs['description'].startswith('raw signal ->')

    def test_coordinate_preservation(self):
        signal, isosbestic = synthetic_fiber_photometry(duration=10.0, fs=100.0, seed=20)
        signal = _add_channel_dim(signal, 'ch0')
        isosbestic = _add_channel_dim(isosbestic, 'ch0')
        baseline = isosbestic_baseline(signal, isosbestic)
        assert list(baseline[DIM_CHAN].values) == ['ch0']
        assert COORD_X in baseline.coords


class TestDfOverF:
    def test_basic_functionality_and_formula(self):
        signal, _ = synthetic_fiber_photometry(duration=30.0, fs=100.0, seed=21)
        baseline = estimate_baseline(signal, method='median', window=5.0)
        dff = df_over_f(signal, baseline)

        assert isinstance(dff, xr.DataArray)
        assert dff.shape == signal.shape
        np.testing.assert_allclose(
            dff.values, (signal.values - baseline.values) / baseline.values
        )

    def test_with_exponential_baseline(self):
        signal, _ = synthetic_fiber_photometry(duration=30.0, fs=100.0, seed=22)
        baseline = exponential_decay_baseline(signal)
        dff = df_over_f(signal, baseline)
        assert dff.shape == signal.shape

    def test_with_global_baseline_broadcast(self):
        signal, _ = synthetic_fiber_photometry(duration=30.0, fs=100.0, seed=23)
        baseline = estimate_baseline(signal, method='median')
        dff = df_over_f(signal, baseline)
        assert dff.shape == signal.shape

    def test_with_isosbestic_baseline(self):
        signal, isosbestic = synthetic_fiber_photometry(duration=30.0, fs=100.0, seed=24)
        baseline = isosbestic_baseline(signal, isosbestic)
        dff = df_over_f(signal, baseline)
        assert dff.shape == signal.shape

    def test_invalid_type_raises(self):
        signal, _ = synthetic_fiber_photometry(duration=10.0, fs=100.0, seed=25)
        baseline = estimate_baseline(signal, method='median')
        with pytest.raises(TypeError):
            df_over_f(np.zeros(10), baseline)
        with pytest.raises(TypeError):
            df_over_f(signal, np.zeros(10))

    def test_attrs_and_provenance(self):
        signal, _ = synthetic_fiber_photometry(duration=10.0, fs=100.0, seed=26)
        signal.attrs['description'] = 'raw signal'
        baseline = estimate_baseline(signal, method='median')
        dff = df_over_f(signal, baseline)
        assert dff.attrs[ATTR_UNITS] == '\u0394F/F0'
        assert dff.attrs['description'].startswith('raw signal ->')

    def test_coordinate_preservation(self):
        signal, _ = synthetic_fiber_photometry(duration=10.0, fs=100.0, seed=27)
        signal = _add_channel_dim(signal, 'ch0')
        baseline = estimate_baseline(signal, method='median', window=2.0)
        dff = df_over_f(signal, baseline)
        assert list(dff[DIM_CHAN].values) == ['ch0']
        assert COORD_X in dff.coords

