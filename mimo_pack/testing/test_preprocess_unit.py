"""Tests for peak-channel extraction from unit waveform metadata."""

import numpy as np
import pytest
import xarray as xr

import mimo_pack.preprocess.unit as unit
from mimo_pack.consts import COORD_X, COORD_Y, COORD_SHANK, DIM_CHAN, DIM_TIME
from mimo_pack.preprocess.unit import extract_peak_channel_waveform, waveform_spread
from mimo_pack.testing.data import synthetic_lfp, synthetic_spikes


def _waveform():
    wave = synthetic_lfp(duration=0.1, fs=1000, n_chans=3, seed=42)
    wave = wave.assign_coords({DIM_TIME: np.arange(wave.sizes[DIM_TIME]) - 20})
    wave.values[:] = 0
    wave.values[19, 0] = -47
    wave.values[21, 1] = -30
    wave.values[30:90, 2] = 20
    wave.values[70, 2] = 100
    return wave


def test_peak_searches_full_probe_near_spike_without_mutating_metadata():
    wave = _waveform()
    original = wave.copy(deep=True)
    metadata = {
        'full_waveform': wave,
        'waveform': wave.isel({DIM_CHAN: [2]}),
        'y': float(wave[COORD_Y].values[2]),
    }

    result = extract_peak_channel_waveform(metadata)

    assert isinstance(result, np.ndarray)
    assert result.shape == (wave.sizes[DIM_TIME],)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, wave.isel({DIM_CHAN: 0}).values)
    xr.testing.assert_identical(wave, original)
    assert metadata['y'] == float(wave[COORD_Y].values[2])


def test_peak_is_independent_of_channel_order_and_dimension_order():
    wave = _waveform()
    reordered = wave.isel({DIM_CHAN: [2, 0, 1]}).transpose(DIM_CHAN, DIM_TIME)
    result = extract_peak_channel_waveform({'full_waveform': reordered})
    np.testing.assert_array_equal(result, wave.isel({DIM_CHAN: 0}).values)


@pytest.mark.parametrize('include_full_none', [True, False])
def test_legacy_local_waveform_uses_amplitude_not_last_channel(include_full_none):
    wave = _waveform()
    metadata = {'waveform': wave}
    if include_full_none:
        metadata['full_waveform'] = None
    result = extract_peak_channel_waveform(metadata)
    np.testing.assert_array_equal(result, wave.isel({DIM_CHAN: 0}).values)


@pytest.mark.parametrize('spike_sample', [-5, 5])
def test_window_includes_boundary_samples(spike_sample):
    wave = _waveform()
    wave.values[:] = 0
    wave.loc[{DIM_TIME: spike_sample, DIM_CHAN: 1}] = -50
    result = extract_peak_channel_waveform({'full_waveform': wave})
    np.testing.assert_array_equal(result, wave.isel({DIM_CHAN: 1}).values)


def test_single_sample_and_channel():
    wave = _waveform().isel({DIM_TIME: [20], DIM_CHAN: [0]})
    wave.values[:] = -12
    np.testing.assert_array_equal(
        extract_peak_channel_waveform({'full_waveform': wave}), [-12.0]
    )


def test_missing_unit():
    assert extract_peak_channel_waveform(None) is None


@pytest.mark.parametrize(
    'metadata, error',
    [
        ([], TypeError),
        ({}, ValueError),
        ({'waveform': np.zeros((5, 2))}, TypeError),
    ],
)
def test_invalid_metadata_raises(metadata, error):
    with pytest.raises(error):
        extract_peak_channel_waveform(metadata)


@pytest.mark.parametrize(
    'invalid',
    [
        lambda w: w.isel({DIM_CHAN: 0}),
        lambda w: w.rename({DIM_TIME: 'sample'}),
        lambda w: w.isel({DIM_CHAN: slice(0, 0)}),
        lambda w: w.isel({DIM_TIME: slice(0, 0)}),
        lambda w: w.drop_vars(DIM_TIME),
        lambda w: w.assign_coords({DIM_TIME: w[DIM_TIME] + 1000}),
        lambda w: w.assign_coords({DIM_TIME: np.full(w.sizes[DIM_TIME], np.nan)}),
        lambda w: w.copy(data=np.full(w.shape, np.nan)),
        lambda w: w.copy(data=np.full(w.shape, np.inf)),
    ],
)
def test_invalid_waveform_raises(invalid):
    with pytest.raises(ValueError):
        extract_peak_channel_waveform({'full_waveform': invalid(_waveform())})


def test_invalid_full_waveform_does_not_silently_fall_back():
    wave = _waveform()
    with pytest.raises(TypeError):
        extract_peak_channel_waveform({'full_waveform': [], 'waveform': wave})


def _line_waveform():
    # 12 sites 20 um apart; channel 11 is an isolated outlier far from the unit
    amps = np.array([5, 5, 30, 60, 100, 100, 60, 30, 5, 5, 5, 100], dtype=float)
    values = np.zeros((21, 12))
    values[10] = -amps
    return xr.DataArray(
        values,
        dims=(DIM_TIME, DIM_CHAN),
        coords={
            DIM_TIME: np.arange(21) - 10,
            DIM_CHAN: np.arange(12),
            COORD_X: (DIM_CHAN, np.zeros(12)),
            COORD_Y: (DIM_CHAN, np.arange(12) * 20.0),
            COORD_SHANK: (DIM_CHAN, np.zeros(12, dtype=int)),
        },
    )


def test_spread_median_filter_removes_outlier_channel_and_preserves_input():
    wave = _line_waveform()
    original = wave.copy(deep=True)

    assert waveform_spread(wave) == 60
    assert waveform_spread({'full_waveform': wave}) == 60
    assert waveform_spread(wave.transpose(DIM_CHAN, DIM_TIME)) == 60
    xr.testing.assert_identical(wave, original)


def test_spread_zero_for_zero_waveform_and_few_channels():
    wave = _line_waveform()
    assert waveform_spread(wave.copy(data=np.zeros(wave.shape))) == 0
    assert waveform_spread(wave.isel({DIM_CHAN: [3]})) == 0
    assert waveform_spread(wave.isel({DIM_CHAN: [3, 4]})) == 20


def test_spread_missing_unit_and_local_fallback():
    assert np.isnan(waveform_spread(None))
    wave = _line_waveform()
    assert waveform_spread({'waveform': wave}) == waveform_spread(wave)
    assert waveform_spread({'full_waveform': None, 'waveform': wave}) == waveform_spread(wave)


@pytest.mark.parametrize(
    'invalid',
    [
        lambda w: w.drop_vars(COORD_X),
        lambda w: w.assign_coords({COORD_Y: 1}),
        lambda w: w.assign_coords({COORD_X: (DIM_CHAN, np.r_[np.nan, np.zeros(11)])}),
        lambda w: w.assign_coords({DIM_TIME: w[DIM_TIME] + 1000}),
    ],
)
def test_spread_invalid_waveforms_raise(invalid):
    with pytest.raises(ValueError):
        waveform_spread(invalid(_line_waveform()))


def test_spread_rejects_malformed_metadata():
    with pytest.raises(ValueError):
        waveform_spread({})
    with pytest.raises(TypeError):
        waveform_spread({'full_waveform': []})


def test_add_waveform_metrics_stores_full_probe_spread(monkeypatch, tmp_path):
    wave = _waveform()
    spikes = synthetic_spikes(n_units=1, duration=1, rate_range=(10, 10), seed=42)
    path = tmp_path / 'waveforms.dcl'
    path.touch()
    scales = {
        DIM_TIME: np.arange(1000) / 1000,
        COORD_X: wave[COORD_X].values,
        COORD_Y: wave[COORD_Y].values,
        COORD_SHANK: wave[COORD_SHANK].values,
    }

    class FakeDcl:
        def scale_values(self, name):
            return scales[name]

        def reset(self):
            pass

    monkeypatch.setattr(unit.dcl, 'dclut', lambda _: FakeDcl())
    monkeypatch.setattr(
        unit, '_extract_dcl_data',
        lambda *args, **kwargs: (np.repeat(wave.values[:, :, None], 2, axis=2), 1000),
    )
    monkeypatch.setattr(unit, 'waveform_snr', lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(unit, 'amp_cutoff', lambda *args, **kwargs: 0.0)

    result = unit.add_waveform_metrics(
        spikes, str(path), wave_period=[-20, 80], exclude_channels=None, verbose=False
    )
    metadata = result.metadata.iloc[0]
    assert metadata['WaveformSpread'] == pytest.approx(waveform_spread(wave))
    assert metadata['WaveformSpread'] == pytest.approx(waveform_spread(metadata['waveform']))
