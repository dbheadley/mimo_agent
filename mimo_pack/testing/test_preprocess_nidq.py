"""
Tests for mimo_pack.preprocess.nidq

All tests use synthetic data factories from mimo_pack.testing.
Real data files are never loaded in unit tests.
"""
import warnings

import numpy as np
import pytest
import xarray as xr
import pynapple as nap

from mimo_pack.consts import DIM_TIME, DIM_CHAN, ATTR_FS, ATTR_UNITS
from mimo_pack.testing import (
    synthetic_nidq_digital,
    synthetic_nidq_analog,
    synthetic_nidq,
)
from mimo_pack.preprocess.signal import (
    digital_to_events,
    digital_to_intervals,
    analog_to_events,
    analog_to_intervals,
    cross_channel_intervals,
    binarize,
    quantize,
    decode_quadrature,
    track_pulse_frequency,
    pulse_stats,
    detect_bouts,
)
from mimo_pack.preprocess.nidq import (
    extract_nidq_events,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_digital(pulses, duration=2.0, fs=1000.0, ch='ttl', pulse_dur=0.05):
    """Single-channel binary DataArray with known pulse positions."""
    return synthetic_nidq_digital(
        duration, fs, [ch], [pulses],
        pulse_duration=pulse_dur, seed=42,
    )


def _make_analog(pulses, duration=2.0, fs=1000.0, ch='photo',
                 pulse_amp=1.0, noise=0.0):
    """Single-channel analog DataArray with square pulses, no noise by default."""
    return synthetic_nidq_analog(
        duration, fs, [ch], [pulses],
        pulse_duration=0.05, pulse_amplitude=pulse_amp,
        baseline=0.0, noise_std=noise, seed=42,
    )


# ---------------------------------------------------------------------------
# TestDigitalToEvents
# ---------------------------------------------------------------------------

class TestDigitalToEvents:
    def test_rising_basic(self):
        da = _make_digital([0.5, 1.0, 1.5])
        ts = digital_to_events(da, 'ttl', edge='rising')
        assert isinstance(ts, nap.Ts)
        assert len(ts) == 3

    def test_falling_basic(self):
        da = _make_digital([0.5, 1.0, 1.5])
        ts = digital_to_events(da, 'ttl', edge='falling')
        assert isinstance(ts, nap.Ts)
        assert len(ts) == 3

    def test_both_edges(self):
        da = _make_digital([0.5])
        ts_rising = digital_to_events(da, 'ttl', edge='rising')
        ts_both = digital_to_events(da, 'ttl', edge='both')
        # 'both' should detect rising + falling: 2× rising count
        assert len(ts_both) == 2 * len(ts_rising)

    def test_empty_signal(self):
        da = _make_digital([])
        ts = digital_to_events(da, 'ttl')
        assert len(ts) == 0

    def test_all_zeros(self):
        da = synthetic_nidq_digital(1.0, 1000.0, ['ttl'], [[]], seed=42)
        ts = digital_to_events(da, 'ttl')
        assert len(ts) == 0

    def test_debounce_removes_close_events(self):
        # Two pulses 0.01 s apart — debounce of 0.05 s should remove second
        da = _make_digital([0.5, 0.505], pulse_dur=0.002)
        ts_no_debounce = digital_to_events(da, 'ttl', debounce=0.0)
        ts_debounce = digital_to_events(da, 'ttl', debounce=0.05)
        assert len(ts_debounce) < len(ts_no_debounce)

    def test_type_error(self):
        with pytest.raises(TypeError, match="Expected xr.DataArray"):
            digital_to_events(np.zeros((100, 2)), 0)

    def test_non_binary_raises(self):
        # Analog noise is not binary
        da = _make_analog([0.5], noise=0.3)
        with pytest.raises(ValueError, match="not binary"):
            digital_to_events(da, 'photo')

    def test_invalid_edge_raises(self):
        da = _make_digital([0.5])
        with pytest.raises(ValueError, match="edge must be"):
            digital_to_events(da, 'ttl', edge='up')

    def test_missing_channel_raises(self):
        da = _make_digital([0.5])
        with pytest.raises(ValueError, match="not found"):
            digital_to_events(da, 'nonexistent')

    def test_multi_channel_returns_tsgroup(self):
        da = synthetic_nidq_digital(
            2.0, 1000.0, ['ttl', 'sync'],
            [[0.5, 1.0], [0.3, 0.8]],
            pulse_duration=0.05, seed=42,
        )
        result = digital_to_events(da, ['ttl', 'sync'])
        assert isinstance(result, nap.TsGroup)
        assert len(result) == 2
        assert list(result.get_info('channel')) == ['ttl', 'sync']


# ---------------------------------------------------------------------------
# TestDigitalToIntervals
# ---------------------------------------------------------------------------

class TestDigitalToIntervals:
    def test_basic(self):
        da = _make_digital([0.3, 0.8, 1.3])
        ivs = digital_to_intervals(da, 'ttl')
        assert isinstance(ivs, nap.IntervalSet)
        assert len(ivs) == 3
        assert np.all(ivs.end > ivs.start)

    def test_signal_starts_high(self):
        """boundary='trim': unclosed leading interval is discarded."""
        t = np.linspace(0.0, 1.0, 1000)
        # starts HIGH, falls at 0.2, rises at 0.5, falls at 0.7
        sig = np.zeros(1000, dtype=np.float32)
        sig[:200] = 1.0
        sig[500:700] = 1.0
        da = xr.DataArray(
            sig[:, None],
            dims=[DIM_TIME, DIM_CHAN],
            coords={DIM_TIME: t, DIM_CHAN: ['ttl']},
            attrs={ATTR_FS: 1000.0},
        )
        ivs = digital_to_intervals(da, 'ttl', boundary='trim')
        # Only the 0.5–0.7 interval is complete
        assert len(ivs) == 1
        assert ivs.start[0] == pytest.approx(t[500], abs=0.002)

    def test_signal_ends_high(self):
        """boundary='trim': unclosed trailing interval is discarded."""
        t = np.linspace(0.0, 1.0, 1000)
        sig = np.zeros(1000, dtype=np.float32)
        sig[200:400] = 1.0  # complete pulse
        sig[700:] = 1.0      # unclosed at end
        da = xr.DataArray(
            sig[:, None],
            dims=[DIM_TIME, DIM_CHAN],
            coords={DIM_TIME: t, DIM_CHAN: ['ttl']},
            attrs={ATTR_FS: 1000.0},
        )
        ivs = digital_to_intervals(da, 'ttl', boundary='trim')
        assert len(ivs) == 1

    def test_boundary_transitions_starts_high(self):
        """boundary='transitions': a HIGH signal at t[0] opens an interval at t[0]."""
        t = np.linspace(0.0, 1.0, 1000)
        sig = np.zeros(1000, dtype=np.float32)
        sig[:200] = 1.0   # on from t[0] to t[199]
        sig[500:700] = 1.0
        da = xr.DataArray(
            sig[:, None],
            dims=[DIM_TIME, DIM_CHAN],
            coords={DIM_TIME: t, DIM_CHAN: ['ttl']},
            attrs={ATTR_FS: 1000.0},
        )
        ivs = digital_to_intervals(da, 'ttl', boundary='transitions')
        assert len(ivs) == 2
        assert ivs.start[0] == pytest.approx(t[0], abs=0.002)

    def test_boundary_transitions_ends_high(self):
        """boundary='transitions': a HIGH signal at t[-1] closes an interval at t[-1]."""
        t = np.linspace(0.0, 1.0, 1000)
        sig = np.zeros(1000, dtype=np.float32)
        sig[200:400] = 1.0  # complete pulse
        sig[700:] = 1.0      # on until t[-1]
        da = xr.DataArray(
            sig[:, None],
            dims=[DIM_TIME, DIM_CHAN],
            coords={DIM_TIME: t, DIM_CHAN: ['ttl']},
            attrs={ATTR_FS: 1000.0},
        )
        ivs = digital_to_intervals(da, 'ttl', boundary='transitions')
        assert len(ivs) == 2
        assert ivs.end[-1] == pytest.approx(t[-1], abs=0.002)

    def test_boundary_invalid_raises(self):
        da = _make_digital([0.3])
        with pytest.raises(ValueError, match="boundary"):
            digital_to_intervals(da, 'ttl', boundary='clip')

    def test_invert_returns_off_periods(self):
        # Two pulses → the gap between them inverts to a complete ON interval
        da = _make_digital([0.2, 0.7], duration=1.0, pulse_dur=0.05)
        on_ivs = digital_to_intervals(da, 'ttl', invert=False)
        off_ivs = digital_to_intervals(da, 'ttl', invert=True)
        # The gap between the two pulses is a complete OFF-period
        assert len(off_ivs) >= 1
        # Off-period start should differ from on-period start
        assert on_ivs.start[0] != pytest.approx(off_ivs.start[0], abs=0.001)

    def test_all_low_returns_empty(self):
        da = _make_digital([], duration=1.0)
        ivs = digital_to_intervals(da, 'ttl')
        assert len(ivs) == 0

    def test_multi_channel_returns_dict(self):
        da = synthetic_nidq_digital(
            2.0, 1000.0, ['ttl', 'sync'],
            [[0.5, 1.0], [0.3, 0.8]],
            pulse_duration=0.05, seed=42,
        )
        result = digital_to_intervals(da, ['ttl', 'sync'])
        assert isinstance(result, dict)
        assert set(result.keys()) == {'ttl', 'sync'}
        assert isinstance(result['ttl'], nap.IntervalSet)


# ---------------------------------------------------------------------------
# TestAnalogToEvents
# ---------------------------------------------------------------------------

class TestAnalogToEvents:
    def test_rising_basic(self):
        da = _make_analog([0.3, 0.8, 1.3])
        ts = analog_to_events(da, 'photo', threshold=0.5)
        assert isinstance(ts, nap.Ts)
        assert len(ts) == 3

    def test_falling_basic(self):
        da = _make_analog([0.3, 0.8])
        ts_r = analog_to_events(da, 'photo', threshold=0.5, edge='rising')
        ts_f = analog_to_events(da, 'photo', threshold=0.5, edge='falling')
        assert len(ts_f) == len(ts_r)

    def test_hysteresis_suppresses_noise(self):
        """With noise near threshold, hysteresis should give fewer crossings
        than no hysteresis."""
        da = _make_analog([0.5], noise=0.15, pulse_amp=1.0)
        ts_no_hyst = analog_to_events(da, 'photo', threshold=0.5,
                                      hysteresis=0.0)
        ts_hyst = analog_to_events(da, 'photo', threshold=0.5,
                                   hysteresis=0.3)
        assert len(ts_hyst) <= len(ts_no_hyst)

    def test_type_error(self):
        with pytest.raises(TypeError, match="Expected xr.DataArray"):
            analog_to_events(np.zeros((100,)), 0, threshold=0.5)

    def test_multi_channel_returns_tsgroup(self):
        da = synthetic_nidq_analog(
            2.0, 1000.0, ['ch1', 'ch2'],
            [[0.3, 0.8], [0.5, 1.2]],
            pulse_duration=0.05, pulse_amplitude=1.0,
            baseline=0.0, noise_std=0.0, seed=42,
        )
        result = analog_to_events(da, ['ch1', 'ch2'], threshold=0.5)
        assert isinstance(result, nap.TsGroup)
        assert len(result) == 2
        assert list(result.get_info('channel')) == ['ch1', 'ch2']


# ---------------------------------------------------------------------------
# TestAnalogToIntervals
# ---------------------------------------------------------------------------

class TestAnalogToIntervals:
    def test_basic(self):
        da = _make_analog([0.2, 0.7, 1.2])
        ivs = analog_to_intervals(da, 'photo', threshold=0.5)
        assert isinstance(ivs, nap.IntervalSet)
        assert len(ivs) == 3

    def test_hysteresis_reduces_spurious_intervals(self):
        da = _make_analog([0.5], noise=0.15, pulse_amp=1.0)
        ivs_no = analog_to_intervals(da, 'photo', threshold=0.5)
        ivs_h = analog_to_intervals(da, 'photo', threshold=0.5,
                                    hysteresis=0.3)
        assert len(ivs_h) <= len(ivs_no)

    def test_unclosed_leading_trimmed(self):
        """Signal starting above threshold should have leading interval trimmed."""
        t = np.linspace(0, 1.0, 1000)
        sig = np.zeros(1000, dtype=np.float32)
        sig[:200] = 2.0          # starts above threshold
        sig[500:700] = 2.0       # full pulse
        da = xr.DataArray(
            sig[:, None],
            dims=[DIM_TIME, DIM_CHAN],
            coords={DIM_TIME: t, DIM_CHAN: ['ch']},
            attrs={ATTR_FS: 1000.0},
        )
        ivs = analog_to_intervals(da, 'ch', threshold=1.0)
        assert len(ivs) == 1

    def test_empty_result(self):
        da = _make_analog([])
        ivs = analog_to_intervals(da, 'photo', threshold=0.5)
        assert len(ivs) == 0

    def test_multi_channel_returns_dict(self):
        da = synthetic_nidq_analog(
            2.0, 1000.0, ['ch1', 'ch2'],
            [[0.3, 0.8], [0.5, 1.2]],
            pulse_duration=0.05, pulse_amplitude=1.0,
            baseline=0.0, noise_std=0.0, seed=42,
        )
        result = analog_to_intervals(da, ['ch1', 'ch2'], threshold=0.5)
        assert isinstance(result, dict)
        assert set(result.keys()) == {'ch1', 'ch2'}
        assert isinstance(result['ch1'], nap.IntervalSet)


# ---------------------------------------------------------------------------
# TestBinarize
# ---------------------------------------------------------------------------

class TestBinarize:
    def test_output_dtype(self):
        da = _make_analog([0.5])
        out = binarize(da, 'photo', threshold=0.5)
        assert out.dtype == np.float32

    def test_values_are_binary(self):
        da = _make_analog([0.5], noise=0.0)
        out = binarize(da, 'photo', threshold=0.5)
        unique = np.unique(out.values)
        assert set(unique.tolist()).issubset({0.0, 1.0})

    def test_dim_time_preserved(self):
        da = _make_analog([0.4, 0.9])
        out = binarize(da, 'photo', threshold=0.5)
        assert DIM_TIME in out.dims
        assert len(out) == len(da[DIM_TIME])

    def test_attrs_preserved(self):
        da = _make_analog([0.5])
        out = binarize(da, 'photo', threshold=0.5)
        assert ATTR_FS in out.attrs

    def test_description_provenance(self):
        da = _make_analog([0.5])
        out = binarize(da, 'photo', threshold=0.5)
        assert 'binarize' in out.attrs.get('description', '')

    def test_feeds_digital_to_events(self):
        """binarize output should be accepted by digital_to_events."""
        da = _make_analog([0.5], noise=0.0)
        bin_da = binarize(da, 'photo', threshold=0.5)
        # Wrap back into a 2-D DataArray to use digital_to_events
        wrapped = bin_da.expand_dims({DIM_CHAN: ['photo']}).transpose(
            DIM_TIME, DIM_CHAN)
        wrapped.attrs[ATTR_FS] = da.attrs[ATTR_FS]
        ts = digital_to_events(wrapped, 'photo')
        assert isinstance(ts, nap.Ts)

    def test_multi_channel_returns_2d_dataarray(self):
        da = synthetic_nidq_analog(
            1.0, 1000.0, ['ch1', 'ch2'],
            [[0.3], [0.6]],
            pulse_duration=0.05, pulse_amplitude=1.0,
            baseline=0.0, noise_std=0.0, seed=42,
        )
        result = binarize(da, ['ch1', 'ch2'], threshold=0.5)
        assert isinstance(result, xr.DataArray)
        assert DIM_CHAN in result.dims
        assert DIM_TIME in result.dims
        assert list(result[DIM_CHAN].values) == ['ch1', 'ch2']


# ---------------------------------------------------------------------------
# TestQuantize
# ---------------------------------------------------------------------------

class TestQuantize:
    def test_output_dtype(self):
        da = _make_analog([0.5])
        out = quantize(da, 'photo', n_levels=4)
        assert out.dtype == np.int32

    def test_level_count(self):
        da = _make_analog([0.2, 0.5, 0.8], noise=0.0)
        out = quantize(da, 'photo', n_levels=8)
        assert out.min() >= 0
        assert out.max() <= 7

    def test_n_levels_in_attrs(self):
        da = _make_analog([0.5])
        out = quantize(da, 'photo', n_levels=4)
        assert out.attrs['n_levels'] == 4

    def test_bin_edges_in_attrs(self):
        da = _make_analog([0.5])
        out = quantize(da, 'photo', n_levels=4)
        assert 'bin_edges' in out.attrs
        assert len(out.attrs['bin_edges']) == 5  # n_levels + 1

    def test_attrs_preserved(self):
        da = _make_analog([0.5])
        out = quantize(da, 'photo', n_levels=4)
        assert ATTR_FS in out.attrs

    def test_invalid_n_levels(self):
        da = _make_analog([0.5])
        with pytest.raises(ValueError, match="n_levels must be >= 2"):
            quantize(da, 'photo', n_levels=1)

    def test_dim_time_preserved(self):
        da = _make_analog([0.5])
        out = quantize(da, 'photo', n_levels=4)
        assert DIM_TIME in out.dims

    def test_multi_channel_returns_2d_dataarray(self):
        da = synthetic_nidq_analog(
            1.0, 1000.0, ['ch1', 'ch2'],
            [[0.3], [0.6]],
            pulse_duration=0.05, pulse_amplitude=1.0,
            baseline=0.0, noise_std=0.0, seed=42,
        )
        result = quantize(da, ['ch1', 'ch2'], n_levels=4)
        assert isinstance(result, xr.DataArray)
        assert DIM_CHAN in result.dims
        assert DIM_TIME in result.dims
        assert list(result[DIM_CHAN].values) == ['ch1', 'ch2']


# ---------------------------------------------------------------------------
# TestDecodeQuadrature
# ---------------------------------------------------------------------------

def _make_quadrature_da(a_seq, b_seq, fs=1000.0):
    """Build a minimal DataArray from A and B encoder sequences."""
    n = len(a_seq)
    t = np.arange(n) / fs
    data = np.stack([np.asarray(a_seq, dtype=np.float32),
                     np.asarray(b_seq, dtype=np.float32)], axis=1)
    return xr.DataArray(
        data,
        dims=[DIM_TIME, DIM_CHAN],
        coords={DIM_TIME: t, DIM_CHAN: ['A', 'B']},
        attrs={ATTR_FS: fs},
    )


class TestDecodeQuadrature:
    # One forward Gray-code cycle: 00→01→11→10→00 = +4 counts
    _FWD_A = [0, 0, 1, 1, 0]
    _FWD_B = [0, 1, 1, 0, 0]

    def test_forward_count(self):
        da = _make_quadrature_da(self._FWD_A, self._FWD_B)
        pos = decode_quadrature(da, 'A', 'B')
        assert float(pos[-1]) == pytest.approx(4.0)

    def test_reverse_count(self):
        # Reverse: 00→10→11→01→00 = -4 counts
        da = _make_quadrature_da(list(reversed(self._FWD_A)),
                                  list(reversed(self._FWD_B)))
        pos = decode_quadrature(da, 'A', 'B')
        assert float(pos[-1]) == pytest.approx(-4.0)

    def test_resolution_scaling(self):
        da = _make_quadrature_da(self._FWD_A, self._FWD_B)
        pos = decode_quadrature(da, 'A', 'B', resolution=0.5)
        assert float(pos[-1]) == pytest.approx(2.0)

    def test_output_dim_time(self):
        da = _make_quadrature_da(self._FWD_A, self._FWD_B)
        pos = decode_quadrature(da, 'A', 'B')
        assert DIM_TIME in pos.dims
        assert len(pos) == len(self._FWD_A)

    def test_units_attr(self):
        da = _make_quadrature_da(self._FWD_A, self._FWD_B)
        pos = decode_quadrature(da, 'A', 'B')
        assert pos.attrs[ATTR_UNITS] == 'counts'

    def test_non_binary_raises(self):
        da = _make_analog([0.3], duration=0.5, noise=0.2)
        with pytest.raises(ValueError, match="not binary"):
            decode_quadrature(da, 'photo', 'photo')

    def test_stays_zero_on_no_transitions(self):
        a = [0] * 10
        b = [0] * 10
        da = _make_quadrature_da(a, b)
        pos = decode_quadrature(da, 'A', 'B')
        assert np.all(pos.values == 0.0)


# ---------------------------------------------------------------------------
# TestTrackPulseFrequency
# ---------------------------------------------------------------------------

class TestTrackPulseFrequency:
    def test_isi_mode_digital(self):
        pulses = [0.1, 0.3, 0.5, 0.7, 0.9]
        da = _make_digital(pulses, duration=1.1, pulse_dur=0.02)
        freq = track_pulse_frequency(da, 'ttl')
        assert isinstance(freq, xr.DataArray)
        assert DIM_TIME in freq.dims
        assert freq.attrs[ATTR_UNITS] == 'Hz'
        assert np.nanmax(freq.values) > 0.0

    def test_isi_mode_analog_with_threshold(self):
        pulses = [0.2, 0.5, 0.8]
        da = _make_analog(pulses, duration=1.1, noise=0.0)
        freq = track_pulse_frequency(da, 'photo', threshold=0.5)
        assert np.nanmax(freq.values) > 0.0

    def test_window_mode(self):
        pulses = [0.1, 0.3, 0.5, 0.7, 0.9]
        da = _make_digital(pulses, duration=1.1, pulse_dur=0.02)
        freq = track_pulse_frequency(da, 'ttl', window=0.3)
        # Window mode should have no NaN values
        assert not np.any(np.isnan(freq.values))
        assert np.max(freq.values) > 0.0

    def test_isi_mode_nan_fill_outside_events(self):
        pulses = [0.4, 0.6]
        da = _make_digital(pulses, duration=1.1, pulse_dur=0.02)
        freq = track_pulse_frequency(da, 'ttl')
        # Samples well before first event and after last should be NaN
        t = freq[DIM_TIME].values
        assert np.isnan(float(freq.sel({DIM_TIME: t[0]})))

    def test_single_event_is_all_nan(self):
        da = _make_digital([0.5], duration=1.0, pulse_dur=0.02)
        freq = track_pulse_frequency(da, 'ttl')
        assert np.all(np.isnan(freq.values))

    def test_units_attr(self):
        da = _make_digital([0.3, 0.6], duration=1.0, pulse_dur=0.02)
        freq = track_pulse_frequency(da, 'ttl')
        assert freq.attrs[ATTR_UNITS] == 'Hz'

    def test_multi_channel_returns_2d_dataarray(self):
        da = synthetic_nidq_digital(
            1.1, 1000.0, ['ttl', 'sync'],
            [[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]],
            pulse_duration=0.02, seed=42,
        )
        result = track_pulse_frequency(da, ['ttl', 'sync'])
        assert isinstance(result, xr.DataArray)
        assert DIM_CHAN in result.dims
        assert DIM_TIME in result.dims
        assert list(result[DIM_CHAN].values) == ['ttl', 'sync']


# ---------------------------------------------------------------------------
# TestPulseStats
# ---------------------------------------------------------------------------

class TestPulseStats:
    def test_digital_path_returns_intervalset(self):
        da = _make_digital([0.2, 0.5, 0.8])
        ivs = pulse_stats(da, 'ttl')
        assert isinstance(ivs, nap.IntervalSet)
        assert len(ivs) == 3

    def test_analog_path_returns_intervalset(self):
        da = _make_analog([0.2, 0.5, 0.8], noise=0.0)
        ivs = pulse_stats(da, 'photo', threshold=0.5)
        assert isinstance(ivs, nap.IntervalSet)
        assert len(ivs) == 3

    def test_columns_present(self):
        da = _make_digital([0.2, 0.5, 0.8])
        ivs = pulse_stats(da, 'ttl')
        for col in ('duration', 'isi', 'bout_id', 'pulse_in_bout'):
            # get_info(key) returns a pandas Series in this pynapple version
            val = ivs.get_info(col)
            assert val is not None, f"Missing column: {col}"

    def test_duration_positive(self):
        da = _make_digital([0.2, 0.5])
        ivs = pulse_stats(da, 'ttl')
        assert np.all(ivs.get_info('duration').values > 0)

    def test_last_isi_is_nan(self):
        da = _make_digital([0.2, 0.5, 0.8])
        ivs = pulse_stats(da, 'ttl')
        assert np.isnan(ivs.get_info('isi').values[-1])

    def test_bout_assignment_two_bouts(self):
        # Three closely spaced pulses, then a long gap, then three more
        close = [0.1, 0.2, 0.3]
        far   = [1.5, 1.6, 1.7]
        da = _make_digital(close + far, duration=2.5)
        ivs = pulse_stats(da, 'ttl')
        bout_ids = ivs.get_info('bout_id').values
        assert bout_ids[0] == 0
        assert bout_ids[-1] == 1

    def test_multi_channel_returns_dict(self):
        da = synthetic_nidq_digital(
            2.0, 1000.0, ['ttl', 'sync'],
            [[0.2, 0.5, 0.8], [0.3, 0.6, 0.9]],
            pulse_duration=0.05, seed=42,
        )
        result = pulse_stats(da, ['ttl', 'sync'])
        assert isinstance(result, dict)
        assert set(result.keys()) == {'ttl', 'sync'}
        assert isinstance(result['ttl'], nap.IntervalSet)


# ---------------------------------------------------------------------------
# TestDetectBouts
# ---------------------------------------------------------------------------

class TestDetectBouts:
    def test_basic_two_bouts(self):
        ts = nap.Ts(t=np.array([0.1, 0.2, 0.3, 1.5, 1.6, 1.7]))
        bouts = detect_bouts(ts, gap_threshold=0.5)
        assert isinstance(bouts, nap.IntervalSet)
        assert len(bouts) == 2

    def test_bout_boundaries(self):
        ts = nap.Ts(t=np.array([0.1, 0.2, 0.3, 1.5, 1.6, 1.7]))
        bouts = detect_bouts(ts, gap_threshold=0.5)
        assert bouts.start[0] == pytest.approx(0.1)
        assert bouts.end[0] == pytest.approx(0.3)
        assert bouts.start[1] == pytest.approx(1.5)

    def test_single_event_one_bout(self):
        ts = nap.Ts(t=np.array([0.5]))
        bouts = detect_bouts(ts, gap_threshold=0.5)
        assert len(bouts) == 1
        assert bouts.start[0] == pytest.approx(0.5)
        # end is start + epsilon (pynapple requires start < end)
        assert bouts.end[0] > bouts.start[0]

    def test_all_in_one_bout(self):
        ts = nap.Ts(t=np.array([0.1, 0.2, 0.3]))
        bouts = detect_bouts(ts, gap_threshold=2.0)
        assert len(bouts) == 1

    def test_empty_ts(self):
        ts = nap.Ts(t=np.array([]))
        bouts = detect_bouts(ts, gap_threshold=0.5)
        assert len(bouts) == 0

    def test_type_error(self):
        with pytest.raises(TypeError, match="Expected pynapple.Ts"):
            detect_bouts(np.array([0.1, 0.2]), 0.5)

    def test_invalid_gap_threshold(self):
        ts = nap.Ts(t=np.array([0.1]))
        with pytest.raises(ValueError, match="gap_threshold must be positive"):
            detect_bouts(ts, gap_threshold=0.0)


# ---------------------------------------------------------------------------
# TestExtractNidqEvents
# ---------------------------------------------------------------------------

class TestExtractNidqEvents:
    def _config(self):
        return [
            {'label': 'ttl_a', 'type': 'digital'},
            {'label': 'ttl_b', 'type': 'ttl'},
            {'label': 'photodiode', 'type': 'analog', 'threshold': 0.5},
        ]

    def test_digital_dispatch(self):
        da = synthetic_nidq(2.0, 1000.0, seed=42)
        out = extract_nidq_events(da, self._config())
        assert 'ttl_a.events' in out
        assert 'ttl_a.intervals' in out
        assert isinstance(out['ttl_a.events'], nap.Ts)
        assert isinstance(out['ttl_a.intervals'], nap.IntervalSet)

    def test_analog_dispatch(self):
        da = synthetic_nidq(2.0, 1000.0, seed=42)
        out = extract_nidq_events(da, self._config())
        assert 'photodiode.events' in out
        assert 'photodiode.intervals' in out

    def test_sync_channel_events_only(self):
        da = synthetic_nidq(2.0, 1000.0,
                            digital_channels={'sync': [0.5, 1.0]},
                            analog_channels={}, seed=42)
        out = extract_nidq_events(da, [{'label': 'sync', 'type': 'sync'}])
        assert 'sync.events' in out
        assert 'sync.intervals' not in out

    def test_other_type_skipped(self):
        da = synthetic_nidq(2.0, 1000.0, seed=42)
        config = [
            {'label': 'ttl_a', 'type': 'other'},
        ]
        out = extract_nidq_events(da, config)
        assert len(out) == 0

    def test_unknown_type_warns(self):
        da = synthetic_nidq(2.0, 1000.0, seed=42)
        config = [{'label': 'ttl_a', 'type': 'mystery_type'}]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            extract_nidq_events(da, config)
        assert any('mystery_type' in str(warning.message) for warning in w)

    def test_missing_channel_raises(self):
        da = synthetic_nidq(2.0, 1000.0, seed=42)
        config = [{'label': 'does_not_exist', 'type': 'digital'}]
        with pytest.raises(ValueError, match="not found"):
            extract_nidq_events(da, config)

    def test_analog_no_threshold_raises(self):
        da = synthetic_nidq(2.0, 1000.0, seed=42)
        config = [{'label': 'photodiode', 'type': 'analog'}]  # no threshold
        with pytest.raises(ValueError, match="threshold"):
            extract_nidq_events(da, config)

    def test_type_error_on_non_dataarray(self):
        with pytest.raises(TypeError, match="Expected xr.DataArray"):
            extract_nidq_events(np.zeros((100, 3)), [])


# ---------------------------------------------------------------------------
# TestCrossChannelIntervals
# ---------------------------------------------------------------------------

class TestCrossChannelIntervals:
    """Tests for cross_channel_intervals."""

    def _two_channel_digital(self, start_onsets, end_onsets,
                              duration=2.0, fs=1000.0):
        """Two-channel binary DataArray with named start/end channels."""
        return synthetic_nidq_digital(
            duration, fs, ['start_ch', 'end_ch'],
            [start_onsets, end_onsets],
            pulse_duration=0.05, seed=42,
        )

    def _two_channel_analog(self, start_onsets, end_onsets,
                             duration=2.0, fs=1000.0):
        """Two-channel analog DataArray with square pulses (amplitude=1, no noise)."""
        return synthetic_nidq_analog(
            duration, fs, ['start_ch', 'end_ch'],
            [start_onsets, end_onsets],
            pulse_duration=0.05, pulse_amplitude=1.0,
            baseline=0.0, noise_std=0.0, seed=42,
        )

    # --- Basic functionality -------------------------------------------------

    def test_basic_digital_digital(self):
        """One start per end: two clean intervals produced."""
        da = self._two_channel_digital([0.1, 0.7], [0.4, 1.2])
        ivs = cross_channel_intervals(da, 'start_ch', 'end_ch',
                                      on_extra='ignore')
        assert isinstance(ivs, nap.IntervalSet)
        assert len(ivs) == 2
        assert float(ivs['start'][0]) < float(ivs['end'][0])

    def test_basic_analog_analog(self):
        """Analog channels with threshold detection."""
        da = self._two_channel_analog([0.1, 0.7], [0.4, 1.2])
        ivs = cross_channel_intervals(
            da, 'start_ch', 'end_ch',
            start_threshold=0.5, end_threshold=0.5,
            on_extra='ignore',
        )
        assert isinstance(ivs, nap.IntervalSet)
        assert len(ivs) == 2

    def test_end_edge_falling(self):
        """end_edge='falling': interval ends when the end channel drops LOW."""
        da = self._two_channel_digital([0.1, 0.7], [0.3, 0.9])
        # rising start, falling end (end of pulse = falling edge)
        ivs = cross_channel_intervals(
            da, 'start_ch', 'end_ch',
            end_edge='falling', on_extra='ignore',
        )
        assert isinstance(ivs, nap.IntervalSet)
        assert len(ivs) == 2
        # ends must be after starts
        for i in range(len(ivs)):
            assert float(ivs['start'][i]) < float(ivs['end'][i])

    def test_start_edge_falling(self):
        """start_edge='falling': interval opens when start channel drops LOW."""
        da = self._two_channel_digital([0.1], [0.3])
        ivs = cross_channel_intervals(
            da, 'start_ch', 'end_ch',
            start_edge='falling', end_edge='rising',
            on_extra='ignore',
        )
        assert isinstance(ivs, nap.IntervalSet)
        # falling edge of [0.1] pulse occurs after rising edge of [0.3]? No — 
        # falling of start (~0.15) is before rising of end (0.3) → 1 interval
        assert len(ivs) == 1

    # --- pair='first' vs pair='last' -----------------------------------------

    def test_pair_first_multiple_starts(self):
        """Two starts before one end: pair='first' uses the earliest start."""
        # starts at 0.1 and 0.3, end at 0.5 → pair=(0.1, 0.5)
        da = self._two_channel_digital([0.1, 0.3], [0.5])
        ivs = cross_channel_intervals(da, 'start_ch', 'end_ch',
                                      pair='first', on_extra='ignore')
        assert len(ivs) == 1
        assert abs(float(ivs['start'][0]) - 0.1) < 0.005

    def test_pair_last_multiple_starts(self):
        """Two starts before one end: pair='last' uses the most recent start."""
        # starts at 0.1 and 0.3, end at 0.5 → pair=(0.3, 0.5)
        da = self._two_channel_digital([0.1, 0.3], [0.5])
        ivs = cross_channel_intervals(da, 'start_ch', 'end_ch',
                                      pair='last', on_extra='ignore')
        assert len(ivs) == 1
        assert abs(float(ivs['start'][0]) - 0.3) < 0.005

    def test_pair_first_and_last_differ(self):
        """pair='first' and pair='last' produce different results when
        multiple starts precede a single end."""
        da = self._two_channel_digital([0.1, 0.3], [0.5])
        ivs_first = cross_channel_intervals(da, 'start_ch', 'end_ch',
                                            pair='first', on_extra='ignore')
        ivs_last = cross_channel_intervals(da, 'start_ch', 'end_ch',
                                           pair='last', on_extra='ignore')
        assert float(ivs_first['start'][0]) < float(ivs_last['start'][0])

    # --- on_extra behaviour --------------------------------------------------

    def test_on_extra_warn_orphan_start(self):
        """Orphaned start (no following end) triggers UserWarning."""
        # 2 starts, 1 end → 1 orphan start
        da = self._two_channel_digital([0.1, 0.3], [0.5])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ivs = cross_channel_intervals(da, 'start_ch', 'end_ch',
                                          pair='first', on_extra='warn')
        assert len(ivs) == 1
        assert any('unmatched' in str(warning.message).lower() for warning in w)

    def test_on_extra_warn_orphan_end(self):
        """Orphaned end (no preceding start) triggers UserWarning."""
        # 1 start, 2 ends → 1 orphan end (second end has no start)
        da = self._two_channel_digital([0.2], [0.1, 0.5])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ivs = cross_channel_intervals(da, 'start_ch', 'end_ch',
                                          on_extra='warn')
        # end at 0.1 has no start before it; start 0.2 pairs with end 0.5
        assert any('unmatched' in str(warning.message).lower() for warning in w)

    def test_on_extra_raise(self):
        """on_extra='raise' raises ValueError when orphans exist."""
        da = self._two_channel_digital([0.1, 0.3], [0.5])
        with pytest.raises(ValueError, match='unmatched'):
            cross_channel_intervals(da, 'start_ch', 'end_ch',
                                    pair='first', on_extra='raise')

    def test_on_extra_ignore_no_warning(self):
        """on_extra='ignore' emits no 'unmatched' warning even with orphans."""
        da = self._two_channel_digital([0.1, 0.3], [0.5])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            cross_channel_intervals(da, 'start_ch', 'end_ch',
                                    pair='first', on_extra='ignore')
        unmatched_warnings = [
            x for x in w
            if issubclass(x.category, UserWarning)
            and 'unmatched' in str(x.message).lower()
        ]
        assert len(unmatched_warnings) == 0

    def test_no_orphans_no_warning(self):
        """Clean 1:1 matching: no warning is issued with on_extra='warn'."""
        da = self._two_channel_digital([0.1, 0.6], [0.4, 0.9])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ivs = cross_channel_intervals(da, 'start_ch', 'end_ch',
                                          on_extra='warn')
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0
        assert len(ivs) == 2

    # --- Edge / empty cases --------------------------------------------------

    def test_empty_starts(self):
        """No start events → empty IntervalSet, no error."""
        da = self._two_channel_digital([], [0.4])
        ivs = cross_channel_intervals(da, 'start_ch', 'end_ch',
                                      on_extra='ignore')
        assert isinstance(ivs, nap.IntervalSet)
        assert len(ivs) == 0

    def test_empty_ends(self):
        """No end events → empty IntervalSet, no error."""
        da = self._two_channel_digital([0.1], [])
        ivs = cross_channel_intervals(da, 'start_ch', 'end_ch',
                                      on_extra='ignore')
        assert isinstance(ivs, nap.IntervalSet)
        assert len(ivs) == 0

    def test_end_before_start_skipped(self):
        """An end event that has no preceding start is correctly dropped."""
        # end at 0.1 has no start before it; start 0.3 → end 0.6 is valid
        da = self._two_channel_digital([0.3], [0.1, 0.6])
        ivs = cross_channel_intervals(da, 'start_ch', 'end_ch',
                                      on_extra='ignore')
        assert len(ivs) == 1
        assert float(ivs['start'][0]) < float(ivs['end'][0])

    # --- Input validation ----------------------------------------------------

    def test_type_error_on_non_dataarray(self):
        with pytest.raises(TypeError, match='Expected xr.DataArray'):
            cross_channel_intervals(np.zeros((100, 2)), 'start_ch', 'end_ch')

    def test_missing_start_channel_raises(self):
        da = self._two_channel_digital([0.1], [0.4])
        with pytest.raises(ValueError, match='not found'):
            cross_channel_intervals(da, 'does_not_exist', 'end_ch',
                                    on_extra='ignore')

    def test_missing_end_channel_raises(self):
        da = self._two_channel_digital([0.1], [0.4])
        with pytest.raises(ValueError, match='not found'):
            cross_channel_intervals(da, 'start_ch', 'does_not_exist',
                                    on_extra='ignore')

    def test_invalid_start_edge_raises(self):
        da = self._two_channel_digital([0.1], [0.4])
        with pytest.raises(ValueError, match='start_edge'):
            cross_channel_intervals(da, 'start_ch', 'end_ch',
                                    start_edge='bad', on_extra='ignore')

    def test_invalid_end_edge_raises(self):
        da = self._two_channel_digital([0.1], [0.4])
        with pytest.raises(ValueError, match='end_edge'):
            cross_channel_intervals(da, 'start_ch', 'end_ch',
                                    end_edge='bad', on_extra='ignore')

    def test_invalid_pair_raises(self):
        da = self._two_channel_digital([0.1], [0.4])
        with pytest.raises(ValueError, match='pair'):
            cross_channel_intervals(da, 'start_ch', 'end_ch',
                                    pair='middle', on_extra='ignore')

    def test_invalid_on_extra_raises(self):
        da = self._two_channel_digital([0.1], [0.4])
        with pytest.raises(ValueError, match='on_extra'):
            cross_channel_intervals(da, 'start_ch', 'end_ch',
                                    on_extra='crash')
