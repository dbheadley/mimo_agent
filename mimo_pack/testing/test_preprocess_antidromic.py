"""Tests for mimo_pack.preprocess.antidromic."""

import numpy as np

from mimo_pack.consts import DIM_CHAN, DIM_TIME
from mimo_pack.preprocess import antidromic as antidromic_preprocess
from mimo_pack.preprocess.antidromic import _detect_antistim_pulses
from mimo_pack.testing import synthetic_nidq_analog


def _make_stim_channel(pulse_amplitudes):
    pulse_times = np.array([0.1, 0.3, 0.5])
    pulse_duration = 0.02
    data = synthetic_nidq_analog(
        duration=0.8,
        fs=1000.0,
        channel_names=['stim'],
        pulse_times=[pulse_times],
        pulse_duration=pulse_duration,
        pulse_amplitude=1.0,
        baseline=0.0,
        noise_std=0.0,
        seed=42,
    ).copy(deep=True)

    times = data[DIM_TIME].values
    values = data.values.copy()
    for pulse_time, amplitude in zip(pulse_times, pulse_amplitudes):
        pulse_mask = (times >= pulse_time) & (times < pulse_time + pulse_duration)
        values[pulse_mask, 0] = amplitude
    data = data.copy(data=values)
    return data.sel({DIM_CHAN: 'stim'})


def test_detect_antistim_pulses_preserves_default_lower_amplitude_pulses():
    data = _make_stim_channel([1.0, 0.8, 1.0])

    pulse_starts = _detect_antistim_pulses(data)

    np.testing.assert_allclose(pulse_starts, [0.1, 0.3, 0.5])


def test_detect_antistim_pulses_default_threshold_is_half_max():
    data = _make_stim_channel([1.0, 0.4, 1.0])

    pulse_starts = _detect_antistim_pulses(data)

    np.testing.assert_allclose(pulse_starts, [0.1, 0.5])


def test_detect_antistim_pulses_explicit_threshold_can_include_lower_pulses():
    data = _make_stim_channel([1.0, 0.4, 1.0])

    pulse_starts = _detect_antistim_pulses(data, voltage_threshold=0.3)

    np.testing.assert_allclose(pulse_starts, [0.1, 0.3, 0.5])


def test_detect_antistim_pulses_explicit_threshold_can_exclude_lower_pulses():
    data = _make_stim_channel([1.0, 0.8, 1.0])

    pulse_starts = _detect_antistim_pulses(data, voltage_threshold=0.9)

    np.testing.assert_allclose(pulse_starts, [0.1, 0.5])


def test_detect_antistim_pulses_zero_start_time_preserves_default_behavior():
    data = _make_stim_channel([1.0, 0.8, 1.0])

    pulse_starts = _detect_antistim_pulses(data, start_time_s=0.0)

    np.testing.assert_allclose(pulse_starts, [0.1, 0.3, 0.5])


def test_detect_antistim_pulses_ignores_pulses_before_start_time():
    data = _make_stim_channel([1.0, 0.8, 1.0])

    pulse_starts = _detect_antistim_pulses(data, start_time_s=0.3)

    np.testing.assert_allclose(pulse_starts, [0.3, 0.5])


def test_detect_antistim_pulses_applies_start_time_with_explicit_threshold():
    data = _make_stim_channel([1.0, 0.4, 1.0])

    pulse_starts = _detect_antistim_pulses(data, start_time_s=0.3, voltage_threshold=0.3)

    np.testing.assert_allclose(pulse_starts, [0.3, 0.5])


def test_get_antistim_times_applies_start_time_s(monkeypatch):
    data = _make_stim_channel([1.0, 0.8, 1.0])

    class FakeDclut:
        def __init__(self, path):
            self.path = path

        def reset(self):
            pass

        def points(self, selection):
            self.selection = selection

        def read(self, format):
            return [data]

    monkeypatch.setattr(antidromic_preprocess, 'dclut', FakeDclut)

    stim_df = antidromic_preprocess.get_antistim_times('unused.json', 4, start_time_s=0.3)

    assert len(stim_df) == 1
    np.testing.assert_allclose(stim_df.loc[0, 'stim_times'], [0.3, 0.5])
    assert np.isclose(stim_df.loc[0, 'start_time'], 0.3)
    assert stim_df.loc[0, 'num_pulses'] == 2


def test_get_antistim_times_returns_metadata_when_requested(monkeypatch):
    data = _make_stim_channel([1.0, 0.4, 1.0])

    class FakeDclut:
        def __init__(self, path):
            self.path = path

        def reset(self):
            pass

        def points(self, selection):
            self.selection = selection

        def read(self, format):
            return [data]

    monkeypatch.setattr(antidromic_preprocess, 'dclut', FakeDclut)

    stim_df, metadata = antidromic_preprocess.get_antistim_times(
        'unused.json',
        4,
        voltage_threshold=0.3,
        return_metadata=True,
    )

    assert len(stim_df) == 1
    np.testing.assert_allclose(stim_df.loc[0, 'stim_times'], [0.1, 0.3, 0.5])
    assert np.isclose(metadata['max_voltage'], 1.0)