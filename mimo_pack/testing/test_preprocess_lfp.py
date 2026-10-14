"""
Tests for mimo_pack.preprocess.lfp.
"""

import numpy as np
import xarray as xr

from mimo_pack.consts import ATTR_FS, DIM_CHAN, DIM_TIME
from mimo_pack.preprocess.lfp import remove_line_noise
from mimo_pack.testing.data import synthetic_lfp


def test_remove_line_noise_handles_duplicate_window_center_times():
    lfp = synthetic_lfp(duration=1.0, fs=100.0, n_chans=2, seed=42)
    times = lfp[DIM_TIME].values.copy()
    times[15] = times[10]

    lfp = lfp.assign_coords({DIM_TIME: times})
    lfp.attrs['session'] = 'duplicate-center-regression'

    cleaned = remove_line_noise(
        lfp,
        noise_freq=10.0,
        window_size_sec=0.1,
        window_step_sec=0.05,
        interp_method='cubic',
    )

    assert isinstance(cleaned, xr.DataArray)
    assert cleaned.shape == lfp.shape
    assert cleaned.dims == lfp.dims
    np.testing.assert_array_equal(cleaned[DIM_TIME], lfp[DIM_TIME])
    np.testing.assert_array_equal(cleaned[DIM_CHAN], lfp[DIM_CHAN])
    assert cleaned.attrs[ATTR_FS] == lfp.attrs[ATTR_FS]
    assert cleaned.attrs['session'] == 'duplicate-center-regression'