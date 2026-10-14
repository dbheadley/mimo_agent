"""Tests for Doric file loading."""

import h5py
import numpy as np

from mimo_pack.consts import ATTR_FS, DIM_CHAN, DIM_TIME
from mimo_pack.fileio.doric import load_doric


def test_load_doric_reads_analog_input_variants(tmp_path):
    """AnalogIn and AnalogInput hierarchy labels produce one rich output."""
    doric_path = tmp_path / "synthetic.doric"
    fp_time = np.arange(4, dtype=float)
    analog_time = np.arange(0, 4, 0.5)

    with h5py.File(doric_path, "w") as doric_file:
        doric_file.create_dataset("Series0001/LockInAOUT01/Time", data=fp_time)
        doric_file.create_dataset(
            "Series0001/LockInAOUT01/AIN01", data=np.arange(fp_time.size)
        )
        doric_file.create_dataset("Series0001/AnalogIn/Time", data=analog_time)
        doric_file.create_dataset(
            "Series0001/AnalogIn/AI01", data=np.arange(analog_time.size)
        )
        doric_file.create_dataset(
            "Series0001/analoginput/AI02", data=np.arange(analog_time.size) + 10
        )

    result = load_doric(str(doric_path), sig_channels=None, iso_channels=None)
    fpsig = result['fpsig']
    fpiso = result['fpiso']
    power_xr = result['power']
    dio_xr = result['dio']
    analog_in_xr = result['analog_in']

    assert fpsig is not None
    assert fpiso is None
    assert power_xr is None
    assert dio_xr is None
    assert analog_in_xr.dims == (DIM_TIME, DIM_CHAN)
    np.testing.assert_array_equal(analog_in_xr[DIM_TIME], analog_time)
    assert list(analog_in_xr[DIM_CHAN].values) == [
        "AnalogIn/AI01",
        "analoginput/AI02",
    ]
    assert analog_in_xr.attrs[ATTR_FS] == 2.0
    assert analog_in_xr.attrs["description"].startswith("Doric analog input:")


def test_load_doric_splits_signal_and_isosbestic_channels(tmp_path):
    """sig_channels/iso_channels select the fpsig/fpiso outputs."""
    doric_path = tmp_path / "synthetic_fp.doric"
    fp_time = np.arange(4, dtype=float)

    with h5py.File(doric_path, "w") as doric_file:
        doric_file.create_dataset("Series0001/LockInAOUT01/Time", data=fp_time)
        doric_file.create_dataset(
            "Series0001/LockInAOUT01/AIN01", data=np.arange(fp_time.size)
        )
        doric_file.create_dataset("Series0001/LockInAOUT02/Time", data=fp_time)
        doric_file.create_dataset(
            "Series0001/LockInAOUT02/AIN01", data=np.arange(fp_time.size) + 10
        )

    result = load_doric(
        str(doric_path),
        sig_channels="LockInAOUT02/AIN01",
        iso_channels="LockInAOUT01/AIN01",
    )
    fpsig = result['fpsig']
    fpiso = result['fpiso']

    assert list(fpsig[DIM_CHAN].values) == ["LockInAOUT02/AIN01"]
    assert list(fpiso[DIM_CHAN].values) == ["LockInAOUT01/AIN01"]
    assert fpsig.attrs["description"] == f"Doric fiber photometry signal: {doric_path}"
    assert fpiso.attrs["description"] == f"Doric fiber photometry isosbestic: {doric_path}"