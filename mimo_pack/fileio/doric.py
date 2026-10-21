"""Load doric files
Author: Drew B. Headley
"""

import os
import h5py
import numpy as np
import xarray as xr
from dclut import create_dclut
from mimo_pack.util.hdf5 import get_h5_hierarchy
from mimo_pack.consts import ATTR_FS, DIM_TIME, DIM_CHAN

def load_doric(doric_file_path: str,
               sig_channels: str | list[str] = ['LockInAOUT02/AIN01', 'LockInAOUT04/AIN02'],
               iso_channels: str | list[str] = ['LockInAOUT01/AIN01', 'LockInAOUT03/AIN02']
               ) -> dict:
    """
    Reads a Doric file and converts it into an xarray.DataArray.

    Parameters
    ----------
    doric_file_path: string
        Path to the Doric file to load.
    sig_channels : str or list of str, optional
        Name(s) of the fiber photometry channel(s) to select as the signal
        channel(s) for `fpsig`. If None, `fpsig` contains all fiber
        photometry channels found in the file. Default is
        `['LockInAOUT02/AIN01', 'LockInAOUT04/AIN02']`.
    iso_channels : str or list of str, optional
        Name(s) of the fiber photometry channel(s) to select as the
        isosbestic control channel(s). If provided, `fpiso` is returned as a
        separate xarray.DataArray. If None, `fpiso` is returned as None.
        Default is `['LockInAOUT01/AIN01', 'LockInAOUT03/AIN02']`.

    Returns
    -------
    dict
        A dictionary with the following keys:

        - 'fpsig': xarray.DataArray
            An xarray.DataArray containing the fiber photometry signal
            channel(s) in the doric file (all fiber photometry channels if
            `sig_channels` is not specified). The data is organized into a
            2D array with the dimensions:
            - time: The time points of the recording.
            - channel: The hierarchy of keys that led to the dataset.
        - 'fpiso': xarray.DataArray or None
            An xarray.DataArray containing the isosbestic channel(s), only
            populated when `iso_channels` resolves to matching channels.
            Otherwise None. Shares the same dimensions as `fpsig`.
        - 'power': xarray.DataArray or None
            An xarray.DataArray containing the power output channels.
            The data is organized into a 2D array with the dimensions:
            - time: The time points of the recording.
            - channel: The hierarchy of keys that led to the dataset.
        - 'dio': xarray.DataArray or None
            An xarray.DataArray containing the digital output signals in
            the doric file at their native sampling (not resampled to the
            fiber photometry time points). The data is organized into a 2D
            array with the dimensions:
            - time: The native digital IO time points.
            - channel: The hierarchy of keys that led to the dataset.
        - 'analog_in': xarray.DataArray or None
            An xarray.DataArray containing the analog input signals in the
            doric file. The data is organized into a 2D array with the
            dimensions:
            - time: The native analog input time points.
            - channel: The hierarchy of keys that led to the dataset.

    Initial code generated with Gemini
    """
    # Load the Doric file using the h5py library
    data = h5py.File(doric_file_path, 'r')

    # Extract the relevant datasets from the Doric file
    data = get_h5_hierarchy(data)

    data_series = []
    hier_names = []
    is_time = []
    is_aout = []
    is_dio = []
    is_ain = []
    is_fp = []
    for (name, item_type, item_value) in data:
        if item_type == 'Dataset':
            lower_name = name.lower()
            data_series.append(item_value)
            hier_names.append(name)
            is_time.append(lower_name.endswith('time'))
            is_fp.append('lockinaout' in lower_name)
            is_aout.append('analogout' in lower_name)
            is_dio.append('digitalio' in lower_name)
            is_ain.append('analogin' in lower_name)


    
    # Convert to numpy arrays
    hier_names = np.array(hier_names)
    is_time = np.array(is_time)
    is_fp = np.array(is_fp)
    is_aout = np.array(is_aout)
    is_dio = np.array(is_dio)
    is_ain = np.array(is_ain)

    # Shorten the hierarchy names to remove the common prefix
    hier_names = [name.split('Series0001/')[-1] for name in hier_names]

    # Check which data types are present
    has_fp = np.any(is_fp)
    has_aout = np.any(is_aout)
    has_dio = np.any(is_dio)
    has_ain = np.any(is_ain)

    # Initialize outputs as None
    fp = None
    power = None
    dio = None
    analogin = None

    # Extract fiber photometry channels if present
    if has_fp:
        fp_time_indices = np.where(is_time & is_fp)[0]
        fp_idx = np.where(~is_time & is_fp)[0]
        
        if len(fp_time_indices) > 0 and len(fp_idx) > 0:
            fp_time_idx = fp_time_indices[0]
            fp_2d = np.array([data_series[i] for i in fp_idx]).T
            fp_ch = [hier_names[i] for i in fp_idx]
            fp_time_coord = np.array(data_series[fp_time_idx])
            sample_rate_fp = 1.0 / np.median(np.diff(fp_time_coord))

            fp = xr.DataArray(
                fp_2d,
                dims=(DIM_TIME, DIM_CHAN),
                coords={
                    DIM_TIME: fp_time_coord,
                    DIM_CHAN: fp_ch
                }
            )
            fp.attrs['doric_file_path'] = doric_file_path
            fp.attrs[ATTR_FS] = sample_rate_fp
            fp.attrs['description'] = f"Doric fiber photometry: {doric_file_path}"
            fp.time.attrs['units'] = 's'

    # Extract power output channels if present
    if has_aout:
        pow_time_indices = np.where(is_time & is_aout)[0]
        pow_idx = np.where(~is_time & is_aout)[0]
        
        if len(pow_time_indices) > 0 and len(pow_idx) > 0:
            pow_time_idx = pow_time_indices[0]
            power_2d = np.array([data_series[i] for i in pow_idx]).T
            power_ch = [hier_names[i] for i in pow_idx]
            power_time_coord = np.array(data_series[pow_time_idx])
            sample_rate_power = 1.0 / np.median(np.diff(power_time_coord))

            power = xr.DataArray(
                power_2d,
                dims=(DIM_TIME, DIM_CHAN),
                coords={
                    DIM_TIME: power_time_coord,
                    DIM_CHAN: power_ch
                }
            )
            power.attrs['doric_file_path'] = doric_file_path
            power.attrs[ATTR_FS] = sample_rate_power
            power.attrs['description'] = f"Doric power output: {doric_file_path}"
            power.time.attrs['units'] = 's'

    # Extract digital IO channels if present
    if has_dio:
        dio_time_indices = np.where(is_time & is_dio)[0]
        dio_idx = np.where(~is_time & is_dio)[0]
        
        if len(dio_time_indices) > 0 and len(dio_idx) > 0:
            dio_time_idx = dio_time_indices[0]
            dio_2d = np.array([data_series[i] for i in dio_idx]).T
            dio_ch = [hier_names[i] for i in dio_idx]
            dio_time_coord = np.array(data_series[dio_time_idx])
            sample_rate_dio = 1.0 / np.median(np.diff(dio_time_coord))

            dio = xr.DataArray(
                dio_2d,
                dims=(DIM_TIME, DIM_CHAN),
                coords={
                    DIM_TIME: dio_time_coord,
                    DIM_CHAN: dio_ch
                }
            )
            dio.attrs['doric_file_path'] = doric_file_path
            dio.attrs[ATTR_FS] = sample_rate_dio
            dio.attrs['description'] = f"Doric digital IO: {doric_file_path}"
            dio.time.attrs['units'] = 's'

    if has_ain:
        analog_in_time_indices = np.where(is_time & is_ain)[0]
        analog_in_idx = np.where(~is_time & is_ain)[0]

        if len(analog_in_time_indices) > 0 and len(analog_in_idx) > 0:
            analog_in_time_idx = analog_in_time_indices[0]
            analog_in_2d = np.array([data_series[i] for i in analog_in_idx]).T
            analog_in_ch = [hier_names[i] for i in analog_in_idx]
            analog_in_time_coord = np.array(data_series[analog_in_time_idx])
            sample_rate_analog_in = 1.0 / np.median(np.diff(analog_in_time_coord))

            analogin = xr.DataArray(
                analog_in_2d,
                dims=(DIM_TIME, DIM_CHAN),
                coords={
                    DIM_TIME: analog_in_time_coord,
                    DIM_CHAN: analog_in_ch
                }
            )
            analogin.attrs['doric_file_path'] = doric_file_path
            analogin.attrs[ATTR_FS] = sample_rate_analog_in
            analogin.attrs['description'] = f"Doric analog input: {doric_file_path}"
            analogin.time.attrs['units'] = 's'

    # Derive the signal and isosbestic outputs from the combined fp_xr array
    fpsig = None
    fpiso = None
    if fp is not None:
        if sig_channels is not None:
            fpsig = fp.sel({DIM_CHAN: sig_channels})
            fpsig.attrs['description'] = f"Doric fiber photometry signal: {doric_file_path}"
        else:
            fpsig = fp

        if iso_channels is not None:
            fpiso = fp.sel({DIM_CHAN: iso_channels})
            fpiso.attrs['description'] = f"Doric fiber photometry isosbestic: {doric_file_path}"

    return {
        'fpsig': fpsig,
        'fpiso': fpiso,
        'power': power,
        'dio': dio,
        'analog_in': analogin,
    }


def doric_to_binary_dclut(doric_file_path: str, out_dir: str = None, dtype: str = 'float32') -> str:
    """
    Convert a Doric file to a binary file containing all traces and a dclut JSON metadata file.

    Parameters
    ----------
    doric_file_path : str
        Path to the Doric file (.h5).
    out_dir : str, optional
        Output directory for binary and dclut files. If None, uses Doric file directory.
    dtype : str, optional
        Data type for binary file. Default is 'float32'.

    Returns
    -------
    bin_path : str
        Path to the binary file.
    dclut_path : str
        Path to the dclut JSON metadata file.
    """
    # Load Doric data
    doric_data = load_doric(doric_file_path)
    fpsig = doric_data['fpsig']
    fpiso = doric_data['fpiso']
    power_xr = doric_data['power']
    dio_xr = doric_data['dio']
    analog_in_xr = doric_data['analog_in']

    # Stack all traces along channel axis
    # Align time axes by interpolating to the fpsig time points
    time = fpsig.time.values
    traces = [fpsig.values]
    ch_names = list(fpsig.channel.values)

    # Interpolate power_xr, dio_xr, analog_in_xr, and fpiso to fpsig time axis if needed
    for xr_data in [fpiso, power_xr, dio_xr, analog_in_xr]:
        if xr_data is None:
            continue
        # Interpolate if time axes differ
        if xr_data is dio_xr and not np.array_equal(xr_data.time.values, time):
            # Digital signals take the nearest sample rather than linear blends
            traces.append(xr_data.sel({DIM_TIME: time}, method='nearest').values)
        elif not np.array_equal(xr_data.time.values, time):
            interp = np.empty((len(time), xr_data.shape[1]), dtype=xr_data.dtype)
            for i in range(xr_data.shape[1]):
                interp[:, i] = np.interp(time, xr_data.time.values, xr_data[:, i])
            traces.append(interp)
        else:
            traces.append(xr_data.values)
        ch_names.extend(list(xr_data.channel.values))

    # Concatenate all traces (time x channel)
    all_traces = np.concatenate(traces, axis=1)
    all_traces = all_traces.astype(dtype)

    # Output paths
    if out_dir is None:
        out_dir = os.path.dirname(doric_file_path)
    base_name = os.path.splitext(os.path.basename(doric_file_path))[0]
    bin_path = os.path.join(out_dir, base_name + '_alltraces.bin')
    dclut_path = os.path.join(out_dir, base_name + '_alltraces_dclut.json')

    # Write binary file
    all_traces.tofile(bin_path)

    # Create dclut metadata
    shape = (all_traces.shape[0], all_traces.shape[1])  # (time, channel)
    scales = [
        {'name': 'time', 'dim': 0, 'unit': 's', 'type': 'list', 'val': time},
        {'name': 'channel', 'dim': 1, 'unit': 'au', 'type': 'list', 'val': ch_names}
    ]
    create_dclut(bin_path, shape, dcl_path=dclut_path, dtype=dtype, 
                 data_name='doric_traces', data_unit='au', scales=scales)

    return bin_path, dclut_path
