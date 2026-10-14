"""Load abf files
Author: Drew B. Headley
"""


import pyabf
import numpy as np
import xarray as xr
from mimo_pack.consts import ATTR_FS, DIM_TIME, DIM_CHAN, DIM_EPOCH


def load_abf_settings(abf_file_path: str) -> dict:
    """
    Reads the settings from an Axon Binary File (ABF) and returns them as a dictionary.

    Parameters
    ----------
    abf_file_path: string
        The full path to the .abf file.

    Returns
    -------
    settings: dict
        A dictionary containing the ABF settings, including protocol, number of sweeps,
        channel names, and units.
    """
    abf = pyabf.ABF(abf_file_path)
    
    settings = {
        'protocol': abf.protocol,
        'n_sweeps': abf.sweepCount,
        'channel_names': abf.adcNames,
        'channel_units': abf.adcUnits,
        'sample_rate': abf.sampleRate,
        'created': abf.abfDateTime
    }
    
    return settings


def get_clamp_mode(abf_file_path: str) -> list:
    """Detect the amplifier clamp mode for each channel using telegraph metadata.

    Reads the ``nTelegraphEnable`` and ``nTelegraphMode`` fields stored in the
    ABF header to determine whether each channel was recorded in voltage clamp,
    current clamp, or zero-current mode.  If telegraph information is not
    available for a channel the mode is reported as ``'unknown'``.

    Parameters
    ----------
    abf_file_path : str
        The full path to the .abf file.

    Returns
    -------
    list of dict
        One dictionary per ADC channel with the keys:

        - ``'channel'`` (int): channel index.
        - ``'mode'`` (str): ``'V-Clamp'``, ``'I-Clamp'``, ``'I=0'``, or
          ``'unknown'``.
        - ``'telegraph_enable'`` (bool): whether telegraph was active.
    """
    abf = pyabf.ABF(abf_file_path, loadData=False)

    TELEGRAPH_MODES = {0: "V-Clamp", 1: "I-Clamp", 2: "I=0"}

    results = []
    for ch in range(abf.channelCount):
        if abf.abfVersion["major"] == 1:
            t_enable = abf._headerV1.nTelegraphEnable[ch]
            t_mode = abf._headerV1.nTelegraphMode[ch]
        else:
            t_enable = abf._adcSection.nTelegraphEnable[ch]
            t_mode = abf._adcSection.nTelegraphMode[ch]

        if t_enable == 1 and t_mode in TELEGRAPH_MODES:
            mode = TELEGRAPH_MODES[t_mode]
        else:
            mode = "unknown"

        results.append({
            "channel": ch,
            "mode": mode,
            "telegraph_enable": t_enable == 1,
        })

    return results


def summarise_clamp_mode(channel_modes: list[str]) -> str:
    """Summarise per-channel clamp modes into a single recording-level label.

    Parameters
    ----------
    channel_modes : list of str
        Per-channel mode strings as returned by :func:`get_clamp_mode`, e.g.
        ``'V-Clamp'``, ``'I-Clamp'``, ``'I=0'``, or ``'unknown'``.
        ``'I=0'`` (zero-current clamp) is treated as ``'I-Clamp'``.

    Returns
    -------
    str
        ``'V-Clamp'``, ``'I-Clamp'``, ``'both'``, or ``'unknown'``.
    """
    modes = set()
    for m in channel_modes:
        if m == 'I=0':
            m = 'I-Clamp'
        if m in ('V-Clamp', 'I-Clamp'):
            modes.add(m)

    if modes == {'V-Clamp', 'I-Clamp'}:
        return 'both'
    elif modes == {'V-Clamp'}:
        return 'V-Clamp'
    elif modes == {'I-Clamp'}:
        return 'I-Clamp'
    else:
        return 'unknown'


def load_abf(abf_file_path: str) -> xr.DataArray:
    """
    Reads an Axon Binary File (ABF) and converts it into an xarray.DataArray.

    Parameters
    ----------
    abf_file_path: string
        The full path to the .abf file.

    Returns
    -------
    data_xr: xarray.DataArray    
        An xarray.DataArray containing the ABF data.
        The data is organized into a 3D array with the dimensions:
        - time: The time points of the recording.
        - channel: The recording channel.
        - trial: The sweep or trial number.

        Per-channel coordinates: ``ch_names``, ``ch_units``, ``clamp_mode``.

        Attributes include ``protocol``, ``fs``, ``created``, ``abf_file_path``,
        and ``clamp_mode`` — a recording-level summary of the amplifier mode:
        ``'V-Clamp'``, ``'I-Clamp'``, ``'both'``, or ``'unknown'``.

    Initial code generated with Gemini
    """
    # Load the ABF file using the pyabf library
    abf = pyabf.ABF(abf_file_path)

    # Get the dimensions of the data from the ABF header
    n_sweeps = abf.sweepCount
    n_channels = abf.channelCount
    n_points = abf.sweepPointCount

    # Create an empty 3D numpy array to store all the data
    # The dimensions are ordered as (trial, channel, time)
    all_data = np.zeros((n_points, n_channels, n_sweeps))

    # Iterate over each sweep (trial) and channel to fill the numpy array
    for sweep_number in range(n_sweeps):
        for channel_number in range(n_channels):
            # Set the sweep and channel to read the data
            abf.setSweep(sweepNumber=sweep_number, channel=channel_number)
            # Store the data in the corresponding location in the 3D array
            all_data[:, channel_number, sweep_number] = abf.sweepY

    # Create the coordinate arrays for the xarray DataArray
    # The time coordinates are taken from the last sweep (they are the same for all sweeps)
    time_coords = abf.sweepX
    # The channel coordinates are a simple range of integers
    channel_coords = np.arange(n_channels)
    # The trial coordinates are also a simple range of integers
    trial_coords = np.arange(n_sweeps)
    # Channel names — dtype=object (VL strings) for netCDF4 compatibility
    names_coords = np.array(abf.adcNames, dtype=object)
    # Channel units — dtype=object (VL strings) for netCDF4 compatibility
    units_coords = np.array(abf.adcUnits, dtype=object)
    # Clamp mode per channel via telegraph metadata
    # Key by channel index so lookup is order-independent
    _clamp_info = {r["channel"]: r["mode"] for r in get_clamp_mode(abf_file_path)}
    clamp_modes = np.array([_clamp_info[ch] for ch in channel_coords], dtype=object)

    # Create the xarray DataArray
    # We provide the 3D numpy array, the dimension names, and the coordinate arrays
    data_xr = xr.DataArray(
        all_data,
        dims=(DIM_TIME, DIM_CHAN, DIM_EPOCH),
        coords={
            DIM_TIME: time_coords,
            DIM_CHAN: channel_coords,
            DIM_EPOCH: trial_coords,
            "ch_names": (DIM_CHAN, names_coords),
            "ch_units": (DIM_CHAN, units_coords),
            "clamp_mode": (DIM_CHAN, clamp_modes),
        }
    )
    
    # Summarise clamp mode across channels as a single recording-level attribute.
    recording_clamp_mode = summarise_clamp_mode(clamp_modes)

    # Add attributes to the DataArray for metadata
    data_xr.attrs['abf_file_path'] = abf_file_path
    data_xr.attrs['abf_version'] = abf.abfVersionString
    data_xr.attrs['protocol'] = abf.protocol
    data_xr.attrs[ATTR_FS] = abf.sampleRate
    data_xr.attrs['created'] = abf.abfDateTime.isoformat()
    data_xr.attrs['clamp_mode'] = recording_clamp_mode
    data_xr.attrs['description'] = f"ABF file: {abf_file_path}"
    data_xr[DIM_TIME].attrs['units'] = 's'

    return data_xr

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    try:
        abf_file_path = os.path.join('..','..','test_data','invitro','21520000.abf')
        
        # Convert the ABF file to an xarray DataArray
        abf_xr = load_abf(abf_file_path)

        # Print the resulting xarray DataArray to see its structure
        print("Successfully converted ABF to xarray.DataArray:")
        print(abf_xr)
        
        volt_chans = np.where(abf_xr.ch_units.values == 'mV')[0]
        fig, ax = plt.subplots(volt_chans.size,1, sharex=True, sharey=True)
        for i, ch in enumerate(volt_chans):
            ax[i].plot(abf_xr.time, abf_xr.isel(channel=ch), color='k')
            ax[i].set_title(f'Channel {ch} - {abf_xr.ch_names.values[ch]}')
            ax[i].set_ylabel(abf_xr.ch_units.values[ch])
            ax[i].set_xlabel(f'Time ({abf_xr.time.attrs["units"]})')

        fig.tight_layout()
        plt.show()
        
        
    except Exception as e:
        print(f"Error details: {e}")
