"""Load dclut LFP files
Author: Drew B. Headley
Refactored to match ap.py functionality with LFP-specific defaults.
"""

import numpy as np
import scipy.signal as ss
import dclut as dcl
import xarray as xr
from mimo_pack.analysis.probe import nearest_grid
from mimo_pack.consts import ATTR_FS, DIM_TIME, DIM_CHAN, DIM_EPOCH, COORD_X, COORD_Y

# ==========================================
# Helper Functions (Internal)
# ==========================================

def _configure_channels(lfp_dcl, dx, dy, chans):
    """Select channels from a dclut probe object based on explicit IDs or grid spacing.

    If *chans* is ``None``, channels nearest to a regular spatial grid
    (defined by *dx* and *dy*) are selected via
    :func:`mimo_pack.analysis.probe.nearest_grid`.

    Parameters
    ----------
    lfp_dcl : dclut.dclut
        An opened dclut probe data object.
    dx : float
        Horizontal grid spacing in micrometers.
    dy : float
        Vertical grid spacing in micrometers.
    chans : array-like or None
        Explicit channel indices to select. If ``None``, ``nearest_grid``
        is used.

    Returns
    -------
    numpy.ndarray
        The selected channel indices.
    """
    if chans is None:
        # Build a minimal DataArray from dclut coordinates so nearest_grid
        # receives the xr.DataArray interface it expects.
        ch_x = lfp_dcl.scale_values('ch_x')
        ch_y = lfp_dcl.scale_values('ch_y')
        ch = lfp_dcl.scale_values('channel')
        probe_da = xr.DataArray(
            np.zeros(len(ch)),
            dims=[DIM_CHAN],
            coords={DIM_CHAN: ch, COORD_X: (DIM_CHAN, ch_x), COORD_Y: (DIM_CHAN, ch_y)},
        )
        chans = nearest_grid(probe_da, dx=dx, dy=dy)[0]

    lfp_dcl.points(select={'channel': chans})
    return chans

def _filter_array(data, fs, notch_freq, notch_width, lfp_band):
    """Apply notch and/or bandpass filters to a numpy array along axis 0.

    Parameters
    ----------
    data : numpy.ndarray
        Input array with time along axis 0.
    fs : float
        Sampling frequency in Hz.
    notch_freq : float or None
        Centre frequency for the IIR notch filter. Skipped if ``None``.
    notch_width : float
        Quality factor width of the notch filter.
    lfp_band : tuple of float or None
        ``(low, high)`` cutoff frequencies for the bandpass filter.
        Skipped if ``None``.

    Returns
    -------
    numpy.ndarray
        Filtered data with the same shape as *data*.
    """
    # Notch filter
    if notch_freq is not None:
        b, a = ss.iirnotch(notch_freq, notch_width, fs=fs)
        data = ss.filtfilt(b, a, data, axis=0)

    # Bandpass filter
    if lfp_band is not None:
        b, a = ss.butter(3, [lfp_band[0]/(fs/2), lfp_band[1]/(fs/2)], btype='bandpass')
        data = ss.filtfilt(b, a, data, axis=0)
        
    return data

def _post_process(xr_obj, fs, notch_freq, notch_width, lfp_band, data_description=None):
    """Apply filtering and attach standard attributes to one or more DataArrays.

    Parameters
    ----------
    xr_obj : xarray.DataArray or list of xarray.DataArray
        Data to process.
    fs : float
        Sampling frequency in Hz.
    notch_freq : float or None
        Notch filter centre frequency. Skipped if ``None``.
    notch_width : float
        Quality factor width of the notch filter.
    lfp_band : tuple of float or None
        Bandpass filter cutoff ``(low, high)`` in Hz. Skipped if ``None``.
    data_description : str, optional
        Provenance string stored in ``attrs['description']``.

    Returns
    -------
    xarray.DataArray or list of xarray.DataArray
        Filtered data with ``ATTR_FS`` (and optionally ``description``)
        stored in ``.attrs``.
    """
    def process_single(x_data):
        """Filter a single DataArray and attach attributes."""
        x_data.data = _filter_array(x_data.values, fs, notch_freq, notch_width, lfp_band)
        attrs = {ATTR_FS: fs}
        if data_description:
            attrs['description'] = data_description
        return x_data.assign_attrs(**attrs)

    if isinstance(xr_obj, list):
        return [process_single(x) for x in xr_obj]
    else:
        return process_single(xr_obj)

def _validate_intervals(dcl_obj, intervals):
    """Remove intervals that fall outside the dclut object's time range.

    Parameters
    ----------
    dcl_obj : dclut.dclut
        An opened dclut data object whose time scale defines the valid range.
    intervals : numpy.ndarray
        ``(N, 2)`` array of ``[start, end]`` time pairs (in seconds).

    Returns
    -------
    numpy.ndarray
        A subset of *intervals* that lie within the data time range.
    """
    time_vals = dcl_obj.scale_values('time')
    t_min, t_max = np.nanmin(time_vals), np.nanmax(time_vals)
    delete_idxs = []
    for idx in range(intervals.shape[0]):
        if intervals[idx, 0] >= t_min and intervals[idx, 1] <= t_max:
            delete_idxs.append(idx)
    
    if len(delete_idxs) > 0:
        print(f"Warning: Had to remove {len(delete_idxs)} intervals outside data time range.")

    return np.delete(intervals, delete_idxs, axis=0)
            
# ==========================================
# Main Loading Functions
# ==========================================

def load_lfp_full(lfp_path, lfp_band=None, remove_nan_time=True,
                     dx=250, dy=100, notch_freq=None, notch_width=10, chans=None):
    """Load the full time course of LFP data from a dclut file.

    Parameters
    ----------
    lfp_path : str
        Path to the LFP dclut file.
    lfp_band : tuple of float or None, optional
        ``(low, high)`` bandpass cutoff in Hz. Default is ``None`` (no filter).
    remove_nan_time : bool, optional
        Drop samples whose time coordinate is NaN. Default is ``True``.
    dx : float, optional
        Horizontal grid spacing (µm) for automatic channel selection.
        Default is 250.
    dy : float, optional
        Vertical grid spacing (µm) for automatic channel selection.
        Default is 100.
    notch_freq : float or None, optional
        Notch filter centre frequency in Hz. Default is ``None``.
    notch_width : float, optional
        Quality factor width of the notch filter. Default is 10.
    chans : array-like or None, optional
        Explicit channel indices. If ``None``, channels are chosen via
        ``nearest_grid``.

    Returns
    -------
    xarray.DataArray
        LFP data with dimensions ``(time, channel)`` and populated
        spatial coordinates and ``attrs`` (including ``ATTR_FS``).
    """
    lfp_dcl = dcl.dclut(lfp_path)
    lfp_dcl.reset()
    
    # Configure channels
    _configure_channels(lfp_dcl, dx, dy, chans)

    # Read data
    lfp_xr = lfp_dcl.read(format='xarray')[0]
    lfp_xr = lfp_xr.sortby([COORD_X, COORD_Y])
    
    # Calculate FS
    fs = 1/np.nanmedian(np.diff(lfp_xr.time.to_numpy().flatten()))

    # Remove time steps with NaN if requested
    if remove_nan_time:
        mask = ~np.isnan(lfp_xr.time.values)
        lfp_xr = lfp_xr.isel(time=mask)

    # Apply filters and attributes
    lfp_xr = _post_process(lfp_xr, fs, notch_freq, notch_width, lfp_band,
                           data_description=f"LFP data (full): {lfp_path}")

    return lfp_xr


def load_lfp_intervals(lfp_path, intervals, lfp_band=None, remove_nan_time=True,
                          dx=250, dy=100, notch_freq=None, notch_width=10, chans=None):
    """Load LFP data for specific time intervals from a dclut file.

    Parameters
    ----------
    lfp_path : str
        Path to the LFP dclut file.
    intervals : numpy.ndarray
        ``(N, 2)`` array of ``[start, end]`` time pairs (in seconds).
    lfp_band : tuple of float or None, optional
        ``(low, high)`` bandpass cutoff in Hz. Default is ``None``.
    remove_nan_time : bool, optional
        Drop samples whose time coordinate is NaN. Default is ``True``.
    dx : float, optional
        Horizontal grid spacing (µm) for automatic channel selection.
        Default is 250.
    dy : float, optional
        Vertical grid spacing (µm) for automatic channel selection.
        Default is 100.
    notch_freq : float or None, optional
        Notch filter centre frequency in Hz. Default is ``None``.
    notch_width : float, optional
        Quality factor width of the notch filter. Default is 10.
    chans : array-like or None, optional
        Explicit channel indices. If ``None``, channels are chosen
        via ``nearest_grid``.

    Returns
    -------
    list of xarray.DataArray
        One DataArray per interval, each with dimensions
        ``(time, channel)`` and standard attributes.
    """
    lfp_dcl = dcl.dclut(lfp_path)
    lfp_dcl.reset()

    # Configure channels
    _configure_channels(lfp_dcl, dx, dy, chans)

    # Validate intervals
    intervals = _validate_intervals(lfp_dcl, intervals)

    # Select intervals
    lfp_dcl.intervals(select={'time': intervals}, select_mode='split')
    
    # Read data
    lfp_xr_list = lfp_dcl.read(format='xarray')
    lfp_xr_list = [lfp.sortby([COORD_X, COORD_Y]) for lfp in lfp_xr_list]
    
    # Calculate FS (from first interval)
    fs = 1/np.nanmedian(np.diff(lfp_xr_list[0].time.to_numpy().flatten()))

    # Remove time steps with NaN if requested
    if remove_nan_time:
        lfp_xr_list = [lfp.isel(time=~np.isnan(lfp.time.values)) for lfp in lfp_xr_list]

    # Apply filters and attributes
    lfp_xr_list = _post_process(lfp_xr_list, fs, notch_freq, notch_width, lfp_band,
                                data_description=f"LFP data (intervals): {lfp_path}")

    return lfp_xr_list


def load_lfp_windows(lfp_path, centers, pre, post, lfp_band=None, remove_nan_time=True,
                        dx=250, dy=100, notch_freq=None, notch_width=10, chans=None):
    """Load epoch windows of LFP data from a dclut file.

    Each window is centred on a time in *centers* and extends from
    *pre* seconds before to *post* seconds after. The returned
    DataArray is concatenated along ``DIM_EPOCH`` with a relative
    time coordinate centred on each epoch.

    Parameters
    ----------
    lfp_path : str
        Path to the LFP dclut file.
    centers : array-like of float
        Timestamps (seconds) at which to centre each epoch window.
    pre : float
        Duration (seconds) before each centre to include.
    post : float
        Duration (seconds) after each centre to include.
    lfp_band : tuple of float or None, optional
        ``(low, high)`` bandpass cutoff in Hz. Default is ``None``.
    remove_nan_time : bool, optional
        Drop windows that overlap NaN time regions. Default is ``True``.
    dx : float, optional
        Horizontal grid spacing (µm) for automatic channel selection.
        Default is 250.
    dy : float, optional
        Vertical grid spacing (µm) for automatic channel selection.
        Default is 100.
    notch_freq : float or None, optional
        Notch filter centre frequency in Hz. Default is ``None``.
    notch_width : float, optional
        Quality factor width of the notch filter. Default is 10.
    chans : array-like or None, optional
        Explicit channel indices. If ``None``, channels are chosen
        via ``nearest_grid``.

    Returns
    -------
    xarray.DataArray
        LFP data with dimensions ``(time, channel, epoch)``.
        The ``time`` coordinate is relative to the epoch centre and
        the ``epoch`` coordinate holds the original centre timestamps.
    """
    lfp_dcl = dcl.dclut(lfp_path)
    
    # Calculate FS from raw time scale
    times = lfp_dcl.scale_values('time')
    fs = int(1/np.nanmedian(np.diff(times)))

    # Calculate center indices and intervals
    center_idxs = np.array([np.nanargmin(np.abs(times - c)) for c in centers])
    pre_samples = int(pre * fs)
    post_samples = int(post * fs)
    intervals = np.column_stack((center_idxs - pre_samples,
                                 center_idxs + post_samples))

    # Configure channels
    lfp_dcl.reset()
    _configure_channels(lfp_dcl, dx, dy, chans)

    # Validate intervals
    intervals = _validate_intervals(lfp_dcl, intervals)
    
    # Select intervals based on sample indices
    lfp_dcl.intervals(select={'s0': intervals}, select_mode='split')
    lfp_xr_list = lfp_dcl.read(format='xarray')

    # Remove windows that overlapped with NaN times entirely
    if remove_nan_time:
        nan_intervals = np.where(~np.any(np.isnan(times[intervals]), axis=1))[0]
        lfp_xr_list = [lfp_xr_list[i] for i in nan_intervals]
        # Adjust centers if windows were dropped
        centers = centers[nan_intervals] if len(centers) == len(intervals) else centers

    # Standardize time coordinates to be relative to center
    relative_inds = np.arange(-pre_samples, post_samples)
    relative_times = relative_inds / fs
    
    for i in range(len(lfp_xr_list)):
        lfp_xr_list[i] = lfp_xr_list[i].assign_coords({
            'time': relative_times,
            's0': ('time', relative_inds)
        })

    # Concatenate
    lfp_xr = xr.concat(lfp_xr_list, dim=DIM_EPOCH).transpose(DIM_TIME, DIM_CHAN, DIM_EPOCH)
    lfp_xr = lfp_xr.sortby([COORD_X, COORD_Y])
    lfp_xr = lfp_xr.assign_coords({DIM_EPOCH: centers})

    # Apply filters and attributes
    lfp_xr = _post_process(lfp_xr, fs, notch_freq, notch_width, lfp_band,
                           data_description=f"LFP data (epoch-windowed): {lfp_path}")

    return lfp_xr
