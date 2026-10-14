"""Load dclut AP files
Author: Drew B. Headley
Refactored to maximize code reuse.
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

def _configure_channels(ap_dcl, dx, dy, chans):
    """Select channels from a dclut probe object based on explicit IDs or grid spacing.

    If *chans* is ``None``, channels nearest to a regular spatial grid
    (defined by *dx* and *dy*) are selected via
    :func:`mimo_pack.analysis.probe.nearest_grid`.

    Parameters
    ----------
    ap_dcl : dclut.dclut
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
        ch_x = ap_dcl.scale_values('ch_x')
        ch_y = ap_dcl.scale_values('ch_y')
        ch = ap_dcl.scale_values('channel')
        probe_da = xr.DataArray(
            np.zeros(len(ch)),
            dims=[DIM_CHAN],
            coords={DIM_CHAN: ch, COORD_X: (DIM_CHAN, ch_x), COORD_Y: (DIM_CHAN, ch_y)},
        )
        chans = nearest_grid(probe_da, dx=dx, dy=dy)[0]

    ap_dcl.points(select={'channel': chans})
    return chans

def _filter_array(data, fs, notch_freq, notch_width, ap_band):
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
    ap_band : tuple of float or None
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
    if ap_band is not None:
        # standardizing to 'bandpass' for clarity, original had 'band' in one function
        b, a = ss.butter(3, [ap_band[0]/(fs/2), ap_band[1]/(fs/2)], btype='bandpass')
        data = ss.filtfilt(b, a, data, axis=0)
        
    return data

def _post_process(xr_obj, fs, notch_freq, notch_width, ap_band, data_description=None):
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
    ap_band : tuple of float or None
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
        # Update data with filtered values
        x_data.data = _filter_array(x_data.values, fs, notch_freq, notch_width, ap_band)
        attrs = {ATTR_FS: fs}
        if data_description:
            attrs['description'] = data_description
        return x_data.assign_attrs(**attrs)

    if isinstance(xr_obj, list):
        return [process_single(x) for x in xr_obj]
    else:
        return process_single(xr_obj)

# ==========================================
# Main Loading Functions
# ==========================================

def load_ap_full(ap_path, ap_band=(300, 6000), remove_nan_time=True,
                dx=10, dy=10, notch_freq=None, notch_width=10, chans=None):
    """Load the full time course of AP-band data from a dclut file.

    Parameters
    ----------
    ap_path : str
        Path to the AP dclut file.
    ap_band : tuple of float, optional
        ``(low, high)`` bandpass cutoff in Hz. Default is ``(300, 6000)``.
    remove_nan_time : bool, optional
        Drop samples whose time coordinate is NaN. Default is ``True``.
    dx : float, optional
        Horizontal grid spacing (µm) for automatic channel selection.
        Default is 10.
    dy : float, optional
        Vertical grid spacing (µm) for automatic channel selection.
        Default is 10.
    notch_freq : float or None, optional
        Notch filter centre frequency in Hz. Default is ``None`` (no notch).
    notch_width : float, optional
        Quality factor width of the notch filter. Default is 10.
    chans : array-like or None, optional
        Explicit channel indices. If ``None``, channels are chosen via
        ``nearest_grid``.

    Returns
    -------
    xarray.DataArray
        AP data with dimensions ``(time, channel)`` and populated
        spatial coordinates and ``attrs`` (including ``ATTR_FS``).
    """
    ap_dcl = dcl.dclut(ap_path)
    ap_dcl.reset()
    
    # Configure channels
    _configure_channels(ap_dcl, dx, dy, chans)

    # Read data
    ap_xr = ap_dcl.read(format='xarray')[0]
    ap_xr = ap_xr.sortby([COORD_X, COORD_Y])
    
    # Calculate FS
    fs = 1/np.nanmedian(np.diff(ap_xr.time.to_numpy().flatten()))

    # Remove time steps with NaN if requested
    if remove_nan_time:
        mask = ~np.isnan(ap_xr.time.values)
        ap_xr = ap_xr.isel(time=mask)

    # Apply filters and attributes
    ap_xr = _post_process(ap_xr, fs, notch_freq, notch_width, ap_band, 
                          data_description=f"AP data (full): {ap_path}")

    return ap_xr


def load_ap_intervals(ap_path, intervals, ap_band=(300, 6000), remove_nan_time=True,
                dx=10, dy=10, notch_freq=None, notch_width=10, chans=None):
    """Load AP-band data for specific time intervals from a dclut file.

    Parameters
    ----------
    ap_path : str
        Path to the AP dclut file.
    intervals : numpy.ndarray
        ``(N, 2)`` array of ``[start, end]`` time pairs (in seconds).
    ap_band : tuple of float, optional
        ``(low, high)`` bandpass cutoff in Hz. Default is ``(300, 6000)``.
    remove_nan_time : bool, optional
        Drop samples whose time coordinate is NaN. Default is ``True``.
    dx : float, optional
        Horizontal grid spacing (µm) for automatic channel selection.
        Default is 10.
    dy : float, optional
        Vertical grid spacing (µm) for automatic channel selection.
        Default is 10.
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
    ap_dcl = dcl.dclut(ap_path)
    ap_dcl.reset()

    # Configure channels
    _configure_channels(ap_dcl, dx, dy, chans)

    # Select intervals
    ap_dcl.intervals(select={'time': intervals}, select_mode='split')
    
    # Read data
    ap_xr_list = ap_dcl.read(format='xarray')
    ap_xr_list = [ap.sortby([COORD_X, COORD_Y]) for ap in ap_xr_list]
    
    # Calculate FS (from first interval)
    fs = 1/np.nanmedian(np.diff(ap_xr_list[0].time.to_numpy().flatten()))

    # Remove time steps with NaN if requested
    if remove_nan_time:
        ap_xr_list = [ap.isel(time=~np.isnan(ap.time.values)) for ap in ap_xr_list]

    # Apply filters and attributes
    ap_xr_list = _post_process(ap_xr_list, fs, notch_freq, notch_width, ap_band,
                               data_description=f"AP data (intervals): {ap_path}")

    return ap_xr_list


def load_ap_windows(ap_path, centers, pre, post, ap_band=(300, 6000), remove_nan_time=True,
                dx=10, dy=10, notch_freq=None, notch_width=10, chans=None):
    """Load epoch windows of AP data from a dclut file.

    Each window is centred on a time in *centers* and extends from
    *pre* seconds before to *post* seconds after. The returned
    DataArray is concatenated along ``DIM_EPOCH`` with a relative
    time coordinate centred on each epoch.

    Parameters
    ----------
    ap_path : str
        Path to the AP dclut file.
    centers : array-like of float
        Timestamps (seconds) at which to centre each epoch window.
    pre : float
        Duration (seconds) before each centre to include.
    post : float
        Duration (seconds) after each centre to include.
    ap_band : tuple of float, optional
        ``(low, high)`` bandpass cutoff in Hz. Default is ``(300, 6000)``.
    remove_nan_time : bool, optional
        Drop windows that overlap NaN time regions. Default is ``True``.
    dx : float, optional
        Horizontal grid spacing (µm) for automatic channel selection.
        Default is 10.
    dy : float, optional
        Vertical grid spacing (µm) for automatic channel selection.
        Default is 10.
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
        AP data with dimensions ``(time, channel, epoch)``.
        The ``time`` coordinate is relative to the epoch centre and
        the ``epoch`` coordinate holds the original centre timestamps.
    """
    ap_dcl = dcl.dclut(ap_path)
    
    # Calculate FS from raw time scale
    times = ap_dcl.scale_values('time')
    fs = int(1/np.nanmedian(np.diff(times)))

    # Calculate center indices and intervals
    center_idxs = np.array([np.nanargmin(np.abs(times - c)) for c in centers])
    pre_samples = int(pre * fs)
    post_samples = int(post * fs)
    intervals = np.column_stack((center_idxs - pre_samples,
                                 center_idxs + post_samples))

    # Configure channels
    ap_dcl.reset()
    _configure_channels(ap_dcl, dx, dy, chans)

    # Select intervals based on sample indices
    ap_dcl.intervals(select={'s0': intervals}, select_mode='split')
    ap_xr_list = ap_dcl.read(format='xarray')

    # Specific Logic: Remove windows that overlapped with NaN times entirely
    if remove_nan_time:
        nan_intervals = np.where(~np.any(np.isnan(times[intervals]), axis=1))[0]
        ap_xr_list = [ap_xr_list[i] for i in nan_intervals]
        # Adjust centers if windows were dropped
        centers = centers[nan_intervals] if len(centers) == len(intervals) else centers

    # Standardize time coordinates to be relative to center
    relative_inds = np.arange(-pre_samples, post_samples)
    relative_times = relative_inds / fs
    
    for i in range(len(ap_xr_list)):
        ap_xr_list[i] = ap_xr_list[i].assign_coords({
            'time': relative_times,
            's0': ('time', relative_inds)
        })

    # Concatenate
    ap_xr = xr.concat(ap_xr_list, dim=DIM_EPOCH).transpose(DIM_TIME, DIM_CHAN, DIM_EPOCH)
    ap_xr = ap_xr.sortby([COORD_X, COORD_Y])
    ap_xr = ap_xr.assign_coords({DIM_EPOCH: centers})

    # Apply filters and attributes
    # Note: process_signals handles axis=0 (time), which matches the transpose above
    ap_xr = _post_process(ap_xr, fs, notch_freq, notch_width, ap_band,
                          data_description=f"AP data (epoch-windowed): {ap_path}")

    return ap_xr


def load_ap_windows_raw(ap_path, centers, pre, post, n_channels, dtype=np.int16):
    """
    Load windows of AP data from the raw binary file as a numpy array
    using binary reading for maximum speed.

    Parameters
    ----------
    ap_path : str
        Path to the action potential data file (e.g., SpikeGLX .bin).
    centers : array-like
        A sequence of timestamps (in sample indices) to center the extraction windows on.
    pre : int
        The number of samples to include before each event time.
    post : int
        The number of samples to include after each event time.
    n_channels : int
        The total number of saved channels in the binary file.
    dtype : np.dtype, optional
        The data type of the samples in the binary file, by default np.int16.

    Returns
    -------
    data : np.ndarray
        A 3D numpy array of shape (time, channels, windows) containing the
        extracted AP data windows.
    """
    window_samples = pre + post
    n_windows = len(centers)
    item_size = np.dtype(dtype).itemsize
    
    # Pre-allocate the output array for efficiency
    output_data = np.zeros((window_samples, n_channels, n_windows), dtype=dtype)

    with open(ap_path, 'rb') as f:
        # Get total file size to calculate total samples
        f.seek(0, 2)  # Move to the end of the file
        total_bytes = f.tell()
        total_samples = total_bytes // (n_channels * item_size)

        # Iterate through each center and extract the corresponding window
        for i, center_idx in enumerate(centers):
            start_idx = center_idx - pre
            end_idx = center_idx + post

            # Boundary check to ensure the window is within the data range
            if start_idx >= 0 and end_idx <= total_samples:
                # Calculate byte offset to start of window
                start_byte = start_idx * n_channels * item_size
                
                # Seek to the start of the window data
                f.seek(start_byte)
                
                # Calculate number of bytes to read for one window
                bytes_to_read = window_samples * n_channels * item_size
                
                # Read the data
                data_bytes = f.read(bytes_to_read)
                
                # Convert bytes to numpy array and reshape
                window_data = np.frombuffer(data_bytes, dtype=dtype).reshape((window_samples, n_channels))
                
                # Assign to the output array
                output_data[:, :, i] = window_data
            else:
                raise ValueError(f"Window {i} with center index {center_idx} is out of bounds.")

    return output_data
