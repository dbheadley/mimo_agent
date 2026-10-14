# LFP preprocessing related functions
# Author: Drew Headley
# Created: 2024-05-19

import os
import shutil
import numpy as np
import pandas as pd
import xarray as xr
import dclut as dcl
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from scipy.interpolate import CubicSpline
from dclut import create_dclut
from ..fileio.spikeglx import read_meta, get_chanmap, get_geommap
from ..math.curvefitting import piecewise_table_monotonic
from ..consts import DIM_TIME, ATTR_FS

def make_lfp_file_dclut(dcl_path, lfp_path, lfp_cutoff=200, lfp_fs=1000, sync = {'channel': [384]},
                        time_scale='time', verbose=False):
    """
    Makes LFP file from binary with a dclut metadata file. Overwrites existing LFP file.

    Parameters
    ----------
    dcl_path : str
        Path to dclut file.
    lfp_path : str
        Path to use when saving the LFP binary file. The corresponding dclut file will be saved with
        the same name but with a _dclut.json extension.

    Optional
    --------
    lfp_cutoff : numeric
        Cutoff frequency for low-pass filter in Hz. Default is 200 Hz.
    lfp_fs : numeric
        Sampling frequency of LFP data in Hz. Default is 1000 Hz.
    sync : dict
        Dictionary that can be passed to dclut with the channel number 
        for the sync signal. If no sync signal is present, give as
        None. Default is {'channel': [384]}.
    time_scale : str
        Name of the time scale in the dclut file. Default is 'time'.
    verbose : bool
        If True, print progress. Default is False.

    Returns
    -------
    lfp_dcl_path : str
        Path to LFP dclut file.
    """

    # Check if the binary file exists
    if not os.path.exists(dcl_path):
        raise FileNotFoundError('Dclut file {} not found'.format(dcl_path))
    
    if sync is None:
        sync_chan = []
    else:
        sync_chan = sync['channel']

    # Create dclut object for reading binary data
    dcl_data = dcl.dclut(dcl_path)
    times = dcl_data.scale_values(time_scale)
    time_dim = dcl_data.dcl['scales'][time_scale]['dim']
    ind_scale = 's' + str(time_dim)
    ind_len = dcl_data.shape[time_dim]

    # determine which time points to keep for downsampling
    fs = np.round(1/np.nanmedian(np.diff(times))).astype(int)
    keep_step = int(fs/lfp_fs)
    down_inds = np.arange(0, ind_len, keep_step)
    chunks = range(0, ind_len, fs*60)
    if chunks[-1] != ind_len:
        chunks = np.append(chunks, ind_len)
    
    # initialize binary data to write
    lfpf = open(lfp_path, mode='wb')

    if verbose:
        iter_chunks = tqdm(range(len(chunks)-1))
    else:
        iter_chunks = range(len(chunks)-1)
    
    # iterate through chunks and write out to LFP file
    lfp_times = []
    for i in iter_chunks:
        # read in chunk of data
        dcl_data.reset()

        # include a buffer of 1sec time points on either side of the chunk
        # to minimize filtering edge effects
        if chunks[i]-fs > 0:
            low_edge = fs
        else:
            low_edge = 0
        
        if chunks[i+1]+fs < ind_len:
            high_edge = fs
        else:
            high_edge = 0
        
        dcl_data.intervals({ind_scale: [chunks[i]-low_edge, chunks[i+1]+high_edge]})
        bin_data = dcl_data.read(format='xarray')[0]
        chunk_times = bin_data[time_scale].values
        chunk_inds = bin_data[ind_scale].values

        if len(chunk_times) > 10:
            # calculate LFP
            lfp_data = calc_lfp(bin_data.data, fs, lfp_cutoff=lfp_cutoff, 
                                down_factor=1, offset=0, ignore_chans=sync_chan)
            
            # remove buffer time points
            # If high_edge is 0, we want to slice to the end (None)
            # If high_edge > 0, we want to slice off that many samples from the end
            end_idx = -high_edge if high_edge > 0 else None
            
            lfp_data = lfp_data[low_edge:end_idx, :]
            chunk_times = chunk_times[low_edge:end_idx]
            chunk_inds = chunk_inds[low_edge:end_idx]

            # # remove buffer time points
            # lfp_data = lfp_data[low_edge:-(high_edge-1),:]
            # chunk_times = chunk_times[low_edge:-(high_edge-1)]
            # chunk_inds = chunk_inds[low_edge:-(high_edge-1)]

            # lfp data to keep for downsampling
            down_bool = np.where(np.isin(chunk_inds, down_inds))[0]
            lfp_data = lfp_data[down_bool,:]
            lfp_times.append(chunk_times[down_bool])
            lfpf.write(lfp_data.tobytes())
    lfpf.close()

    # write new dclut file for the LFP by copying the original and changing the time fields
    lfp_dcl_path = lfp_path.replace('.bin', '_dclut.json')
    lfp_times = np.concatenate(lfp_times).flatten()
    dcl_data.reset()
    dcl_data.dcl['file']['name'] = os.path.split(lfp_path)[1]
    dcl_data.dcl['file']['shape'][time_dim] = len(lfp_times)
    dcl_data.dcl['scales'][time_scale]['type'] = 'table'

    # simplfy the time values using piecewise monotonic function, preserving NaNs
    nan_inds = np.where(np.isnan(lfp_times))[0]
    nan_times = np.column_stack((nan_inds, np.full(len(nan_inds), np.nan)))
    pw_lfp_times = piecewise_table_monotonic(lfp_times, thresh=0.001)

    pw_lfp_times = np.vstack((pw_lfp_times, nan_times))
    pw_lfp_times = pw_lfp_times[np.argsort(pw_lfp_times[:,0]),:]

    dcl_data.dcl['scales'][time_scale]['values'] = pw_lfp_times
    dcl_data.save(path=lfp_dcl_path)
    
    return lfp_path


def make_lfp_dclut(ap_dclut_path, lfp_path, lfp_fs=1000, sync={'channel': [384]}, time_scale='time'):
    """
    Create an LFP dclut JSON from an existing AP dclut, without recreating the LFP binary.

    Useful when the LFP binary already exists but its dclut is missing.  The
    time mapping is derived by reading the AP dclut's time scale and taking
    every ``fs / lfp_fs``-th sample — matching the downsampling applied by
    :func:`make_lfp_file_dclut`.

    Parameters
    ----------
    ap_dclut_path : str
        Path to the AP dclut JSON file.
    lfp_path : str
        Path to the (existing) LFP binary file.  The dclut will be written
        alongside it as ``lfp_path.replace('.bin', '_dclut.json')``.
    lfp_fs : numeric, optional
        LFP sampling rate in Hz.  Must match the rate used when the binary
        was originally created.  Default is 1000 Hz.
    sync : dict or None, optional
        Unused; present for API consistency with :func:`make_lfp_file_dclut`.
    time_scale : str, optional
        Name of the time scale in the AP dclut.  Default is ``'time'``.

    Returns
    -------
    lfp_dcl_path : str
        Path to the written LFP dclut file.
    """
    if not os.path.exists(ap_dclut_path):
        raise FileNotFoundError(f'AP dclut file not found: {ap_dclut_path}')

    dcl_data = dcl.dclut(ap_dclut_path)
    times = dcl_data.scale_values(time_scale)
    time_dim = dcl_data.dcl['scales'][time_scale]['dim']

    fs = np.round(1 / np.nanmedian(np.diff(times))).astype(int)
    keep_step = int(fs / lfp_fs)
    lfp_times = times[::keep_step]

    lfp_dcl_path = lfp_path.replace('.bin', '_dclut.json')
    dcl_data.reset()
    dcl_data.dcl['file']['name'] = os.path.split(lfp_path)[1]
    dcl_data.dcl['file']['shape'][time_dim] = len(lfp_times)
    dcl_data.dcl['scales'][time_scale]['type'] = 'table'

    nan_inds = np.where(np.isnan(lfp_times))[0]
    nan_times = np.column_stack((nan_inds, np.full(len(nan_inds), np.nan)))
    pw_lfp_times = piecewise_table_monotonic(lfp_times, thresh=0.001)

    pw_lfp_times = np.vstack((pw_lfp_times, nan_times))
    pw_lfp_times = pw_lfp_times[np.argsort(pw_lfp_times[:, 0]), :]

    dcl_data.dcl['scales'][time_scale]['values'] = pw_lfp_times
    dcl_data.save(path=lfp_dcl_path)

    return lfp_dcl_path


def make_lfp_file_spikeglx(bin_path, lfp_cutoff=200, lfp_fs=1000, suffix='lfp', verbose=False):
    """
    Makes LFP file from raw SpikeGLX binary. Overwrites existing LFP file.

    Parameters
    ----------
    bin_path : str
        Path to binary file.
    

    Optional
    --------
    lfp_cutoff : numeric
        Cutoff frequency for low-pass filter in Hz. Default is 200 Hz.
    lfp_fs : numeric
        Sampling frequency of LFP data in Hz. Default is 1000 Hz.
    suffix : str
        Suffix for LFP file. Default is 'lfp'.
    verbose : bool
        If True, print progress. Default is True.

    Returns
    -------
    lfp_path : str
        Path to LFP file.
    """

    # Check if the binary file exists
    if not os.path.exists(bin_path):
        raise FileNotFoundError('Binary file {} not found'.format(bin_path))
    
    # Get the meta data
    meta = read_meta(bin_path)

    fs = meta['imSampRate']
    chan_num = meta['nSavedChans']
    sy_yes = meta['acqApLfSy'][2]
    byte_num = 2

    if sy_yes:
        sy_chan = chan_num-1
    else:
        sy_chan = []

    # open binary file, read only
    binf = open(bin_path, mode='rb')
    bin_bytes = os.path.getsize(bin_path)

    # test if fs is evenly divisible by lfp_fs
    down_factor = fs/lfp_fs
    if down_factor % 1 != 0:
        raise ValueError('Sampling rate of {} Hz is not evenly divisible by ' 
                         'by LFP sampling rate of {} Hz'.format(fs, lfp_fs))
    else:
        down_factor = int(down_factor)

    chunk_dur = 60*fs # number of time points to convert at a time
    step_bytes = byte_num*chan_num # number of bytes per time point
    bin_dur = bin_bytes//step_bytes # number of time points in binary file

    # initialize binary data to write
    lfp_path = bin_path.replace('.ap.bin', '.{}.bin'.format(suffix))
    lfpf = open(lfp_path, mode='wb')

    # convert to LFP, in 1 minute chunks, 
    chunks = np.arange(0, bin_dur, chunk_dur).astype(np.int64)
    if chunks[-1] != bin_dur:
        chunks = np.append(chunks, bin_dur)

    # downsampled LFP time points to keep
    lfp_keep = np.arange(0, bin_dur, down_factor).astype(np.int64)

    if verbose:
        iter_chunks = tqdm(range(len(chunks)-1))
    else:
        iter_chunks = range(len(chunks)-1)
    
    # iterate through chunks
    for i in iter_chunks:
        # read in chunk of data
        binf.seek(chunks[i]*step_bytes)
        chunk_len = chunks[i+1]-chunks[i]
        bin_data = np.fromfile(binf, dtype='int16', count=chunk_len*chan_num)
        bin_data = bin_data.reshape((chunk_len, chan_num))

        # offset is the first index in lfp_keep that is greater than or 
        # equal to chunks[i]
        offset = lfp_keep[np.nonzero(lfp_keep >= chunks[i])[0][0]]-chunks[i]

        # calculate LFP
        lfp_data = calc_lfp(bin_data, fs, lfp_cutoff=lfp_cutoff, 
                            down_factor=down_factor, offset=offset, ignore_chans=sy_chan)
        
        lfpf.write(lfp_data.tobytes())

    binf.close()
    lfpf.close()
    return lfp_path

def dclut_from_meta_lfp(lfp_path, dcl_path=None, lfp_fs=1000):
    """
    Create a dclut json file from the .meta file associated with a SpikeGLX. bin file
    that has been converted to LFP.

    Parameters
    ----------
    lfp_path : str
        Path to the LFP binary file
    
    Optional
    --------
    dcl_path : str
        Path to save the dclut json file. If not provided, the file will be saved in
        the same directory as the binary file with the same name but with a _dclut.json extension.
    
    Returns
    -------
    dcl_path : str
        Path to the dclut json file
    """
    
    if dcl_path is None:
        dcl_path = lfp_path.replace('.bin', '_dclut.json')
    
    bin_path = lfp_path.replace('.lfp.bin', '.ap.bin')
    meta = read_meta(bin_path)
    chmap = get_chanmap(bin_path)
    
    chan_num = meta['nSavedChans']

    gmap = get_geommap(bin_path)
    chan_props = chmap.merge(gmap, left_index=True, right_index=True, how='outer')
    scales = [{'name': 'time', 'dim': 0, 'unit': 'seconds', 
                'type': 'linear', 'val': [1/lfp_fs, 0]}, 
                {'name': 'channel', 'dim': 1, 'unit': 'none', 
                'type': 'index', 'val': None}, 
                {'name': 'ch_name', 'dim': 1, 'unit': 'none', 
                'type': 'list', 'val': chan_props['name'].values}, 
                {'name': 'ch_order', 'dim': 1, 'unit': 'none', 
                'type': 'list', 'val': chan_props['order'].values}, 
                {'name': 'ch_x', 'dim': 1, 'unit': 'um', 
                'type': 'list', 'val': chan_props['x'].values}, 
                {'name': 'ch_y', 'dim': 1, 'unit': 'um', 
                'type': 'list', 'val': chan_props['y'].values}, 
                {'name': 'ch_shank', 'dim': 1, 'unit': 'none', 
                'type': 'list', 'val': chan_props['shank'].values}]

    dcl_path = create_dclut(lfp_path, [-1, chan_num], dcl_path=dcl_path, 
                            dtype='int16', data_name='data', data_unit='au', 
                            scales = scales)
    return dcl_path

def calc_lfp(raw_data, fs, lfp_cutoff=200, down_factor=30, offset=0,
             ignore_chans=[]):
    """
    Calculate the LFP from raw data.
    
    Parameters
    ----------
    raw_data : np.ndarray
        Raw data with time as the first axis and channels as the second axis.
    fs : numeric
        Sampling frequency of the raw data.
    lfp_cutoff : numeric, optional
        Cutoff frequency for low-pass filter in Hz. Default is 200 Hz.
    down_factor : numeric, optional
        Factor for downsampling the LFP data, i.e. number of skipped
        samples. Default is 30.
    offset : numeric, optional
        Offset for downsampling the LFP data. Default is 0.
    ignore_chans : list of int, optional
        List of channel indices to exclude from LFP filtering. 
        Default is [].
        
    Returns
    -------
    lfp_data : np.ndarray
        LFP data.
    """

    # determine channels to process
    chan_num = raw_data.shape[1]
    chans = np.where(np.isin(np.arange(chan_num), ignore_chans, invert=True))[0]

    # create butterworth filter
    b, a = butter(2, lfp_cutoff/(fs/2), 'low')

    # initialize LFP data
    lfp_data = raw_data.copy()
    
    # filter the data
    lfp_data[:, chans] = filtfilt(b, a, raw_data[:, chans], axis=0)

    # downsample the data
    lfp_data = lfp_data[offset::down_factor]

    return lfp_data


def remove_line_noise(lfp, noise_freq, window_size_sec=1.0, window_step_sec=0.025,
                     interp_method='cubic'):
    """
    Remove line noise from LFP data using a sliding-window sine-fit algorithm.

    For each channel, a sine and cosine at ``noise_freq`` are fitted to successive
    overlapping windows of the signal via ordinary least squares.  The resulting
    amplitude envelopes are interpolated to the full time axis and subtracted from
    the raw signal.  This approach handles slowly drifting noise amplitude better
    than a static notch filter.

    Parameters
    ----------
    lfp : xarray.DataArray
        Input LFP data.  Must have a ``time`` dimension (``DIM_TIME``).  The time
        coordinate values must be in seconds.  If ``lfp.attrs`` contains the key
        ``'fs'`` (``ATTR_FS``), that value is used as the sampling rate; otherwise
        ``fs`` is inferred from the median spacing of the time coordinate.
    noise_freq : float
        Frequency of the line noise to remove in Hz (e.g. 60 for North America,
        50 for Europe).
    window_size_sec : float, optional
        Half-width of each fitting window in seconds.  The full window spans
        ``2 * window_size_sec`` seconds.  Default is 1.0.
    window_step_sec : float, optional
        Step between successive window centers in seconds.  Smaller values give
        smoother coefficient trajectories at the cost of more computation.
        Default is 0.025.
    interp_method : {'cubic', 'linear'}, optional
        Interpolation method for the amplitude envelopes between window centers.
        ``'cubic'`` uses ``scipy.interpolate.CubicSpline`` when at least 4 window
        centers are available; otherwise falls back to linear.  ``'linear'`` always
        uses linear interpolation.  Default is ``'cubic'``.

    Returns
    -------
    xarray.DataArray
        Cleaned signal with the same shape, dimensions, coordinates, attributes,
        and ``name`` as *lfp*.  Only the data values are modified.

    Raises
    ------
    TypeError
        If *lfp* is not an ``xarray.DataArray``.
    """
    if not isinstance(lfp, xr.DataArray):
        raise TypeError('lfp must be an xarray.DataArray')

    t = lfp.coords[DIM_TIME].values.astype(float)

    if ATTR_FS in lfp.attrs:
        fs = float(lfp.attrs[ATTR_FS])
    else:
        fs = 1.0 / float(np.median(np.diff(t)))

    half_win = int(round(window_size_sec * fs))
    step = max(1, int(round(window_step_sec * fs)))
    N = len(t)

    # Move the time axis to position 0 and flatten all other dimensions so the
    # algorithm can operate on a simple (n_time, n_signals) array.
    time_axis = lfp.dims.index(DIM_TIME)
    data = np.moveaxis(lfp.values, time_axis, 0)  # (n_time, ...)
    orig_shape = data.shape
    data_2d = data.reshape(N, -1)  # (n_time, n_signals)

    cleaned_2d = np.empty_like(data_2d)

    for ch in range(data_2d.shape[1]):
        signal = data_2d[:, ch]

        t_centers, A_vals, B_vals = [], [], []
        for center in range(half_win, N - half_win, step):
            sl = slice(center - half_win, center + half_win)
            tw = t[sl]
            yw = signal[sl]

            X = np.column_stack([
                np.sin(2 * np.pi * noise_freq * tw),
                np.cos(2 * np.pi * noise_freq * tw),
                np.ones(len(tw))
            ])
            coeffs, _, _, _ = np.linalg.lstsq(X, yw, rcond=None)
            t_centers.append(t[center])
            A_vals.append(coeffs[0])
            B_vals.append(coeffs[1])

        t_centers = np.array(t_centers)
        A_vals = np.array(A_vals)
        B_vals = np.array(B_vals)

        order = np.argsort(t_centers)
        t_centers = t_centers[order]
        A_vals = A_vals[order]
        B_vals = B_vals[order]

        unique_centers, inverse, counts = np.unique(
            t_centers, return_inverse=True, return_counts=True
        )
        if len(unique_centers) != len(t_centers):
            A_vals = np.bincount(inverse, weights=A_vals) / counts
            B_vals = np.bincount(inverse, weights=B_vals) / counts
            t_centers = unique_centers

        if interp_method == 'cubic' and len(t_centers) >= 4:
            A_interp = CubicSpline(t_centers, A_vals, extrapolate=False)(t)
            B_interp = CubicSpline(t_centers, B_vals, extrapolate=False)(t)
            # Fill any extrapolated NaNs at the edges with linear interpolation.
            A_interp = np.where(np.isnan(A_interp), np.interp(t, t_centers, A_vals), A_interp)
            B_interp = np.where(np.isnan(B_interp), np.interp(t, t_centers, B_vals), B_interp)
        else:
            A_interp = np.interp(t, t_centers, A_vals)
            B_interp = np.interp(t, t_centers, B_vals)

        noise_est = (A_interp * np.sin(2 * np.pi * noise_freq * t) +
                     B_interp * np.cos(2 * np.pi * noise_freq * t))
        cleaned_2d[:, ch] = signal - noise_est

    # Restore original layout (non-time dims back in their original positions).
    cleaned_data = cleaned_2d.reshape(orig_shape)
    cleaned_data = np.moveaxis(cleaned_data, 0, time_axis)

    return xr.DataArray(
        data=cleaned_data,
        dims=lfp.dims,
        coords=lfp.coords,
        attrs=lfp.attrs,
        name=lfp.name
    )
