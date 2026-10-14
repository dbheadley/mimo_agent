""" 
Function for working with probe data
Created on 2025-06-21
Drew Headley
"""

import numpy as np
import pandas as pd
import xarray as xr

from mimo_pack.consts import DIM_CHAN, COORD_X, COORD_Y


def nearest_grid(data, dx=250.0, dy=100.0, max_dist=100.0):
    """Find the channels on a probe nearest to regular grid points.

    Parameters
    ----------
    data : xr.DataArray
        DataArray with ``COORD_X`` and ``COORD_Y`` coordinates along
        ``DIM_CHAN``.  The actual data values are not used; only the
        spatial coordinates are needed.
    dx : float, optional
        Horizontal grid spacing in micrometers.  Default is 250.0.
    dy : float, optional
        Vertical grid spacing in micrometers.  Default is 100.0.
    max_dist : float, optional
        Maximum distance (µm) for a channel to be assigned to a grid
        point.  Default is 100.0.

    Returns
    -------
    ch_near : numpy.ndarray
        Channel indices nearest to each grid point.
    x_near : numpy.ndarray
        X positions of the selected channels.
    y_near : numpy.ndarray
        Y positions of the selected channels.

    Raises
    ------
    TypeError
        If *data* is not an ``xr.DataArray``.
    ValueError
        If ``COORD_X`` or ``COORD_Y`` coordinates are missing.

    Examples
    --------
    >>> from mimo_pack.testing import synthetic_lfp
    >>> lfp = synthetic_lfp(duration=0.1, fs=1000, n_chans=32, seed=42)
    >>> ch, x, y = nearest_grid(lfp, dx=250, dy=100)
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError(
            f"Expected xr.DataArray, got {type(data).__name__}. "
            "Pass the probe DataArray (or a DataArray built from probe coords)."
        )
    missing = [c for c in (COORD_X, COORD_Y) if c not in data.coords]
    if missing:
        raise ValueError(
            f"DataArray is missing required coordinate(s): {missing}. "
            f"Available coordinates: {list(data.coords.keys())}"
        )

    x_pos = data.coords[COORD_X].values
    y_pos = data.coords[COORD_Y].values
    ch_idx = data.coords[DIM_CHAN].values
    
    x_min, x_max = np.nanmin(x_pos), np.nanmax(x_pos)
    y_min, y_max = np.nanmin(y_pos), np.nanmax(y_pos)
    
    x_grid = np.arange(x_min, x_max, dx)
    y_grid = np.arange(y_min, y_max, dy)
    
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # create arrays to hold the nearest channel indices and positions
    ch_near = np.empty(X.size, dtype=ch_idx.dtype)
    x_near = np.empty(X.size, dtype=x_pos.dtype)
    y_near = np.empty(X.size, dtype=y_pos.dtype)
    bad_chans = []
    for i, (x, y) in enumerate(zip(X.flatten(), Y.flatten())):
        # calculate the distance from each channel to the grid point
        distances = np.sqrt((x_pos - x)**2 + (y_pos - y)**2)
        nearest_idx = np.nanargmin(distances)
        
        # check if the nearest channel is within the maximum distance
        if distances[nearest_idx] > max_dist:
            bad_chans.append(i)
        else:
            ch_near[i] = ch_idx[nearest_idx]
            x_near[i] = x_pos[nearest_idx]
            y_near[i] = y_pos[nearest_idx]

    ch_near = np.delete(ch_near, bad_chans)
    x_near = np.delete(x_near, bad_chans)
    y_near = np.delete(y_near, bad_chans)

    # remove duplicate channels
    _, unique_idxs = np.unique(ch_near, return_index=True)
    ch_near = ch_near[unique_idxs]
    x_near = x_near[unique_idxs]
    y_near = y_near[unique_idxs]
    
    return ch_near, x_near, y_near


def find_lfp_channel(spks, lfp, min_dist=100, max_dist=200):
    """
    Find the closest LFP channel for each unit on a probe.

    Parameters
    ----------
    spks : nap.TsGroup
        Pynapple TsGroup containing spike times for each unit
    lfp : xarray.DataArray
        LFP data with channel coordinates
    min_dist : int, optional
        Minimum distance to consider a channel as a candidate, by default 100
    max_dist : int, optional
        Maximum distance to consider a channel as a candidate, by default 200

    Returns
    -------
    nap.TsGroup
        Pynapple TsGroup with added LFP channel for each unit
    """

    # get the coordinates of the LFP channels
    lfp_coords = np.array([lfp.coords[COORD_X].values,
                           lfp.coords[COORD_Y].values]).T
    lfp_chans = lfp.coords[DIM_CHAN].values
    # get the coordinates of the units
    unit_coords = spks.metadata[['x', 'y']].values

    unit_lfp_ch = pd.Series(np.nan, index=spks.metadata.index, name='lfp_channel')
    # calculate the distance between each unit and each LFP channel
    for i, unit_coord in enumerate(unit_coords):
        dist = np.linalg.norm(lfp_coords - unit_coord, axis=1)
        # find the closest LFP channel within the specified distance range
        candidates = np.where((dist >= min_dist) & (dist <= max_dist))[0]
        if len(candidates) > 0:
            closest_ch = candidates[np.argmin(dist[candidates])]
            unit_lfp_ch.iloc[i] = lfp_chans[closest_ch]
    
    # add the LFP channel to the metadata of the spike times
    spks.set_info({'lfp_channel': unit_lfp_ch})

    return spks
