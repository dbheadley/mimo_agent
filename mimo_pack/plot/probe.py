"""Probe-layout and channel-spatial plotting for xarray DataArrays.

All functions accept :class:`xarray.DataArray` objects with standard
``COORD_X`` / ``COORD_Y`` (and optionally ``COORD_SHANK``) coordinates
attached along the channel dimension.

Created on 2026-03-06
Drew Headley
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from mimo_pack.consts import DIM_TIME, DIM_CHAN, COORD_X, COORD_Y, COORD_SHANK, ATTR_UNITS
from mimo_pack.plot.colors import MIMO_PALETTE, MIMO_SEQ_CMAP
from mimo_pack.plot.map import _amp_map_numpy
from mimo_pack.plot.timeseries import stacked_lines
from mimo_pack.plot.xarray import _apply_subplots


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_probe_coords(data, require_shank=False):
    """Validate that *data* is an xr.DataArray with probe coordinates.

    Parameters
    ----------
    data : object
        Input to validate.
    require_shank : bool, optional
        If ``True``, also require ``COORD_SHANK``.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If *data* is not an ``xr.DataArray``.
    ValueError
        If required coordinates are missing.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError(
            f"Expected xr.DataArray, got {type(data).__name__}. "
            "Load data with mimo_pack.fileio first."
        )
    missing = [c for c in (COORD_X, COORD_Y) if c not in data.coords]
    if require_shank and COORD_SHANK not in data.coords:
        missing.append(COORD_SHANK)
    if missing:
        raise ValueError(
            f"DataArray is missing required coordinate(s): {missing}. "
            f"Available coordinates: {list(data.coords.keys())}"
        )


# ---------------------------------------------------------------------------
# probe_layout
# ---------------------------------------------------------------------------

def probe_layout(data, chan_labels=True, ax=None, scat_args=None, text_args=None):
    """Plot electrode positions from an xarray DataArray.

    Reads ``COORD_X``, ``COORD_Y``, and (optionally) ``COORD_SHANK``
    from the DataArray's coordinates and draws a scatter plot of each
    electrode site.  When ``COORD_SHANK`` is present each shank is
    coloured separately.

    Parameters
    ----------
    data : xr.DataArray
        DataArray with ``COORD_X`` and ``COORD_Y`` coordinates along the
        channel dimension.  ``COORD_SHANK`` is used for colour grouping
        when available.
    chan_labels : bool, optional
        Whether to annotate each electrode with its channel index from
        ``DIM_CHAN``.  Default is ``True``.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.  A new figure is created if ``None``.
    scat_args : dict or None, optional
        Extra keyword arguments forwarded to ``ax.scatter``.
    text_args : dict or None, optional
        Extra keyword arguments forwarded to ``ax.text`` for labels.

    Returns
    -------
    ax : matplotlib.axes.Axes

    Raises
    ------
    TypeError
        If *data* is not an ``xr.DataArray``.
    ValueError
        If ``COORD_X`` or ``COORD_Y`` are missing.

    Examples
    --------
    >>> from mimo_pack.testing import synthetic_lfp
    >>> lfp = synthetic_lfp(duration=0.1, fs=1000, n_chans=32, seed=42)
    >>> ax = probe_layout(lfp)
    """
    _validate_probe_coords(data)

    if ax is None:
        fig, ax = plt.subplots()

    # ---- defaults ----
    scat_kw = dict(s=20, marker='s', alpha=0.7)
    if scat_args is not None:
        scat_kw.update(scat_args)
    text_kw = dict(fontsize=5, ha='left', va='center')
    if text_args is not None:
        text_kw.update(text_args)

    x = data.coords[COORD_X].values
    y = data.coords[COORD_Y].values
    ch = data.coords[DIM_CHAN].values

    has_shank = COORD_SHANK in data.coords

    if has_shank:
        shank = data.coords[COORD_SHANK].values
        for i, s in enumerate(np.unique(shank)):
            mask = shank == s
            color = MIMO_PALETTE[i % len(MIMO_PALETTE)]
            ax.scatter(x[mask], y[mask], color=color, label=f'shank {int(s)}', **scat_kw)
            if chan_labels:
                # offset text to the right of the point
                x_offset = (np.nanmax(x) - np.nanmin(x)) * 0.02 + 2
                for xi, yi, ci in zip(x[mask], y[mask], ch[mask]):
                    ax.text(xi + x_offset, yi, str(ci), **text_kw)
        ax.legend(fontsize=7, loc='best')
    else:
        scat_kw.setdefault('c', 'k')
        ax.scatter(x, y, **scat_kw)
        if chan_labels:
            x_offset = (np.nanmax(x) - np.nanmin(x)) * 0.02 + 2
            for xi, yi, ci in zip(x, y, ch):
                ax.text(xi + x_offset, yi, str(ci), **text_kw)

    # Auto-widen narrow x-range (single-shank probes)
    x_range = np.nanmax(x) - np.nanmin(x)
    if x_range < 250:
        x_mid = (np.nanmax(x) + np.nanmin(x)) / 2
        ax.set_xlim(x_mid - 150, x_mid + 150)

    ax.set_xlabel('X Position (µm)')
    ax.set_ylabel('Y Position (µm)')

    return ax


# ---------------------------------------------------------------------------
# probe_layout_subplots
# ---------------------------------------------------------------------------

def probe_layout_subplots(data, subplot_coord=COORD_SHANK, axs=None, **kwargs):
    """Plot electrode positions with one subplot per shank.

    Splits *data* by *subplot_coord* (default ``COORD_SHANK``) and draws
    a :func:`probe_layout` panel for each group using
    :func:`~mimo_pack.plot.xarray._apply_subplots`.

    Parameters
    ----------
    data : xr.DataArray
        DataArray with probe coordinates.
    subplot_coord : str, optional
        Coordinate used to create subplots.  Default is ``COORD_SHANK``.
    axs : array-like of matplotlib.axes.Axes or None, optional
        Pre-existing axes.  Created automatically if ``None``.
    **kwargs
        Forwarded to :func:`probe_layout` (e.g. *chan_labels*,
        *scat_args*, *text_args*).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : numpy.ndarray of matplotlib.axes.Axes
    last_return
        Return value of the last :func:`probe_layout` call.

    Raises
    ------
    TypeError
        If *data* is not an ``xr.DataArray``.
    ValueError
        If required coordinates are missing.

    Examples
    --------
    >>> from mimo_pack.testing import synthetic_lfp
    >>> lfp = synthetic_lfp(duration=0.1, fs=1000, n_chans=32, seed=42)
    >>> fig, axs, _ = probe_layout_subplots(lfp)
    """
    _validate_probe_coords(data, require_shank=True)
    return _apply_subplots(data, probe_layout, subplot_coord, axs=axs, **kwargs)


# ---------------------------------------------------------------------------
# channel_values
# ---------------------------------------------------------------------------

def channel_values(data, x_coord=COORD_X, y_coord=COORD_Y,
                   cmap=MIMO_SEQ_CMAP, ax=None, colorbar=True, **kwargs):
    """Colour-coded scatter of one scalar value per channel on the probe.

    Wraps :func:`mimo_pack.plot.map._amp_map_numpy` with input
    validation, auto-labelling, and an optional colour bar.

    Parameters
    ----------
    data : xr.DataArray
        A 1-D DataArray indexed by ``DIM_CHAN`` with one value per
        channel (e.g. RMS amplitude, mean power, z-score).  Must carry
        ``COORD_X`` and ``COORD_Y`` coordinates.
    x_coord : str, optional
        Name of the x-position coordinate.  Default ``COORD_X``.
    y_coord : str, optional
        Name of the y-position coordinate.  Default ``COORD_Y``.
    cmap : str, optional
        Matplotlib colourmap.  Default ``'viridis'``.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.  A new figure is created if ``None``.
    colorbar : bool, optional
        Whether to add a colour bar.  Default ``True``.
    **kwargs
        Forwarded to :func:`_amp_map_numpy` (e.g. ``vmin``, ``vmax``).

    Returns
    -------
    ax : matplotlib.axes.Axes
    scat : matplotlib.collections.PathCollection

    Raises
    ------
    TypeError
        If *data* is not an ``xr.DataArray``.
    ValueError
        If *data* is not 1-D, or required coordinates are missing.

    Notes
    -----
    If your data has extra dimensions (e.g. ``DIM_EPOCH`` from windowing),
    reduce them first::

        # Trial-averaged RMS per channel
        rms = (lfp_win.mean(DIM_EPOCH) ** 2).mean(DIM_TIME) ** 0.5
        channel_values(rms)

    Examples
    --------
    >>> from mimo_pack.testing import synthetic_lfp
    >>> lfp = synthetic_lfp(duration=1.0, fs=1000, n_chans=8, seed=42)
    >>> rms = (lfp ** 2).mean('time') ** 0.5
    >>> ax, scat = channel_values(rms)
    """
    _validate_probe_coords(data)

    if data.ndim != 1:
        raise ValueError(
            f"Expected a 1-D DataArray (one value per channel), "
            f"got {data.ndim}-D with dims {data.dims}. "
            "Reduce over other dimensions first (e.g. .mean('time'))."
        )

    ax, scat = _amp_map_numpy(
        data.values,
        data.coords[x_coord].values,
        data.coords[y_coord].values,
        cmap=cmap,
        ax=ax,
        **kwargs,
    )

    ax.set_xlabel('X Position (µm)')
    ax.set_ylabel('Y Position (µm)')

    if colorbar:
        label = data.attrs.get(ATTR_UNITS, '')
        ax.figure.colorbar(scat, ax=ax, label=label)

    return ax, scat


# ---------------------------------------------------------------------------
# depth_lines
# ---------------------------------------------------------------------------

def depth_lines(data, time_dim=DIM_TIME, chan_dim=DIM_CHAN,
                depth_coord=COORD_Y, ax=None, **kwargs):
    """Stacked line plot of channels ordered by probe depth.

    Sorts channels by ``depth_coord`` (ascending) and delegates to
    :func:`mimo_pack.plot.timeseries.stacked_lines`.

    Parameters
    ----------
    data : xr.DataArray
        2-D DataArray with *time_dim* and *chan_dim* dimensions.
    time_dim : str, optional
        Name of the time dimension.  Default ``DIM_TIME``.
    chan_dim : str, optional
        Name of the channel dimension.  Default ``DIM_CHAN``.
    depth_coord : str, optional
        Coordinate used for depth ordering and right-axis labels.
        Default ``COORD_Y``.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.
    **kwargs
        Forwarded to :func:`stacked_lines` (e.g. ``sep_ratio``,
        ``abs_sep``, ``disp_color``).

    Returns
    -------
    ax : matplotlib.axes.Axes

    Raises
    ------
    TypeError
        If *data* is not an ``xr.DataArray``.
    ValueError
        If required dimensions or depth coordinate are missing.

    Notes
    -----
    If your data has an ``DIM_EPOCH`` dimension (from windowing), select
    a single trial or average across epochs first::

        depth_lines(lfp_win.mean(DIM_EPOCH))        # trial-averaged
        depth_lines(lfp_win.isel({DIM_EPOCH: 0}))   # single trial

    Examples
    --------
    >>> from mimo_pack.testing import synthetic_lfp
    >>> lfp = synthetic_lfp(duration=0.5, fs=1000, n_chans=16, seed=42)
    >>> ax = depth_lines(lfp)
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError(
            f"Expected xr.DataArray, got {type(data).__name__}. "
            "Load data with mimo_pack.fileio first."
        )
    for dim in (time_dim, chan_dim):
        if dim not in data.dims:
            raise ValueError(
                f"Dimension '{dim}' not found in DataArray. "
                f"Available dimensions: {list(data.dims)}"
            )
    if depth_coord not in data.coords:
        raise ValueError(
            f"Coordinate '{depth_coord}' not found in DataArray. "
            f"Available coordinates: {list(data.coords.keys())}"
        )

    # Sort channels by depth (ascending)
    sort_idx = np.argsort(data.coords[depth_coord].values)
    data_sorted = data.isel({chan_dim: sort_idx})

    # Delegate to stacked_lines with depth labels
    ax = stacked_lines(
        data_sorted,
        time_dim=time_dim,
        chan_dim=chan_dim,
        label_coord=depth_coord,
        ax=ax,
        **kwargs,
    )

    ax.set_xlabel('Time (s)')

    return ax


# ---------------------------------------------------------------------------
# unit_lfp_mapping
# ---------------------------------------------------------------------------

def unit_lfp_mapping(spks, lfp, jitter=30, ax=None):
    """Plot arrows from spike-sorted unit positions to their assigned LFP channels.

    Draws annotated arrows on a probe coordinate plot — one arrow per unit
    in *spks* that has an ``'lfp_channel'`` metadata entry — so you can
    visually inspect the unit-to-LFP-channel assignment produced by
    :func:`mimo_pack.analysis.probe.find_lfp_channel`.

    Parameters
    ----------
    spks : pynapple.TsGroup
        Spike-sorted units.  Must have a ``.metadata`` DataFrame with
        columns ``'x'``, ``'y'`` (unit positions in µm) and
        ``'lfp_channel'`` (channel index, or ``NaN`` when unassigned).
    lfp : xr.DataArray
        LFP DataArray with ``COORD_X`` and ``COORD_Y`` coordinates along
        ``DIM_CHAN``.
    jitter : float, optional
        Random jitter (µm) added to unit positions for visibility when
        multiple units share the same location.  Default is 30.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on.  A new figure is created if ``None``.

    Returns
    -------
    ax : matplotlib.axes.Axes

    Raises
    ------
    TypeError
        If *spks* is not a ``pynapple.TsGroup`` or *lfp* is not an
        ``xr.DataArray``.
    ValueError
        If *lfp* is missing ``COORD_X`` or ``COORD_Y`` coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> import pynapple as nap
    >>> import pandas as pd
    >>> from mimo_pack.testing import synthetic_lfp
    >>> lfp = synthetic_lfp(duration=1.0, fs=1000, n_chans=8, seed=42)
    >>> meta = pd.DataFrame({'x': [0.0], 'y': [50.0], 'lfp_channel': [lfp.coords['channel'].values[0]]})
    >>> spks = nap.TsGroup({0: nap.Ts(t=np.array([0.1, 0.5, 0.9]))}, metadata=meta)
    >>> ax = unit_lfp_mapping(spks, lfp)
    """
    import pynapple as nap

    if not isinstance(spks, nap.TsGroup):
        raise TypeError(
            f"Expected pynapple.TsGroup for 'spks', got {type(spks).__name__}."
        )
    if not isinstance(lfp, xr.DataArray):
        raise TypeError(
            f"Expected xr.DataArray for 'lfp', got {type(lfp).__name__}. "
            "Load data with mimo_pack.fileio first."
        )
    _validate_probe_coords(lfp)

    if ax is None:
        fig, ax = plt.subplots()

    rng = np.random.default_rng()

    for unit in spks:
        unit_pos = spks.metadata.loc[unit, ['x', 'y']].values.astype(float).copy()
        unit_pos += rng.uniform(-jitter, jitter, size=2)
        lfp_ch = spks.metadata.loc[unit, 'lfp_channel']

        if np.isnan(lfp_ch):
            ax.plot(unit_pos[0], unit_pos[1], 'go', markersize=5, alpha=0.5)
            continue

        ch_sel = lfp.sel({DIM_CHAN: int(lfp_ch)})
        lfp_x = float(ch_sel.coords[COORD_X])
        lfp_y = float(ch_sel.coords[COORD_Y])
        ax.annotate(
            '',
            xy=(lfp_x, lfp_y),
            xytext=(unit_pos[0], unit_pos[1]),
            arrowprops=dict(arrowstyle="-|>", color='k', lw=1, alpha=0.2),
        )
        ax.plot(unit_pos[0], unit_pos[1], 'bo', markersize=5, alpha=0.5)
        ax.plot(lfp_x, lfp_y, 'r+', markersize=5, alpha=0.5)

    ax.set_xlabel('X position (µm)')
    ax.set_ylabel('Y position (µm)')
    ax.set_title('Unit to LFP Channel Mapping')

    return ax
