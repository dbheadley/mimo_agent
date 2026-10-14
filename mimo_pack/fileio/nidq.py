"""Load dclut nidq files
Author: Drew B. Headley
"""

import re

import numpy as np
import dclut as dcl
import xarray as xr
from mimo_pack.consts import ATTR_FS, DIM_TIME, DIM_CHAN

# SpikeGLX two-letter channel-type prefixes
_SGLX_ANALOG_PREFIXES = frozenset({'MN', 'MA', 'XA'})
_SGLX_DIGITAL_PREFIXES = frozenset({'XD'})
_SGLX_ALL_PREFIXES = frozenset({'MN', 'MA', 'XA', 'XD', 'SY', 'AP', 'LF', 'OB'})


def _sglx_prefix(name_str: str) -> str:
    """Return the alphabetic prefix of a SpikeGLX channel name.

    Examples
    --------
    >>> _sglx_prefix('XD0;12')
    'XD'
    >>> _sglx_prefix('MN0C0;0')
    'MN'
    """
    base = name_str.split(';')[0]
    m = re.match(r'^([A-Za-z]+)', base)
    return m.group(1).upper() if m else ''


def _find_chan_name_coord(da: xr.DataArray, chan_dim: str) -> str | None:
    """Return the name of the first coordinate indexed by *chan_dim* whose
    values look like SpikeGLX channel names (contain ``;`` or start with a
    known prefix such as ``XA``, ``XD``, ``MN`` …).  Returns ``None`` if no
    such coordinate is found.
    """
    for coord_name, coord_arr in da.coords.items():
        if coord_name == chan_dim:
            continue
        if chan_dim not in coord_arr.dims:
            continue
        vals = coord_arr.values
        for v in vals[:min(len(vals), 5)]:
            s = str(v)
            if ';' in s or _sglx_prefix(s) in _SGLX_ALL_PREFIXES:
                return coord_name
    return None


def _expand_xd_channels(da: xr.DataArray) -> xr.DataArray:
    """Unpack XD (digital word) channels into individual binary channels.

    SpikeGLX stores up to 16 digital input lines packed as bits in a single
    unsigned 16-bit word called an XD channel.  Bit 0 is the lowest-numbered
    line (LSB-first).  This function replaces each XD channel with 16
    individual channels named ``XD0_b00`` … ``XD0_b15`` (values 0 or 1).

    A ``chan_type`` coordinate is added to the channel dimension:
    ``'analog'`` for MN / MA / XA channels, ``'digital'`` for all
    XD-derived bit channels.

    Parameters
    ----------
    da : xr.DataArray
        Loaded NIDQ DataArray with dimensions ``(time, channel)``.

    Returns
    -------
    xr.DataArray
        DataArray with XD channels expanded and ``chan_type`` coordinate
        attached.  If no XD channels are present, returns *da* with only
        the ``chan_type`` coordinate added.
    """
    chan_dim = next((d for d in da.dims if d != DIM_TIME), DIM_CHAN)
    name_coord = _find_chan_name_coord(da, chan_dim)

    # rows: (new_id, values_1d, chan_type, display_name)
    rows: list[tuple[str, np.ndarray, str, str]] = []
    has_digital = False

    for i, ch_val in enumerate(da[chan_dim].values):
        ch_data = da.isel({chan_dim: i}).values          # (n_time,)
        raw_name = (
            str(da[name_coord].values[i]) if name_coord else str(ch_val)
        )
        prefix = _sglx_prefix(raw_name)
        xd_base = raw_name.split(';')[0]                 # e.g. 'XD0'

        if prefix in _SGLX_DIGITAL_PREFIXES:
            # Cast to uint16 (dclut may return floats)
            raw_uint = np.round(ch_data).astype(np.uint16)
            for bit in range(16):
                bit_vals = ((raw_uint >> bit) & np.uint16(1)).astype(np.float32)
                bit_id = f"{xd_base}_b{bit:02d}"
                rows.append((bit_id, bit_vals, 'digital', bit_id))
            has_digital = True
        else:
            ctype = 'analog' if prefix in _SGLX_ANALOG_PREFIXES else 'analog'
            rows.append((str(ch_val), ch_data.astype(np.float32), ctype, raw_name))

    if not has_digital:
        # No XD channels; just stamp all channels as analog
        n = da.sizes[chan_dim]
        return da.assign_coords({'chan_type': (chan_dim, ['analog'] * n)})

    # Assemble expanded DataArray
    data_mat = np.stack([r[1] for r in rows], axis=1)   # (n_time, n_chan)
    new_ids = [r[0] for r in rows]
    new_types = [r[2] for r in rows]
    new_names = [r[3] for r in rows]

    coords: dict = {
        DIM_TIME:    da[DIM_TIME].values,
        chan_dim:    new_ids,
        'chan_type': (chan_dim, new_types),
    }
    if name_coord:
        coords[name_coord] = (chan_dim, new_names)

    return xr.DataArray(
        data_mat,
        dims=(DIM_TIME, chan_dim),
        coords=coords,
        attrs={
            **da.attrs,
            'description': (
                da.attrs.get('description', '') + ' -> XD bits expanded'
            ),
        },
    )


def load_nidq(nidq_path: str, chan_names: dict = None,
              remove_nan_time: bool = True,
              expand_digital: bool = True) -> xr.DataArray:
    """Load NIDQ data from a dclut file as an xarray DataArray.

    Parameters
    ----------
    nidq_path : str
        Path to the dclut NIDQ file (``*.nidq_dclut.json``).
    chan_names : dict, optional
        Mapping of ``{original_channel_id: new_label}`` used to rename
        the channel coordinate after loading.  Original IDs are compared
        as strings.  After digital expansion, analog channel IDs are the
        stringified original integer indices (e.g. ``'0'``, ``'1'``), and
        digital bit channel IDs are strings like ``'XD0_b00'``.
        Example: ``{'0': 'photodiode', 'XD0_b00': 'ttl_trial'}``
    remove_nan_time : bool, optional
        Remove time samples whose timestamp is NaN (default ``True``).
    expand_digital : bool, optional
        If ``True`` (default), each XD channel (unsigned 16-bit word of
        packed digital lines) is unpacked into 16 individual binary
        channels named ``XD0_b00`` … ``XD0_b15`` (values 0 or 1).  A
        ``chan_type`` coordinate is added: ``'analog'`` for XA / MN / MA
        channels, ``'digital'`` for all XD-derived bit channels.
        Set to ``False`` to load the raw packed word without expansion.

    Returns
    -------
    xr.DataArray
        NIDQ data with dimensions ``(time, channel)``.  Attrs include ``fs``
        (sampling rate) and ``description``.  When *expand_digital* is ``True``
        a ``chan_type`` coordinate is added to the channel dimension.  If the
        dclut file contains a ``gain`` scale (written by
        :func:`mimo_pack.fileio.spikeglx.dclut_from_meta`), the raw int16
        counts are multiplied by the per-channel gain and ``attrs['units']``
        is set to ``'V'``.

    Raises
    ------
    FileNotFoundError
        If *nidq_path* does not point to a valid dclut file.

    Examples
    --------
    >>> from mimo_pack.fileio.nidq import load_nidq
    >>> nidq = load_nidq('/data/session/session_g0_t0.nidq_dclut.json')
    >>> analog  = nidq.sel(channel=nidq.chan_type == 'analog')
    >>> digital = nidq.sel(channel=nidq.chan_type == 'digital')
    """
    # Load dclut object
    nidq_dcl = dcl.dclut(nidq_path)
    nidq_dcl.reset()
    # Select all channels (required by dclut — at least one dim must be selected)
    n_chans = nidq_dcl.shape[1]
    nidq_dcl.points(select={'channel': list(range(n_chans))})
    nidq_xr = nidq_dcl.read(format='xarray')[0]

    # Extract original gain list before digital expansion (keyed by channel index)
    raw_gain_scale = nidq_dcl.dcl.get('scales', {}).get('gain', None)
    if raw_gain_scale is not None:
        orig_gains = list(raw_gain_scale.get('values', []))
    else:
        orig_gains = None

    fs = 1 / np.nanmedian(np.diff(nidq_xr.time.to_numpy().flatten()))
    nidq_xr = nidq_xr.assign_attrs(
        **{ATTR_FS: fs, 'description': f"NIDQ data: {nidq_path}"}
    )

    # Remove time samples with NaN timestamps if requested
    if remove_nan_time:
        mask = ~np.isnan(nidq_xr[DIM_TIME].values)
        nidq_xr = nidq_xr.isel({DIM_TIME: mask})

    # Unpack XD digital word channels into individual bit channels
    if expand_digital:
        nidq_xr = _expand_xd_channels(nidq_xr)

    # Apply per-channel voltage gain if a gain scale was present in the dclut
    if orig_gains is not None:
        chan_dim = next(d for d in nidq_xr.dims if d != DIM_TIME)
        name_coord = _find_chan_name_coord(nidq_xr, chan_dim)

        # Build gain lookup: original_channel_index (int) → gain float
        gain_by_orig_idx = {
            i: float(orig_gains[i]) for i in range(len(orig_gains))
        }

        # Determine expanded channel names to map back to original indices.
        # After expansion the channel coord contains strings like 'XD0_b00' or
        # the original integer index as a string ('0', '1', …).
        expanded_ids = nidq_xr[chan_dim].values
        expanded_gains = []
        xd_seen: dict[str, int] = {}   # XD base name → original index

        # Build a reverse map: SpikeGLX name → original index (if name coord exists)
        name_to_orig: dict[str, int] = {}
        if name_coord and name_coord in nidq_xr.coords:
            for ei, ch_id in enumerate(nidq_xr[chan_dim].values):
                sglx_name = str(nidq_xr[name_coord].values[ei])
                # Analogue channels: sglx_name is something like 'XA0;0'
                # Digital bit channels: sglx_name matches bit_id (e.g. 'XD0_b00')
                name_to_orig[str(ch_id)] = ei  # fallback — set below

        for ei, ch_id in enumerate(expanded_ids):
            ch_str = str(ch_id)
            if '_b' in ch_str:
                # Digital bit channel — gain is 1.0 regardless
                expanded_gains.append(1.0)
            else:
                # Analog channel — channel coord value is the original index string
                try:
                    orig_idx = int(ch_str)
                    expanded_gains.append(gain_by_orig_idx.get(orig_idx, 1.0))
                except ValueError:
                    expanded_gains.append(1.0)

        gain_da = xr.DataArray(expanded_gains, dims=[chan_dim])
        nidq_xr = (nidq_xr * gain_da).assign_attrs(
            **nidq_xr.attrs,
            units='V',
        )

    # Rename channels if a mapping was provided
    if chan_names is not None:
        chan_dim = next(d for d in nidq_xr.dims if d != DIM_TIME)
        orig_vals = nidq_xr[chan_dim].values
        new_vals = [chan_names.get(str(v), str(v)) for v in orig_vals]
        nidq_xr = nidq_xr.assign_coords({chan_dim: new_vals})

    return nidq_xr
