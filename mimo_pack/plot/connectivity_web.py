"""Standalone interactive connectivity web pages.

Builds self-contained HTML pages (no external dependencies, viewable offline
from ``file://``) that combine a spatial monosynaptic connectivity map, a
connection matrix, per-connection cross-correlograms, and per-unit spike
waveforms. All data is embedded in the page as JSON.

Public API
----------
connectivity_page(units, connections, cc_lags_s, ...) : str
    Build one probe's interactive page and return it as an HTML string.
connectivity_index_page(pages, ...) : str
    Build a simple index page linking a set of connectivity pages.

Interactions on the generated page: clicking a unit highlights its outbound
(solid) and inbound (dashed) connections and shows stair-plot CC sparklines
plus the unit's spike waveform; clicking an arrow or matrix cell lists just
that connection; clicking a matrix row label lists only outbound connections,
a column label only inbound ones. Clicking a sparkline in the CC panel shows
the full cross-correlogram and thickens that connection on the map without
changing the list.

Created on 2026-10-04
Drew Headley
"""

import html
import json
from importlib import resources

import matplotlib as mpl
import numpy as np
import pandas as pd
import xarray as xr

from mimo_pack.consts import COORD_X, COORD_Y, ATTR_FS
from mimo_pack.plot.colors import MIMO_SEQ_CMAP

# Column aliases accepted for the *units* table.
_UNIT_COL_ALIASES = {'region_name': 'region', 'laminar_layer': 'layer'}

# Laminar boundary conventions (depths below surface, um) used by the
# LaminarBoundaries pipeline. Bands between successive boundaries are labeled
# with DEFAULT_LAYER_LABELS; the band below the last boundary uses the
# table's 'floor_type' column when present.
DEFAULT_BOUNDARY_COLS = ('surface_um', 'L1_L23_um', 'L23_L5_um', 'L5_L6_um', 'floor_um')
DEFAULT_LAYER_LABELS = ('L1', 'L2/3', 'L5', 'L6')

_TEMPLATE_RESOURCE = 'templates/connectivity_interactive.html'
_N_CMAP_STOPS = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jsnum(v, nd=2):
    """Round *v* to *nd* decimals; non-finite values become None (JSON null)."""
    v = float(v)
    return round(v, nd) if np.isfinite(v) else None


def _jslist(arr, nd=2):
    return [_jsnum(v, nd) for v in np.asarray(arr, dtype=float)]


def _load_template():
    return (resources.files('mimo_pack.plot')
            .joinpath(_TEMPLATE_RESOURCE).read_text(encoding='utf-8'))


def _cmap_stops():
    """Sample MIMO_SEQ_CMAP into integer RGB stops for the page's JS."""
    cmap = mpl.colormaps[MIMO_SEQ_CMAP]
    return [[int(round(c * 255)) for c in cmap(i / (_N_CMAP_STOPS - 1))[:3]]
            for i in range(_N_CMAP_STOPS)]


def _normalize_units(units):
    """Validate the units table and apply column aliases."""
    if not isinstance(units, pd.DataFrame):
        raise TypeError(f"units must be a pandas DataFrame, got {type(units).__name__}")
    u = units.reset_index() if units.index.name == 'unit_id' else units.copy()
    u = u.rename(columns=_UNIT_COL_ALIASES)
    missing = [c for c in ('unit_id', 'x', 'y') if c not in u.columns]
    if missing:
        raise ValueError(f"units table is missing required column(s): {missing}")
    for col in ('cell_type', 'region', 'layer', 'shank'):
        if col not in u.columns:
            u[col] = np.nan
    return u.drop_duplicates('unit_id')


def _normalize_connections(connections, n_lags):
    """Validate the connections table against the lag axis length."""
    if not isinstance(connections, pd.DataFrame):
        raise TypeError(
            f"connections must be a pandas DataFrame, got {type(connections).__name__}")
    missing = [c for c in ('pre_unit', 'post_unit', 'cc') if c not in connections.columns]
    if missing:
        raise ValueError(f"connections table is missing required column(s): {missing}")
    c = connections.copy()
    for col in ('sign', 'lag_s', 'amplitude'):
        if col not in c.columns:
            c[col] = np.nan
    if len(c):
        n_cc = len(np.asarray(c['cc'].iloc[0]))
        if n_cc != n_lags:
            raise ValueError(
                f"cc arrays have {n_cc} bins but cc_lags_s has {n_lags} entries")
    return c


def _units_payload(u):
    out = []
    for _, r in u.iterrows():
        out.append({
            'id': int(r['unit_id']),
            'x': _jsnum(r['x'], 1), 'y': _jsnum(r['y'], 1),
            'cellType': r['cell_type'] if pd.notna(r['cell_type']) else None,
            'region': r['region'] if pd.notna(r['region']) else None,
            'layer': r['layer'] if pd.notna(r['layer']) else None,
            'shank': int(r['shank']) if pd.notna(r['shank']) else None,
        })
    return out


def _connections_payload(c):
    out = []
    for _, r in c.iterrows():
        out.append({
            'pre': int(r['pre_unit']), 'post': int(r['post_unit']),
            'sign': int(r['sign']) if pd.notna(r['sign']) else 0,
            'lagMs': _jsnum(r['lag_s'] * 1000.0, 2) if pd.notna(r['lag_s']) else None,
            'amp': _jsnum(r['amplitude'], 3) if pd.notna(r['amplitude']) else None,
            'cc': _jslist(r['cc'], 4),
        })
    return out


def _boundaries_payload(boundaries, boundary_cols, layer_labels):
    if boundaries is None or not len(boundaries):
        return []
    if not isinstance(boundaries, pd.DataFrame):
        raise TypeError(
            f"boundaries must be a pandas DataFrame, got {type(boundaries).__name__}")
    out = []
    for _, b in boundaries.iterrows():
        if 'shank' not in b or pd.isna(b['shank']):
            continue
        depths = [(_jsnum(b[k], 0) if k in b and pd.notna(b[k]) else None)
                  for k in boundary_cols]
        floor_type = b.get('floor_type', 'WM')
        names = list(layer_labels) + [floor_type if pd.notna(floor_type) else 'WM']
        labels = [{'name': nm, 'mid': _jsnum((lo + hi) / 2, 0)}
                  for lo, hi, nm in zip(depths[:-1], depths[1:], names)
                  if lo is not None and hi is not None]
        out.append({'shank': int(b['shank']), 'depths': depths, 'labels': labels})
    return out


def _matrix_order(conns, u):
    """Connection-matrix unit ordering: RS/other then FS, then shank, then depth.

    Returns (ordered unit ids, group-break indices, shank-break indices).
    """
    unit_ids = sorted(set(conns['pre_unit']).union(conns['post_unit']))
    meta = u.set_index('unit_id')
    order = pd.DataFrame({'unit_id': unit_ids})
    for col in ('cell_type', 'shank', 'y'):
        order[col] = order['unit_id'].map(meta[col]) if len(meta) else np.nan
    order['group_rank'] = (order['cell_type'] == 'FS').astype(int)   # RS/other first
    order = order.sort_values(['group_rank', 'shank', 'y', 'unit_id'],
                              kind='stable', na_position='last').reset_index(drop=True)
    group_breaks = [i for i in range(1, len(order))
                    if order['group_rank'][i] != order['group_rank'][i - 1]]
    shank_breaks = [i for i in range(1, len(order))
                    if order['group_rank'][i] == order['group_rank'][i - 1]
                    and pd.notna(order['shank'][i]) and pd.notna(order['shank'][i - 1])
                    and order['shank'][i] != order['shank'][i - 1]]
    return [int(v) for v in order['unit_id']], group_breaks, shank_breaks


def _waveform_payload(waveforms, unit_ids, wf_fs):
    """Convert per-unit waveform entries into the page's JSON structures.

    Returns (wf_dict, sites) where sites is the probe-wide electrode geometry
    taken from the first entry that provides a ``full_waveform``.
    """
    if not waveforms:
        return {}, []
    sites = None
    out = {}
    for uid, entry in waveforms.items():
        if uid not in unit_ids:
            continue
        if isinstance(entry, xr.DataArray):
            da, full = entry, None
        elif isinstance(entry, dict) and 'waveform' in entry:
            da = entry['waveform']
            full = entry.get('full_waveform')
        else:
            continue
        if not isinstance(da, xr.DataArray):
            raise TypeError(
                f"waveform for unit {uid} must be an xr.DataArray, "
                f"got {type(da).__name__}")
        missing = [c for c in (COORD_X, COORD_Y) if c not in da.coords]
        if missing:
            raise ValueError(
                f"waveform for unit {uid} is missing coordinate(s): {missing}")
        fs = float(da.attrs.get(ATTR_FS, wf_fs))
        t = np.asarray(da[da.dims[0]], dtype=float)   # sample indices
        rec = {
            'chs': [[_jsnum(x, 1), _jsnum(y, 1)]
                    for x, y in zip(np.asarray(da[COORD_X], float),
                                    np.asarray(da[COORD_Y], float))],
            'v': [_jslist(col, 2) for col in np.asarray(da.values, float).T],
            'dtMs': _jsnum((t[1] - t[0]) / fs * 1000.0, 4),
        }
        if isinstance(full, xr.DataArray) and COORD_X in full.coords:
            full = full.sortby(full.dims[1])
            if sites is None:
                sites = [[_jsnum(x, 1), _jsnum(y, 1)]
                         for x, y in zip(np.asarray(full[COORD_X], float),
                                         np.asarray(full[COORD_Y], float))]
            amps = np.nanmax(full.values, axis=0) - np.nanmin(full.values, axis=0)
            rec['amps'] = _jslist(amps, 1)
        out[int(uid)] = rec
    return out, (sites or [])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def connectivity_page(units, connections, cc_lags_s, *, waveforms=None,
                      boundaries=None, title='probe',
                      mono_window_s=(0.001, 0.010), wf_fs=30000.0,
                      boundary_cols=DEFAULT_BOUNDARY_COLS,
                      layer_labels=DEFAULT_LAYER_LABELS,
                      index_href=None):
    """Build a self-contained interactive connectivity page for one probe.

    Parameters
    ----------
    units : pd.DataFrame
        One row per unit. Required columns: ``unit_id``, ``x`` (um), ``y``
        (depth below surface, um; plotted downward). Optional: ``cell_type``
        ('RS'/'FS'/'MSN'/'UN'), ``region`` (alias ``region_name``), ``layer``
        (alias ``laminar_layer``), ``shank``. A ``unit_id`` index is also
        accepted.
    connections : pd.DataFrame
        One row per significant connection. Required columns: ``pre_unit``,
        ``post_unit``, ``cc`` (array of CC counts per lag bin). Optional:
        ``sign`` (1/-1/0), ``lag_s``, ``amplitude``.
    cc_lags_s : array-like
        CC lag-bin centers in seconds, shared by all ``cc`` arrays.
    waveforms : dict, optional
        Maps ``unit_id`` to either an ``xr.DataArray`` of shape
        ``(time, channel)`` with ``COORD_X`` / ``COORD_Y`` channel
        coordinates (the unit's mean waveform on its nearest sites), or a
        dict with keys ``'waveform'`` (same DataArray) and optionally
        ``'full_waveform'`` (same layout over all probe sites, used for the
        per-site amplitude heatmap). This matches the waveform metadata
        entries produced by :func:`mimo_pack.preprocess.unit.add_waveform_metrics`.
    boundaries : pd.DataFrame, optional
        Laminar boundaries with columns ``shank``, *boundary_cols* (depths
        below surface, um) and optionally ``floor_type``.
    title : str, optional
        Page title (probe name).
    mono_window_s : tuple of float, optional
        (min, max) absolute monosynaptic lag in seconds; shaded on CC plots.
    wf_fs : float, optional
        Waveform sampling rate (Hz) used to convert the sample-index time
        coordinate to ms. Overridden per-unit by an ``fs`` attr if present.
    boundary_cols, layer_labels : sequence of str, optional
        Boundary depth columns and the labels of the bands between them.
    index_href : str, optional
        If given, the page header links back to this URL.

    Returns
    -------
    str
        A complete standalone HTML document.
    """
    lags_ms = np.asarray(cc_lags_s, dtype=float) * 1000.0
    u = _normalize_units(units)
    c = _normalize_connections(connections, len(lags_ms))

    unit_ids = set(int(v) for v in u['unit_id'])
    c = c[c['pre_unit'].isin(unit_ids) & c['post_unit'].isin(unit_ids)]

    order, group_breaks, shank_breaks = _matrix_order(c, u)
    wf, sites = _waveform_payload(waveforms, unit_ids, wf_fs)

    payload = {
        'probe': str(title),
        'lagsMs': _jslist(lags_ms, 2),
        'monoMinMs': mono_window_s[0] * 1000.0,
        'monoMaxMs': mono_window_s[1] * 1000.0,
        'units': _units_payload(u),
        'conns': _connections_payload(c),
        'bounds': _boundaries_payload(boundaries, boundary_cols, layer_labels),
        'order': order,
        'groupBreaks': group_breaks, 'shankBreaks': shank_breaks,
        'wf': wf, 'sites': sites,
    }

    link = (f'<a href="{html.escape(index_href, quote=True)}">&larr; all probes</a>'
            if index_href else '')
    return (_load_template()
            .replace('__TITLE__', html.escape(str(title)))
            .replace('__INDEX_LINK__', link)
            .replace('__CMAP_STOPS__', json.dumps(_cmap_stops()))
            .replace('/*__DATA__*/null', json.dumps(payload, separators=(',', ':'))))


def connectivity_index_page(pages, title='Interactive connectivity maps'):
    """Build an index page linking a set of connectivity pages.

    Parameters
    ----------
    pages : sequence of dict or pd.DataFrame
        One entry per page with required keys/columns ``href`` and ``title``;
        any additional keys are rendered as extra table columns in order.
    title : str, optional
        Heading and document title.

    Returns
    -------
    str
        A complete standalone HTML document.
    """
    df = pd.DataFrame(pages)
    if df.empty or not {'href', 'title'}.issubset(df.columns):
        raise ValueError("pages must provide 'href' and 'title' for every entry")
    extra_cols = [c for c in df.columns if c not in ('href', 'title')]

    head = ''.join(f'<th>{html.escape(str(c))}</th>' for c in ['probe'] + extra_cols)
    rows = []
    for _, r in df.iterrows():
        cells = (f'<td><a href="{html.escape(str(r["href"]), quote=True)}">'
                 f'{html.escape(str(r["title"]))}</a></td>')
        cells += ''.join(
            f'<td>{html.escape("" if pd.isna(r[c]) else str(r[c]))}</td>'
            for c in extra_cols)
        rows.append(f'<tr>{cells}</tr>')
    rows = '\n'.join(rows)

    t = html.escape(str(title))
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{t}</title>
<style>
 body {{ font-family: "Segoe UI", Helvetica, Arial, sans-serif; margin: 24px; color: #222; }}
 h1 {{ font-size: 18px; }}
 table {{ border-collapse: collapse; font-size: 13px; }}
 th, td {{ padding: 4px 14px; border-bottom: 1px solid #e0e0e0; text-align: left; }}
 th {{ background: #f5f5f5; }}
 a {{ color: #1f77b4; text-decoration: none; }} a:hover {{ text-decoration: underline; }}
</style></head><body>
<h1>{t}</h1>
<p>One page per probe. Click units, arrows, or matrix cells/labels to explore
connections, cross-correlograms, and spike waveforms.</p>
<table><tr>{head}</tr>
{rows}
</table></body></html>"""
