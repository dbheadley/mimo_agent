"""NIDQ-specific event extraction for mimo_pack."""

import warnings

import xarray as xr

from mimo_pack.preprocess.signal import (
    analog_to_events,
    analog_to_intervals,
    digital_to_events,
    digital_to_intervals,
)


def extract_nidq_events(
        data: xr.DataArray,
        channel_config: list,
) -> dict:
    """Extract events from labeled NIDQ channels.

    Parameters
    ----------
    data : xr.DataArray
        NIDQ data as returned by ``mimo_pack.fileio.nidq.load_nidq``.
    channel_config : list of dict
        Channel metadata with ``label`` and ``type`` keys. Recognized types
        are ``digital``, ``ttl``, ``analog``, ``sync``, and ``other``.
        Analog channels require a ``threshold`` value.

    Returns
    -------
    dict
        Event and interval objects keyed by ``"{label}.events"`` and
        ``"{label}.intervals"``. Sync channels return events only.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError(
            f"Expected xr.DataArray, got {type(data).__name__}. "
            "Use mimo_pack.fileio.nidq.load_nidq() first."
        )

    result = {}
    known_types = {'digital', 'ttl', 'analog', 'sync', 'other'}

    for entry in channel_config:
        label = entry.get('label', entry.get('channel', ''))
        channel_type = str(entry.get('type', 'other')).lower()

        if channel_type == 'other':
            continue
        if channel_type not in known_types:
            warnings.warn(
                f"Unknown channel type {channel_type!r} for label {label!r}; skipping.",
                UserWarning,
                stacklevel=2,
            )
            continue

        if channel_type in ('digital', 'ttl'):
            result[f'{label}.events'] = digital_to_events(data, label)
            result[f'{label}.intervals'] = digital_to_intervals(data, label)
        elif channel_type == 'sync':
            result[f'{label}.events'] = digital_to_events(data, label)
        else:
            threshold = entry.get('threshold')
            if threshold is None:
                raise ValueError(
                    f"Analog channel {label!r} has no 'threshold' key in channel_config."
                )
            result[f'{label}.events'] = analog_to_events(
                data, label, threshold=float(threshold)
            )
            result[f'{label}.intervals'] = analog_to_intervals(
                data, label, threshold=float(threshold)
            )

    return result


__all__ = ['extract_nidq_events']
