"""Generic signal and event preprocessing for mimo_pack."""

import warnings

import numpy as np
import pynapple as nap
import xarray as xr

from mimo_pack.consts import ATTR_UNITS, DIM_CHAN, DIM_TIME


def _get_channel(data: xr.DataArray, channel) -> xr.DataArray:
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"Expected xr.DataArray, got {type(data).__name__}.")
    if DIM_TIME not in data.dims or DIM_CHAN not in data.dims:
        raise ValueError(f"DataArray must have '{DIM_TIME}' and '{DIM_CHAN}' dimensions. Found: {data.dims}")
    try:
        return data.sel({DIM_CHAN: channel})
    except KeyError:
        raise ValueError(f"Channel {channel!r} not found. Available channels: {list(data[DIM_CHAN].values)}")


def _assert_binary(arr: np.ndarray, channel) -> None:
    unique = np.unique(arr)
    if not np.all(np.isin(np.round(unique).astype(int), [0, 1])):
        raise ValueError(
            f"Channel {channel!r} is not binary (values: {unique[:6]}). "
            "Use binarize() first or provide a binary signal."
        )


def _schmitt_mask(signal: np.ndarray, threshold: float, hysteresis: float) -> np.ndarray:
    upper = threshold + hysteresis / 2.0
    lower = threshold - hysteresis / 2.0
    mask = np.zeros(len(signal), dtype=bool)
    state = signal[0] >= upper
    for index, value in enumerate(signal):
        if state and value < lower:
            state = False
        elif not state and value >= upper:
            state = True
        mask[index] = state
    return mask


def digital_to_events(data: xr.DataArray, channel, edge: str = 'rising', debounce: float = 0.0) -> nap.Ts:
    """Detect edge events in a binary digital channel."""
    if edge not in ('rising', 'falling', 'both'):
        raise ValueError(f"edge must be 'rising', 'falling', or 'both'; got {edge!r}.")
    if isinstance(channel, (list, tuple)):
        events = [digital_to_events(data, item, edge=edge, debounce=debounce) for item in channel]
        group = nap.TsGroup({index: event for index, event in enumerate(events)})
        group.set_info(channel=np.array(channel))
        return group
    channel_data = _get_channel(data, channel)
    signal = channel_data.values.astype(float)
    _assert_binary(signal, channel)
    difference = np.diff(signal)
    if edge == 'rising':
        indices = np.where(difference == 1)[0] + 1
    elif edge == 'falling':
        indices = np.where(difference == -1)[0] + 1
    else:
        indices = np.where(difference != 0)[0] + 1
    event_times = channel_data[DIM_TIME].values.astype(float)[indices]
    if debounce > 0.0 and len(event_times) > 1:
        keep = np.ones(len(event_times), dtype=bool)
        last_kept = event_times[0]
        for index in range(1, len(event_times)):
            if event_times[index] - last_kept < debounce:
                keep[index] = False
            else:
                last_kept = event_times[index]
        event_times = event_times[keep]
    return nap.Ts(t=event_times)


def digital_to_intervals(data: xr.DataArray, channel, invert: bool = False, boundary: str = 'transitions') -> nap.IntervalSet:
    """Convert a binary digital channel into ON-period intervals."""
    if boundary not in ('transitions', 'trim'):
        raise ValueError(f"boundary must be 'transitions' or 'trim'; got {boundary!r}.")
    if isinstance(channel, (list, tuple)):
        return {item: digital_to_intervals(data, item, invert=invert, boundary=boundary) for item in channel}
    channel_data = _get_channel(data, channel)
    signal = channel_data.values.astype(float)
    _assert_binary(signal, channel)
    if invert:
        signal = 1.0 - signal
    difference = np.diff(signal)
    rising = np.where(difference == 1)[0] + 1
    falling = np.where(difference == -1)[0] + 1
    if boundary == 'transitions':
        if signal[0] == 1.0:
            rising = np.concatenate([[0], rising])
        if signal[-1] == 1.0:
            falling = np.concatenate([falling, [len(signal) - 1]])
    else:
        if len(falling) > 0 and (len(rising) == 0 or falling[0] < rising[0]):
            falling = falling[1:]
        if len(rising) > 0 and (len(falling) == 0 or rising[-1] > falling[-1]):
            rising = rising[:-1]
    count = min(len(rising), len(falling))
    if count == 0:
        return nap.IntervalSet(start=np.array([]), end=np.array([]))
    time = channel_data[DIM_TIME].values.astype(float)
    return nap.IntervalSet(start=time[rising[:count]], end=time[falling[:count]])


def analog_to_events(data: xr.DataArray, channel, threshold: float, edge: str = 'rising', hysteresis: float = 0.0) -> nap.Ts:
    """Detect threshold crossings in an analog channel."""
    if edge not in ('rising', 'falling', 'both'):
        raise ValueError(f"edge must be 'rising', 'falling', or 'both'; got {edge!r}.")
    if isinstance(channel, (list, tuple)):
        events = [analog_to_events(data, item, threshold=threshold, edge=edge, hysteresis=hysteresis) for item in channel]
        group = nap.TsGroup({index: event for index, event in enumerate(events)})
        group.set_info(channel=np.array(channel))
        return group
    channel_data = _get_channel(data, channel)
    mask = _schmitt_mask(channel_data.values.astype(float), threshold, hysteresis)
    difference = np.diff(mask.astype(int), prepend=0)
    if edge == 'rising':
        indices = np.where(difference == 1)[0]
    elif edge == 'falling':
        indices = np.where(difference == -1)[0]
    else:
        indices = np.where(difference != 0)[0]
    return nap.Ts(t=channel_data[DIM_TIME].values.astype(float)[indices])


def analog_to_intervals(data: xr.DataArray, channel, threshold: float, hysteresis: float = 0.0) -> nap.IntervalSet:
    """Convert an analog channel to ON-period intervals."""
    if isinstance(channel, (list, tuple)):
        return {item: analog_to_intervals(data, item, threshold=threshold, hysteresis=hysteresis) for item in channel}
    channel_data = _get_channel(data, channel)
    mask = _schmitt_mask(channel_data.values.astype(float), threshold, hysteresis)
    difference = np.diff(mask.astype(int))
    rising = np.where(difference == 1)[0] + 1
    falling = np.where(difference == -1)[0] + 1
    if len(falling) > 0 and (len(rising) == 0 or falling[0] < rising[0]):
        falling = falling[1:]
    if len(rising) > 0 and (len(falling) == 0 or rising[-1] > falling[-1]):
        rising = rising[:-1]
    count = min(len(rising), len(falling))
    if count == 0:
        return nap.IntervalSet(start=np.array([]), end=np.array([]))
    time = channel_data[DIM_TIME].values.astype(float)
    return nap.IntervalSet(start=time[rising[:count]], end=time[falling[:count]])


def cross_channel_intervals(data: xr.DataArray, start_channel, end_channel, *, start_threshold: float = None, end_threshold: float = None, hysteresis: float = 0.0, start_edge: str = 'rising', end_edge: str = 'rising', pair: str = 'first', on_extra: str = 'warn') -> nap.IntervalSet:
    """Pair events from two channels into intervals."""
    valid_edges = ('rising', 'falling', 'both')
    if start_edge not in valid_edges or end_edge not in valid_edges:
        raise ValueError(f"start_edge and end_edge must be one of {valid_edges}.")
    if pair not in ('first', 'last'):
        raise ValueError(f"pair must be 'first' or 'last'; got {pair!r}.")
    if on_extra not in ('warn', 'raise', 'ignore'):
        raise ValueError(f"on_extra must be 'warn', 'raise', or 'ignore'; got {on_extra!r}.")
    start_events = digital_to_events(data, start_channel, edge=start_edge) if start_threshold is None else analog_to_events(data, start_channel, threshold=start_threshold, edge=start_edge, hysteresis=hysteresis)
    end_events = digital_to_events(data, end_channel, edge=end_edge) if end_threshold is None else analog_to_events(data, end_channel, threshold=end_threshold, edge=end_edge, hysteresis=hysteresis)
    starts, ends = np.sort(start_events.t), np.sort(end_events.t)
    paired_starts, paired_ends = [], []
    if pair == 'first':
        start_index = 0
        for end in ends:
            if start_index < len(starts) and starts[start_index] < end:
                paired_starts.append(starts[start_index])
                paired_ends.append(end)
                start_index += 1
    else:
        start_index = len(starts) - 1
        for end in reversed(ends):
            while start_index >= 0 and starts[start_index] >= end:
                start_index -= 1
            if start_index >= 0:
                paired_starts.append(starts[start_index])
                paired_ends.append(end)
                start_index -= 1
        paired_starts.reverse()
        paired_ends.reverse()
    unmatched_starts = len(starts) - len(paired_starts)
    unmatched_ends = len(ends) - len(paired_ends)
    if unmatched_starts or unmatched_ends:
        message = f"cross_channel_intervals: {unmatched_starts} unmatched start event(s) and {unmatched_ends} unmatched end event(s) were discarded."
        if on_extra == 'warn':
            warnings.warn(message, UserWarning, stacklevel=2)
        elif on_extra == 'raise':
            raise ValueError(message)
    if not paired_starts:
        return nap.IntervalSet(start=np.array([]), end=np.array([]))
    return nap.IntervalSet(start=np.array(paired_starts), end=np.array(paired_ends))


def binarize(data: xr.DataArray, channel, threshold: float | None = None) -> xr.DataArray:
    """Threshold an analog channel to a binary DataArray."""
    if isinstance(channel, (list, tuple)):
        arrays = [binarize(data, item, threshold) for item in channel]
        return xr.DataArray(np.stack([array.values for array in arrays], axis=1), dims=[DIM_TIME, DIM_CHAN], coords={DIM_TIME: arrays[0][DIM_TIME], DIM_CHAN: list(channel)}, attrs=arrays[0].attrs)
    channel_data = _get_channel(data, channel)
    if threshold is None:
        threshold = (channel_data.values.min() + channel_data.values.max()) / 2
    previous = data.attrs.get('description', '')
    description = (previous + ' -> ' if previous else '') + f'binarize(channel={channel!r}, threshold={threshold})'
    return xr.DataArray((channel_data.values >= threshold).astype(np.float32), dims=[DIM_TIME], coords={DIM_TIME: channel_data[DIM_TIME]}, attrs={**data.attrs, 'description': description})


def quantize(data: xr.DataArray, channel, n_levels: int) -> xr.DataArray:
    """Quantize an analog channel into discrete integer levels."""
    if n_levels < 2:
        raise ValueError(f"n_levels must be >= 2; got {n_levels}.")
    if isinstance(channel, (list, tuple)):
        arrays = [quantize(data, item, n_levels) for item in channel]
        return xr.DataArray(np.stack([array.values for array in arrays], axis=1), dims=[DIM_TIME, DIM_CHAN], coords={DIM_TIME: arrays[0][DIM_TIME], DIM_CHAN: list(channel)}, attrs=arrays[0].attrs)
    channel_data = _get_channel(data, channel)
    edges = np.linspace(channel_data.values.min(), channel_data.values.max(), n_levels + 1)
    levels = np.clip(np.digitize(channel_data.values.astype(float), edges[1:-1]), 0, n_levels - 1).astype(np.int32)
    previous = data.attrs.get('description', '')
    description = (previous + ' -> ' if previous else '') + f'quantize(channel={channel!r}, n_levels={n_levels})'
    return xr.DataArray(levels, dims=[DIM_TIME], coords={DIM_TIME: channel_data[DIM_TIME]}, attrs={**data.attrs, 'description': description, 'n_levels': n_levels, 'bin_edges': edges})


def decode_quadrature(data: xr.DataArray, channel_a, channel_b, resolution: float = 1.0) -> xr.DataArray:
    """Decode binary quadrature encoder channels into cumulative position."""
    channel_a_data, channel_b_data = _get_channel(data, channel_a), _get_channel(data, channel_b)
    a, b = np.round(channel_a_data.values).astype(int), np.round(channel_b_data.values).astype(int)
    _assert_binary(a, channel_a)
    _assert_binary(b, channel_b)
    state = (a << 1) | b
    table = np.array([0, 1, -1, 0, -1, 0, 0, 1, 1, 0, 0, -1, 0, -1, 1, 0], dtype=np.int8)
    position = np.concatenate([[0.0], np.cumsum(table[state[:-1] * 4 + state[1:]] * resolution)])
    previous = data.attrs.get('description', '')
    description = (previous + ' -> ' if previous else '') + f'decode_quadrature(channel_a={channel_a!r}, channel_b={channel_b!r}, resolution={resolution})'
    return xr.DataArray(position, dims=[DIM_TIME], coords={DIM_TIME: channel_a_data[DIM_TIME].values.astype(float)}, attrs={**data.attrs, ATTR_UNITS: 'counts', 'description': description})


def track_pulse_frequency(data: xr.DataArray, channel, threshold: float = None, window: float = None) -> xr.DataArray:
    """Compute time-varying pulse frequency from a binary or analog channel."""
    if isinstance(channel, (list, tuple)):
        arrays = [track_pulse_frequency(data, item, threshold=threshold, window=window) for item in channel]
        return xr.DataArray(np.stack([array.values for array in arrays], axis=1), dims=[DIM_TIME, DIM_CHAN], coords={DIM_TIME: arrays[0][DIM_TIME], DIM_CHAN: list(channel)}, attrs=arrays[0].attrs)
    channel_data = _get_channel(data, channel)
    time = channel_data[DIM_TIME].values.astype(float)
    events = digital_to_events(data, channel) if threshold is None else analog_to_events(data, channel, threshold=threshold)
    event_times = events.times()
    previous = data.attrs.get('description', '')
    description = (previous + ' -> ' if previous else '') + f'track_pulse_frequency(channel={channel!r}, threshold={threshold}, window={window})'
    attrs = {**data.attrs, ATTR_UNITS: 'Hz', 'description': description}
    if len(event_times) < 2:
        values = np.zeros(len(time)) if window is not None else np.full(len(time), np.nan)
    elif window is not None:
        values = np.array([(np.searchsorted(event_times, current_time, side='right') - np.searchsorted(event_times, current_time - window, side='left')) / window for current_time in time])
    else:
        values = np.interp(time, 0.5 * (event_times[:-1] + event_times[1:]), 1.0 / np.diff(event_times), left=np.nan, right=np.nan)
    return xr.DataArray(values, dims=[DIM_TIME], coords={DIM_TIME: time}, attrs=attrs)


def pulse_stats(data: xr.DataArray, channel, threshold: float = None, edge: str = 'rising') -> nap.IntervalSet:
    """Characterize pulses from a digital or analog channel."""
    if isinstance(channel, (list, tuple)):
        return {item: pulse_stats(data, item, threshold=threshold, edge=edge) for item in channel}
    intervals = digital_to_intervals(data, channel) if threshold is None else analog_to_intervals(data, channel, threshold=threshold)
    if len(intervals) == 0:
        return intervals
    starts, ends = intervals.start, intervals.end
    isis = np.full(len(starts), np.nan)
    isis[:-1] = starts[1:] - starts[:-1]
    bout_ids = np.zeros(len(starts), dtype=int)
    pulse_in_bout = np.zeros(len(starts), dtype=int)
    valid_isis = isis[~np.isnan(isis)]
    if len(valid_isis):
        bout_threshold = 2.0 * np.median(valid_isis)
        bout, position = 0, 0
        for index in range(len(starts)):
            bout_ids[index], pulse_in_bout[index] = bout, position
            if index < len(isis) - 1 and isis[index] > bout_threshold:
                bout, position = bout + 1, 0
            else:
                position += 1
    result = nap.IntervalSet(start=starts, end=ends)
    result.set_info(duration=ends - starts, isi=isis, bout_id=bout_ids, pulse_in_bout=pulse_in_bout)
    return result


def detect_bouts(events: nap.Ts, gap_threshold: float) -> nap.IntervalSet:
    """Group event timestamps into bouts separated by large gaps."""
    if not isinstance(events, nap.Ts):
        raise TypeError(f"Expected pynapple.Ts, got {type(events).__name__}.")
    if gap_threshold <= 0:
        raise ValueError(f"gap_threshold must be positive; got {gap_threshold}.")
    times = events.times()
    if len(times) == 0:
        return nap.IntervalSet(start=np.array([]), end=np.array([]))
    boundaries = np.where(np.diff(times) > gap_threshold)[0]
    starts = np.concatenate([[times[0]], times[boundaries + 1]])
    ends = np.concatenate([times[boundaries], [times[-1]]])
    ends[ends == starts] += 1e-9
    return nap.IntervalSet(start=starts, end=ends)

__all__ = [
    'digital_to_events',
    'digital_to_intervals',
    'analog_to_events',
    'analog_to_intervals',
    'cross_channel_intervals',
    'binarize',
    'quantize',
    'decode_quadrature',
    'track_pulse_frequency',
    'pulse_stats',
    'detect_bouts',
]
