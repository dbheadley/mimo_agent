# Antidromic stimulation

import numpy as np
import pandas as pd
from dclut import dclut

from mimo_pack.consts import DIM_TIME
from mimo_pack.analysis.lfp import evoked_lfp
from mimo_pack.analysis.ap import evoked_ap
from mimo_pack.analysis.evoked import evoked_field_summary
from mimo_pack.fileio.ap import load_ap_windows
from mimo_pack.fileio.lfp import load_lfp_windows


_STIM_SUMMARY_COLUMNS = [
    'stim_times',
    'start_time',
    'end_time',
    'num_pulses',
    'duration',
    'mean_interval',
]


def _empty_stim_summary():
    stim_df = pd.DataFrame(columns=_STIM_SUMMARY_COLUMNS)
    stim_df.index.name = 'stim_group'
    return stim_df


def _stim_channel_values(data):
    values = np.asarray(data.values).squeeze()
    if values.ndim != 1:
        raise ValueError("Expected a single stimulation channel for antidromic pulse detection.")
    return values


def _max_stim_channel_voltage(data):
    values = _stim_channel_values(data)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.nan
    return float(np.max(finite_values))


def _detect_antistim_pulses(data, start_time_s=0.0, voltage_threshold=None):
    """Detect stimulation pulse starts."""
    values = _stim_channel_values(data)

    times = data[DIM_TIME].values
    if values.size == 0:
        return np.array([], dtype=times.dtype)

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.array([], dtype=times.dtype)

    max_peak = np.max(finite_values)
    if voltage_threshold is None:
        if not np.isfinite(max_peak) or max_peak <= 0:
            return np.array([], dtype=times.dtype)
        threshold = max_peak / 2
    else:
        threshold = float(voltage_threshold)
        if not np.isfinite(threshold):
            raise ValueError("voltage_threshold must be finite when supplied.")

    bin_data = values > threshold
    transitions = np.diff(bin_data.astype(int))
    pulse_start_idxs = np.where(transitions == 1)[0] + 1
    pulse_starts = times[pulse_start_idxs]

    if start_time_s is not None and start_time_s > 0:
        pulse_starts = pulse_starts[pulse_starts >= start_time_s]

    return np.asarray(pulse_starts, dtype=times.dtype)


def get_antistim_times(
    dclut_path,
    channel,
    start_time_s=0.0,
    voltage_threshold=None,
    return_metadata=False,
):
    """Extract antidromic stimulation times from an NIDQ digital channel.

    Reads a continuous digital/analog stimulus signal from the specified
    NIDQ channel, thresholds at half-maximum unless ``voltage_threshold``
    is supplied, and groups detected pulses into stimulation bouts separated by gaps > 2× the median inter-pulse
    interval.

    Parameters
    ----------
    dclut_path : str
        Path to the NIDQ dclut JSON file.
    channel : int or list of int
        Channel index (or indices) on the NIDQ recording that carries
        the stimulation trigger signal.
    start_time_s : float, optional
        Recording time in seconds before which detected pulse starts are
        ignored. Defaults to 0.0, which preserves detection across the
        full recording.
    voltage_threshold : float, optional
        Voltage threshold used for rising-edge pulse detection. When omitted,
        detection uses half of the maximum value on the selected stimulation
        channel.
    return_metadata : bool, optional
        If True, return ``(stim_df, metadata)`` where metadata includes the
        maximum voltage measured on the selected stimulation channel.

    Returns
    -------
    pandas.DataFrame
        One row per stimulation bout with columns:

        - ``stim_times`` : array of pulse-onset timestamps (s)
        - ``start_time`` : first pulse time in the bout
        - ``end_time`` : last pulse time in the bout
        - ``num_pulses`` : number of pulses in the bout
        - ``duration`` : bout duration (s)
        - ``mean_interval`` : mean inter-pulse interval within the bout
    """
    nidq = dclut(dclut_path)
    nidq.reset()
    nidq.points({'channel': [channel]})
    data = nidq.read(format='xarray')[0]
    metadata = {'max_voltage': _max_stim_channel_voltage(data)}

    pulse_starts = _detect_antistim_pulses(
        data,
        start_time_s=start_time_s,
        voltage_threshold=voltage_threshold,
    )
    if len(pulse_starts) == 0:
        stim_df = _empty_stim_summary()
        if return_metadata:
            return stim_df, metadata
        return stim_df

    if len(pulse_starts) > 1:
        med_interval = np.median(np.diff(pulse_starts))
        stim_grps = np.split(pulse_starts, np.where(np.diff(pulse_starts) > med_interval*2)[0]+1)
    else:
        stim_grps = [pulse_starts]

    stim_summary = []
    for grp in stim_grps:
        start_time = grp[0]
        end_time = grp[-1]
        num_pulses = len(grp)
        duration = end_time - start_time
        if num_pulses > 1:
            mean_interval = np.mean(np.diff(grp))
        else:
            mean_interval = np.nan
        stim_summary.append({
            'stim_times': grp,
            'start_time': start_time,
            'end_time': end_time,
            'num_pulses': num_pulses,
            'duration': duration,
            'mean_interval': mean_interval
        })

    stim_df = pd.DataFrame(stim_summary, columns=_STIM_SUMMARY_COLUMNS)
    stim_df.index.name = 'stim_group'
    if return_metadata:
        return stim_df, metadata
    return stim_df


def create_stimulation_summary(stim_df):
    """
    Create a summary table with key statistics for each stimulation group.
    
    Parameters
    ----------
    stim_df : pd.DataFrame
        DataFrame containing stimulation data with 'stim_times' column.
        
    Returns
    -------
    pd.DataFrame
        Summary DataFrame with columns:
        - stim_group: Group index
        - start_time_s: When stimulation started
        - end_time_s: When stimulation ended
        - duration_s: Total duration
        - n_pulses: Number of stimulation pulses
        - mean_interval_s: Average time between pulses
        - frequency_hz: Stimulation frequency
    """
    summary_data = []
    
    for idx, row in stim_df.iterrows():
        stim_times = row['stim_times']
        
        # Calculate statistics
        start_time = stim_times.min()
        end_time = stim_times.max()
        n_pulses = len(stim_times)
        
        # Calculate mean interval (if more than 1 pulse)
        if n_pulses > 1:
            intervals = np.diff(stim_times)
            mean_interval = intervals.mean()
        else:
            mean_interval = np.nan
        
        summary_data.append({
            'stim_group': idx,
            'start_time_s': start_time,
            'end_time_s': end_time,
            'duration_s': end_time - start_time,
            'n_pulses': n_pulses,
            'mean_interval_s': mean_interval,
            'frequency_hz': 1/mean_interval if not np.isnan(mean_interval) else np.nan
        })
    
    return pd.DataFrame(summary_data)


def get_antidromic_evoked_summary(ap_dclut, lfp_dclut, stim_times, pre=0.005, post=0.02):
    """
    Generate evoked field summary for antidromic stimulation.
    
    Parameters
    ----------
    ap_dclut : str or Path
        Path to the AP dclut file.
    lfp_dclut : str or Path
        Path to the LFP dclut file.
    stim_times : array-like
        Array of stimulation times to center windows on.
    pre : float, optional
        Time before stimulation in seconds (default: 0.005, i.e., 5ms).
    post : float, optional
        Time after stimulation in seconds (default: 0.02, i.e., 20ms).
        
    Returns
    -------
    dict
        Dictionary containing 'ap', 'lfp', and 'csd' keys with evoked responses.
        
    Notes
    -----
    This function loads windowed data around stimulation times and computes
    the evoked field summary including AP envelope, LFP mean, and CSD.
    Windows are typically 100ms total (50ms pre + 50ms post stimulation).
    """
    # Load windows centered on stim times
    ap_windowed = load_ap_windows(ap_dclut, centers=stim_times, pre=pre, post=post)
    lfp_windowed = load_lfp_windows(lfp_dclut, centers=stim_times, pre=pre, post=post)
    
    # Generate evoked field summary
    summary = evoked_field_summary(ap_windowed, lfp_windowed)
    
    return summary

