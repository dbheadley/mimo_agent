""" Organizes data from in vitro patch clamp experiments
Created by: Drew Headley
"""

import xarray as xr
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

from mimo_pack.consts import DIM_TIME, DIM_CHAN, DIM_EPOCH, ATTR_FS

# for each image in imgs_xr, identify the x,y location of the center of mass. If no pixels are above 0, then return np.nan
def _center_of_mass(image: np.ndarray) -> np.ndarray:
    """Calculate the (y, x) centre of mass of a 2D image.

    Parameters
    ----------
    image : numpy.ndarray
        2-D array of pixel intensities.

    Returns
    -------
    numpy.ndarray
        Length-2 array ``[y_center, x_center]``. Returns ``[NaN, NaN]``
        if all pixels are zero.
    """
    if np.all(image == 0):
        return np.array([np.nan, np.nan])  # No pixels above 0

    y_indices, x_indices = np.indices(image.shape)
    total_mass = np.sum(image)

    if total_mass == 0:
        return np.array([np.nan, np.nan])  # Avoid division by zero

    y_center = np.sum(y_indices * image) / total_mass
    x_center = np.sum(x_indices * image) / total_mass

    return np.array([y_center, x_center])

def format_ivic(rec: xr.DataArray) -> xr.DataArray:
    """Format IV current-clamp recording with stimulus coordinates.

    Attaches injected current as a coordinate along DIM_EPOCH. The time axis
    is re-zeroed to the onset of the current step.

    Parameters
    ----------
    rec : xr.DataArray
        The recording data with a DIM_EPOCH dimension containing
        current-clamp sweeps.

    Returns
    -------
    xr.DataArray
        The recording data with:

        - Coordinate 'i_inject' on DIM_EPOCH: injected current per sweep (pA).
        - Coordinate 'hold_current' on DIM_EPOCH: baseline holding current per sweep (pA).
        - Time axis re-zeroed to current step onset.
        - Attribute 'step_dur': duration of current step (s).
        - Attribute 'stimulus_properties': ['i_inject'].
    """

    # get the current injection channel
    inj_ch = rec['ch_units'].to_numpy().tolist().index('pA')
    inj_current = rec.sel({DIM_CHAN: inj_ch})
    hold_current = inj_current.median(dim=DIM_TIME)
    inj_current = inj_current - hold_current  # remove DC offset

    time_data = inj_current[DIM_TIME].to_numpy()
    step_start_idx = np.where(inj_current.sel({DIM_EPOCH: 0}).data <=-150)[0][0]
    step_stop_idx = np.where(inj_current.sel({DIM_EPOCH: 0}).data <= -150)[0][-1]
    step_start_time = time_data[step_start_idx]
    step_stop_time = time_data[step_stop_idx]
    step_dur = step_stop_time - step_start_time

    # replace time dimension with step_start_time
    rec = rec.assign_coords({DIM_TIME: time_data - step_start_time})

    # Attach stimulus property as coordinate on DIM_EPOCH
    inj_current_vals = inj_current.sel({DIM_TIME: step_start_time+0.005}).data
    hold_current_vals = hold_current.data
    rec = rec.assign_coords({
        'i_inject': (DIM_EPOCH, inj_current_vals),
        'hold_current': (DIM_EPOCH, hold_current_vals),
    })

    rec.attrs['step_dur'] = step_dur
    rec.attrs['stimulus_properties'] = ['i_inject']
    
    # Add description for data provenance
    desc_base = rec.attrs.get('description', 'IV clamp recording')
    rec.attrs['description'] = f"{desc_base} -> formatted with current injection info"

    return rec

def format_focal_stimulation(
    rec: xr.DataArray,
    imgs: xr.DataArray,
    px_per_um: float = float('nan'),
) -> xr.DataArray:
    """Format focal stimulation recording with spatial stimulus coordinates.

    Attaches the (x, y) centroid of each stimulus image as coordinates along
    DIM_EPOCH. The time axis is re-zeroed to the first stimulation pulse.

    Parameters
    ----------
    rec : xr.DataArray
        The recording data with a DIM_EPOCH dimension.
    imgs : xr.DataArray
        The images used for focal stimulation, with an 'image' dimension.
    px_per_um : float, optional
        Pixel-to-micron conversion factor (pixels per µm). When provided,
        ``stim_x`` and ``stim_y`` coordinates are divided by this value so
        that they are expressed in µm rather than pixels. When ``nan``
        (default), coordinates are left in pixel units.

    Returns
    -------
    xr.DataArray
        The recording data with:

        - Coordinates 'stim_x', 'stim_y' on DIM_EPOCH: centroid of each
          stimulus in µm (if ``px_per_um`` is given) or pixels.
        - Coordinate 'image_name' on DIM_EPOCH: filename of each stimulus image.
        - Coordinate 'rep' on DIM_EPOCH: 0-indexed repetition block for each trial.
        - Time axis re-zeroed to stimulation onset.
        - Attribute 'led_power': LED power level.
        - Attribute 'px_per_um': conversion factor used (nan if not supplied).
        - Attribute 'stimulus_properties': ['stim_x', 'stim_y', 'image_name'].
    """

    # Apply the center_of_mass function to each image
    centers = xr.apply_ufunc(
        _center_of_mass,
        imgs,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["center"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    
    # Assign coordinate names to the center dimension
    centers = centers.assign_coords(center=["y", "x"])

    # Extract the centers of mass for each pattern
    stim_cents = centers.data

    # Convert pixel centroids to µm if a calibration factor was supplied
    if not math.isnan(float(px_per_um)):
        stim_cents = stim_cents / px_per_um

    # number of trials
    n_trials = rec[DIM_EPOCH].size
    n_imgs = imgs['image'].size
    img_inds = np.mod(np.arange(n_trials), n_imgs)
    rep = np.floor(np.arange(n_trials) / n_imgs).astype(int)

    # get start of first stimulation pulse
    led_pow_ch = rec['ch_names'].to_numpy().tolist().index('LEDPow')
    led_pow = rec.sel({DIM_CHAN: led_pow_ch, DIM_EPOCH: 0}).data
    start_idx = np.where(led_pow >= 0.1)[0][0]
    start_time = rec[DIM_TIME].to_numpy()[start_idx]
    stim_time = rec[DIM_TIME].to_numpy() - start_time

    # replace time dimension with stim_time
    rec = rec.assign_coords({DIM_TIME: stim_time})

    # get the LED power level
    led_pow_lvl = np.max(led_pow)

    # add coordinate for stim_cents to iv
    rec.coords['stim_x'] = (DIM_EPOCH, stim_cents[img_inds,0])
    rec.coords['stim_y'] = (DIM_EPOCH, stim_cents[img_inds,1])
    rec.coords['image_name'] = (DIM_EPOCH, imgs['image_name'].values[img_inds])
    rec.coords['rep'] = (DIM_EPOCH, rep)

    # add protocol attributes
    rec.attrs['led_power'] = led_pow_lvl
    rec.attrs['px_per_um'] = px_per_um
    rec.attrs['stimulus_properties'] = ['stim_x', 'stim_y', 'image_name']
    
    # Add description for data provenance
    desc_base = rec.attrs.get('description', 'focal stimulation recording')
    rec.attrs['description'] = f"{desc_base} -> aligned with stimulation coordinates"

    return rec

def format_pattern_stim(patt_xr: xr.DataArray) -> xr.DataArray:
    """Format patterned stimulation recording with stimulus coordinate.

    Attaches the peak LED power as a coordinate along DIM_EPOCH. The time
    axis is re-zeroed to the onset of the patterned stimulation.

    Parameters
    ----------
    patt_xr : xr.DataArray
        The recording data with a DIM_EPOCH dimension containing
        patterned stimulation sweeps.

    Returns
    -------
    xr.DataArray
        The recording data with:

        - Coordinate 'patt_peak' on DIM_EPOCH: peak LED power per pattern.
        - Time axis re-zeroed to pattern onset.
        - Attribute 'patt_dur': duration of the pattern stimulation (s).
        - Attribute 'stimulus_properties': ['patt_peak'].
    """

    # get number of trials
    n_trials = patt_xr.sizes[DIM_EPOCH]

    # get the LED channel
    ch_idx = patt_xr['ch_names'].values.tolist().index('LEDPow')
    led = patt_xr.sel({DIM_CHAN: ch_idx})
    patt_peak = led.max(dim=DIM_TIME).data

    time_data = led[DIM_TIME].to_numpy()
    patt_start_idx = np.where(led.sel({DIM_EPOCH: n_trials-1}).data >= 0.02)[0][0]
    patt_stop_idx = np.where(led.sel({DIM_EPOCH: 0}).data >= 0.02)[0][-1]
    patt_start_time = time_data[patt_start_idx]
    patt_stop_time = time_data[patt_stop_idx]
    patt_dur = patt_stop_time - patt_start_time

    # replace time dimension with step_start_time
    patt_xr = patt_xr.assign_coords({DIM_TIME: time_data - patt_start_time})

    # add patterned properties to patt_xr
    patt_xr.coords['patt_peak'] = (DIM_EPOCH, patt_peak)
    patt_xr.attrs['patt_dur'] = patt_dur
    patt_xr.attrs['stimulus_properties'] = ['patt_peak']
    
    # Add description for data provenance
    desc_base = patt_xr.attrs.get('description', 'pattern stimulation recording')
    patt_xr.attrs['description'] = f"{desc_base} -> organized by stimulus patterns"

    return patt_xr

def format_var_ramp(vr_xr: xr.DataArray) -> xr.DataArray:
    """Format variable ramp recording with stimulus coordinate.

    Attaches the peak ramp intensity as a coordinate along DIM_EPOCH. The
    time axis is re-zeroed to the onset of the light ramp.

    Parameters
    ----------
    vr_xr : xr.DataArray
        The recording data with a DIM_EPOCH dimension containing
        variable ramp sweeps.

    Returns
    -------
    xr.DataArray
        The recording data with:

        - Coordinate 'ramp_peak' on DIM_EPOCH: peak ramp intensity per trial.
        - Time axis re-zeroed to ramp onset.
        - Attribute 'ramp_dur': duration of the ramp stimulation (s).
        - Attribute 'stimulus_properties': ['ramp_peak'].
    """

    # get number of trials
    n_trials = vr_xr.sizes[DIM_EPOCH]

    # get the LED ramp channel
    ch_idx = vr_xr['ch_names'].values.tolist().index('IN 5')
    ramp_led = vr_xr.sel({DIM_CHAN: ch_idx})
    ramp_peak = ramp_led.max(dim=DIM_TIME).data

    time_data = ramp_led[DIM_TIME].to_numpy()
    ramp_start_idx = np.where(ramp_led.sel({DIM_EPOCH: n_trials-1}).data >= 0.02)[0][0]
    ramp_stop_idx = np.where(ramp_led.sel({DIM_EPOCH: 0}).data >= 0.02)[0][-1]
    ramp_start_time = time_data[ramp_start_idx]
    ramp_stop_time = time_data[ramp_stop_idx]
    ramp_dur = ramp_stop_time - ramp_start_time

    # replace time dimension with step_start_time
    vr_xr = vr_xr.assign_coords({DIM_TIME: time_data - ramp_start_time})

    # add ramp properties to vr_xr
    vr_xr.coords['ramp_peak'] = (DIM_EPOCH, ramp_peak)
    vr_xr.attrs['ramp_dur'] = ramp_dur
    vr_xr.attrs['stimulus_properties'] = ['ramp_peak']
    
    # Add description for data provenance
    desc_base = vr_xr.attrs.get('description', 'variable ramp recording')
    vr_xr.attrs['description'] = f"{desc_base} -> organized by ramp parameters"

    return vr_xr


def format_blue_tonic_led(
    rec: xr.DataArray,
    led_channel=None,
    threshold: float = 0.1,
    min_duration: float = 0.01,
    duration: float = 0.5,
    onset: float = 0.235,
) -> xr.DataArray:
    """Format a blue tonic LED current-clamp recording with stimulus coordinates.

    Detects LED pulses per epoch, re-zeros the time axis to the LED onset,
    and attaches per-epoch LED duration and power as coordinates on DIM_EPOCH.

    Parameters
    ----------
    rec : xr.DataArray
        Raw recording with dims (DIM_TIME, DIM_CHAN, DIM_EPOCH) and 'ch_names'
        coordinate on DIM_CHAN.
    led_channel : str or list of str, optional
        Channel name(s) to search for in 'ch_names'. When a list is given, the
        first name found in the recording is used. Defaults to
        ``['LEDPow', 'BlueLED', 'IN 5']``.
    threshold : float
        Voltage level (V) above which the LED is considered ON. Default 0.1.
    min_duration : float
        Minimum valid pulse duration in seconds. Epochs with a detected pulse
        shorter than this are assigned NaN for led_dur and led_power. Default 0.01.
    duration : float or None
        Fallback LED pulse duration (s) written to ``led_dur`` for epochs where
        automatic per-epoch detection fails (no threshold crossing or pulse
        shorter than ``min_duration``). Only applied to epochs where detection
        yields NaN; successfully detected durations are never overwritten.
        Set to ``None`` to leave failed epochs as NaN. Default 0.5.
    onset : float or None
        Fallback for the LED onset time (s) used to re-zero the time axis.
        Automatic threshold-crossing detection on epoch 0 is always attempted
        first; this value is used only when detection fails. Set to ``None``
        to raise a ``ValueError`` if auto-detection fails. Default 0.235.

    Returns
    -------
    xr.DataArray
        Recording with:

        - Time axis re-zeroed to LED onset.
        - Coordinate 'led_dur' on DIM_EPOCH: LED pulse duration per epoch (s).
        - Coordinate 'led_power' on DIM_EPOCH: peak LED power per epoch (V).
        - Attribute 'stimulus_properties': ['led_dur', 'led_power'].
        - Attribute 'led_onset': detected LED onset time (s) used to re-zero, before re-zeroing.
        - Attribute 'led_dur': median LED pulse duration across epochs (s).
        - Attribute 'led_power': median peak LED power across epochs (V).

    Raises
    ------
    TypeError
        If rec is not an xr.DataArray.
    ValueError
        If none of the led_channel name(s) are found in ch_names, or if
        automatic onset detection finds no LED pulse in epoch 0 above threshold
        and ``onset`` is ``None``.
    """
    if not isinstance(rec, xr.DataArray):
        raise TypeError(
            f"rec must be an xr.DataArray, got {type(rec).__name__}. "
            "Load the recording with load_abf() first."
        )

    if led_channel is None:
        led_channel = ['LEDPow', 'BlueLED', 'IN 5']

    ch_names = rec['ch_names'].to_numpy().tolist()

    # Resolve a list of candidates to the first match found in the recording
    if isinstance(led_channel, (list, tuple)):
        candidates = led_channel
        matched = next((c for c in candidates if c in ch_names), None)
        if matched is None:
            raise ValueError(
                f"None of the LED channel candidates {candidates} were found "
                f"in ch_names: {ch_names}. Set led_channel to the correct name."
            )
        led_channel = matched
    else:
        if led_channel not in ch_names:
            raise ValueError(
                f"LED channel '{led_channel}' not found in ch_names: {ch_names}. "
                "Set led_channel to the correct channel name string."
            )

    led_ch_idx = ch_names.index(led_channel)
    led = rec.sel({DIM_CHAN: led_ch_idx})
    time_data = rec[DIM_TIME].to_numpy()
    n_trials = rec.sizes[DIM_EPOCH]

    # Per-epoch LED duration and power
    led_dur_vals = np.full(n_trials, np.nan)
    led_power_vals = np.full(n_trials, np.nan)

    for ep in range(n_trials):
        led_trace = led.sel({DIM_EPOCH: ep}).to_numpy()
        above = led_trace >= threshold
        if not np.any(above):
            continue
        onset_idx = int(np.where(above)[0][0])
        offset_idx = int(np.where(above)[0][-1])
        dur = time_data[offset_idx] - time_data[onset_idx]
        if dur < min_duration:
            continue
        led_dur_vals[ep] = dur
        led_power_vals[ep] = float(np.max(led_trace))

    # Apply fallback duration to epochs where per-epoch detection failed
    if duration is not None:
        led_dur_vals[np.isnan(led_dur_vals)] = duration

    # Determine onset time: always attempt auto-detection first;
    # use onset as fallback if detection fails; raise only if both fail.
    led_trace_ep0 = led.sel({DIM_EPOCH: 0}).to_numpy()
    above_ep0 = led_trace_ep0 >= threshold
    if np.any(above_ep0):
        onset_idx_ep0 = int(np.where(above_ep0)[0][0])
        onset_time = float(time_data[onset_idx_ep0])
    elif onset is not None:
        onset_time = float(onset)
    else:
        raise ValueError(
            f"No LED pulse detected in epoch 0 at threshold={threshold} V and "
            "no fallback onset provided. Check the threshold, led_channel, or set onset explicitly."
        )

    rec = rec.assign_coords({DIM_TIME: time_data - onset_time})
    rec = rec.assign_coords({
        'led_dur': (DIM_EPOCH, led_dur_vals),
        'led_power': (DIM_EPOCH, led_power_vals),
    })

    rec.attrs['stimulus_properties'] = ['led_dur', 'led_power']
    rec.attrs['led_onset'] = onset_time
    rec.attrs['led_dur'] = float(np.nanmedian(led_dur_vals)) if not np.all(np.isnan(led_dur_vals)) else np.nan
    rec.attrs['led_power'] = float(np.nanmedian(led_power_vals)) if not np.all(np.isnan(led_power_vals)) else np.nan
    desc_base = rec.attrs.get('description', 'blue tonic LED recording')
    rec.attrs['description'] = f"{desc_base} -> formatted with tonic LED parameters"

    return rec


def remove_ap(abf_xr: xr.DataArray,
                              voltage_channel,
                              peak_threshold: float = 0.0,
                              pre: float = 1.0,
                              post: float = 3.0,
                              interp_method: str = 'linear') -> xr.DataArray:
    """
    Remove action potentials (APs) from an ABF xarray.DataArray by replacing spike windows
    with interpolated values. Operates along the 'time' dimension and preserves other dims.

    Parameters
    ----------
    abf_xr : xr.DataArray
        ABF recording with a 'time' dimension and a 'channel' coordinate.
    voltage_channel :
        Value used to select the voltage channel (passed to abf_xr.sel(channel=...)).
    peak_threshold : float
        Voltage threshold for classifying AP samples (default 0.0).
    pre : float
        Milliseconds before spike peak to include in removal window (default 0.001).
    post : float
        Milliseconds after spike peak to include in removal window (default 0.003).
    interp_method : str
        Interpolation method for scipy.interp1d (e.g. 'linear', 'nearest', 'cubic').

    Returns
    -------
    xr.DataArray
        New DataArray with the selected channel's spikes removed (interpolated).
    """

    # precompute kernel size and convert pre/post to samples
    s_rate = abf_xr.attrs.get(ATTR_FS, abf_xr.attrs.get('sample_rate'))
    kern_size = int(0.0005 * s_rate)
    pre = int(s_rate * float(pre) / 1000)
    post = int(s_rate * float(post) / 1000)

    # define the spike removal function to apply along 'time'
    def _remove_spikes_np(v):
        """Remove detected spikes from a 1-D voltage trace via interpolation."""
        # voltage, time_values are 1D numpy arrays
        if v is None or v.size == 0:
            return v

        # find local maxima whose height is above peak_threshold
        peaks = find_peaks(v, height=peak_threshold)[0]
        if peaks.size == 0:
            return v

        remove_mask = np.zeros(v.shape, dtype=bool)
        for peak_idx in peaks:
            # define removal window in time around the detected peak
            start_idx = peak_idx - pre
            end_idx = peak_idx + post
            start_idx = max(0, start_idx)
            end_idx = min(len(t) - 1, end_idx)
            remove_mask[start_idx:end_idx + 1] = True

        # if nothing left to interpolate from, return original
        if (~remove_mask).sum() < 2:
            return v

        # perform interpolation across removed samples
        keep_t = t[~remove_mask]

        v_smooth = np.convolve(v, np.ones(kern_size)/kern_size, mode='same')
        keep_v = v_smooth[~remove_mask]
        interp_fn = interp1d(keep_t, keep_v, kind=interp_method, bounds_error=False, fill_value="extrapolate")
        v_removed = v.copy()
        v_removed[remove_mask] = interp_fn(t[remove_mask])
        return v_removed

    # get time values once
    if DIM_TIME not in abf_xr.dims and DIM_TIME not in abf_xr.coords:
        raise ValueError(f"abf_xr must have a '{DIM_TIME}' coordinate/dimension")

    t = abf_xr[DIM_TIME].values

    # select the channel to process (keeps other dims if present)
    target = abf_xr.sel({DIM_CHAN: voltage_channel}).copy()
    
    # apply function along 'time' core dim for all other dims
    cleaned = xr.apply_ufunc(
        _remove_spikes_np,
        target,
        input_core_dims=[[DIM_TIME]],
        output_core_dims=[[DIM_TIME]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[target.dtype]
    )

    # place cleaned channel back into a copy of the original DataArray
    abf_out = abf_xr.copy()
    abf_out.loc[{DIM_CHAN: voltage_channel}] = cleaned

    return abf_out


def get_holding_potential(rec: xr.DataArray,
                          pre_window: float | None = None) -> xr.Dataset:
    """Return the median holding potential per epoch for each voltage channel.

    Identifies voltage channels (``ch_units == 'mV'``) and computes the median
    over a baseline time window for each epoch.  The baseline is either an
    explicit pre-stimulus window or, when the recording has been formatted with
    :func:`format_blue_tonic_led`, the pre-stimulus period (``time < 0``) is
    used automatically.  If no negative times are present and ``pre_window`` is
    ``None``, the entire epoch is used.

    Parameters
    ----------
    rec : xr.DataArray
        Recording as returned by :func:`~mimo_pack.fileio.abf.load_abf` or one
        of the ``format_*`` preprocessing functions.  Must contain ``ch_units``
        and ``ch_names`` coordinates on the channel dimension.
    pre_window : float or None, optional
        Duration (s) of the baseline window measured backward from
        ``time == 0``.  When ``None`` the function auto-selects: negative times
        if present, otherwise the full epoch.

    Returns
    -------
    xr.Dataset
        One variable per ``mV`` channel, each shaped ``(epoch,)``.  Variable
        names are ``holding_{ch_name}`` using the ``ch_names`` coordinate.
        Each variable carries ``attrs['units'] = 'mV'``.

    Raises
    ------
    ValueError
        If ``rec.attrs['clamp_mode']`` is not ``'V-Clamp'``.
    RuntimeError
        If no channels with ``ch_units == 'mV'`` are found.
    """
    clamp_mode = rec.attrs.get('clamp_mode', 'unknown')
    if clamp_mode != 'V-Clamp':
        raise ValueError(
            f"Recording clamp_mode is '{clamp_mode}'; expected 'V-Clamp'. "
            "Use load_abf() to load the recording — the clamp_mode attribute is "
            "set automatically from telegraph metadata."
        )

    ch_units = rec['ch_units'].values
    mv_indices = np.where(ch_units == 'mV')[0]
    if mv_indices.size == 0:
        raise RuntimeError(
            "No channels with ch_units == 'mV' found. "
            f"Available units: {list(ch_units)}"
        )

    ch_names = rec['ch_names'].values

    # Determine baseline time window once, applied to all channels
    time = rec[DIM_TIME].values
    if pre_window is not None:
        t_sel = slice(-pre_window, 0.0)
        if rec.sel({DIM_TIME: t_sel}).sizes[DIM_TIME] == 0:
            t_sel = slice(time[0], time[0] + pre_window)
    elif (time < 0).any():
        t_sel = slice(None, 0.0)
    else:
        t_sel = slice(None, None)

    data_vars = {}
    for ch_idx in mv_indices:
        ch_name = str(ch_names[ch_idx])
        baseline = rec.isel({DIM_CHAN: ch_idx}).sel({DIM_TIME: t_sel})
        holding = baseline.median(dim=DIM_TIME)
        holding.attrs['units'] = 'mV'
        holding.attrs['description'] = f'median holding potential — channel {ch_name}'
        data_vars[f'holding_{ch_name}'] = holding

    return xr.Dataset(data_vars)


def remove_current_offset(rec: xr.DataArray,
                          pre_window: float | None = None) -> xr.DataArray:
    """Subtract the baseline current from each epoch of a V-Clamp recording.

    For every current channel (``ch_units == 'pA'``), the median value over a
    baseline time window is computed per epoch and subtracted from the full
    trace, zeroing the holding current so that stimulus-evoked deflections
    are measured relative to zero.

    Parameters
    ----------
    rec : xr.DataArray
        Recording with dims ``(time, channel, epoch)``.  Must have
        ``ch_units`` and ``ch_names`` coordinates on the channel dimension.
    pre_window : float or None, optional
        Duration (s) of the baseline window measured backward from
        ``time == 0``.  When ``None`` the function auto-selects: all negative
        times if present, otherwise the full epoch.

    Returns
    -------
    xr.DataArray
        Copy of *rec* with the current offset removed from every ``pA``
        channel.  Non-current channels are returned unchanged.  The removed
        offset for each channel is stored as a coordinate on the ``epoch``
        dimension named ``offset_{ch_name}`` (in pA).

    Raises
    ------
    ValueError
        If ``rec.attrs['clamp_mode']`` is not ``'V-Clamp'``.
    RuntimeError
        If no channels with ``ch_units == 'pA'`` are found.
    """
    clamp_mode = rec.attrs.get('clamp_mode', 'unknown')
    if clamp_mode != 'V-Clamp':
        raise ValueError(
            f"Recording clamp_mode is '{clamp_mode}'; expected 'V-Clamp'. "
            "Use load_abf() to load the recording — the clamp_mode attribute is "
            "set automatically from telegraph metadata."
        )

    ch_units = rec['ch_units'].values
    pa_indices = np.where(ch_units == 'pA')[0]
    if pa_indices.size == 0:
        raise RuntimeError(
            "No channels with ch_units == 'pA' found. "
            f"Available units: {list(ch_units)}"
        )

    # Determine baseline time window
    time = rec[DIM_TIME].values
    if pre_window is not None:
        t_sel = slice(-pre_window, 0.0)
        if rec.sel({DIM_TIME: t_sel}).sizes[DIM_TIME] == 0:
            t_sel = slice(time[0], time[0] + pre_window)
    elif (time < 0).any():
        t_sel = slice(None, 0.0)
    else:
        t_sel = slice(None, None)

    out = rec.copy()
    ch_names = rec['ch_names'].values
    for ch_idx in pa_indices:
        baseline = out.isel({DIM_CHAN: ch_idx}).sel({DIM_TIME: t_sel})
        offset = baseline.median(dim=DIM_TIME)          # shape (epoch,)
        out.loc[{DIM_CHAN: ch_idx}] = (
            out.isel({DIM_CHAN: ch_idx}) - offset
        )
        ch_name = str(ch_names[ch_idx])
        out = out.assign_coords({f'offset_{ch_name}': (DIM_EPOCH, offset.values)})

    return out


def remove_line_noise(
    rec: xr.DataArray,
    channels,
    line_freq: float = 60.0,
    half_width: float = 1.0,
    n_harmonics: int = 4,
) -> xr.DataArray:
    """Remove power-line (60 Hz) noise from selected channels via FFT zeroing.

    For each specified channel, the FFT is computed along the time axis,
    frequency bins around the fundamental and its harmonics up to
    ``n_harmonics`` are zeroed, and the trace is reconstructed with an
    inverse FFT.

    Parameters
    ----------
    rec : xr.DataArray
        Recording with at least a ``DIM_TIME`` dimension.  The ``ATTR_FS``
        attribute (sample rate in Hz) must be present.
    channels : int, list of int, or array-like
        Channel index value(s) along ``DIM_CHAN`` to denoise.  Non-selected
        channels are returned unchanged.
    line_freq : float
        Fundamental frequency of the power-line noise in Hz. Default 60.0.
    half_width : float
        Half-bandwidth around the *fundamental* harmonic to zero (Hz).  The
        notch widens linearly with harmonic index *k*: bins within
        ``[k * line_freq - k * half_width, k * line_freq + k * half_width]``
        are zeroed.  This accounts for the proportional increase in frequency
        uncertainty (phase jitter) at higher harmonics. Default 1.0.
    n_harmonics : int
        Number of harmonics to notch, counting from the fundamental.
        For example, ``n_harmonics=4`` zeros 60, 120, 180, and 240 Hz
        (assuming ``line_freq=60``). Default 4.

    Returns
    -------
    xr.DataArray
        Copy of *rec* with line noise removed from the specified channels.
        All coordinates and attributes are preserved unchanged.

    Raises
    ------
    TypeError
        If *rec* is not an ``xr.DataArray``.
    ValueError
        If ``ATTR_FS`` is not found in ``rec.attrs``.
    """
    if not isinstance(rec, xr.DataArray):
        raise TypeError(
            f"rec must be an xr.DataArray, got {type(rec).__name__}."
        )

    fs = rec.attrs.get(ATTR_FS)
    if fs is None:
        raise ValueError(
            f"rec.attrs['{ATTR_FS}'] not found. "
            "Load the recording with load_abf() to ensure the sample rate is set."
        )
    fs = float(fs)

    if not isinstance(channels, (list, tuple, np.ndarray)):
        channels = [channels]

    n_samples = rec.sizes[DIM_TIME]
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    nyquist = fs / 2.0

    # Build a boolean mask of frequency bins to zero.
    # The notch half-width scales linearly with harmonic index k so that
    # phase jitter (±δf at the fundamental → ±k·δf at the k-th harmonic)
    # is fully captured at every harmonic.
    zero_mask = np.zeros(len(freqs), dtype=bool)
    for k in range(1, n_harmonics + 1):
        center = k * line_freq
        if center > nyquist:
            break
        bw = k * half_width
        zero_mask |= (freqs >= center - bw) & (freqs <= center + bw)

    def _notch_trace(x):
        """Zero line-noise harmonics in a 1-D array via rfft/irfft."""
        spectrum = np.fft.rfft(x)
        spectrum[zero_mask] = 0.0
        return np.fft.irfft(spectrum, n=n_samples)

    out = rec.copy()
    for ch in channels:
        target = out.sel({DIM_CHAN: ch})
        cleaned = xr.apply_ufunc(
            _notch_trace,
            target,
            input_core_dims=[[DIM_TIME]],
            output_core_dims=[[DIM_TIME]],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[target.dtype],
        )
        out.loc[{DIM_CHAN: ch}] = cleaned

    return out
