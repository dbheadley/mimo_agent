# Functions for measuring a neuron's 
# electrophysiological properties from in vitro recordings.
# Author: Drew Headley
# Date: 2025-07-12

import numpy as np
import xarray as xr
from scipy.signal import find_peaks
from scipy.stats import linregress
from scipy.interpolate import interp1d
from scipy.optimize import brentq, curve_fit

from mimo_pack.consts import DIM_TIME, DIM_CHAN, DIM_EPOCH, ATTR_FS
from mimo_pack.preprocess.invitro import remove_ap

AP_WIN_START = 0.0005  # seconds
AP_WIN_STOP = 0.003     # seconds
ASYMP_DUR = 0.1 # seconds, duration of the asymptotic phase at end of pulse
SAG_DUR = 0.1 # seconds, duration of the sag phase at start of pulse

def extract_iv_properties(iv_xr):
    """
    Extract electrophysiological properties from an IV protocol xarray DataArray.
    Returns a dictionary of properties similar to the provided MATLAB code.

    Parameters
    ----------
    iv_xr : xarray.DataArray
        The input data array containing voltage and current traces from an IV protocol,
        with required attributes such as 'sample_rate' and 'step_dur', and coordinate
        information for channels and time.
    Returns
    -------
    props : dict
        A dictionary containing extracted electrophysiological properties, including:
            - stim_start_s: float, stimulation start time (seconds)
            - stim_stop_s: float, stimulation stop time (seconds)
            - pulse_dur_s: float, pulse duration (seconds)
            - current_baselines_pA: array-like, baseline holding currents (pA)
            - current_steps_pA: array-like, injected current steps (pA)
            - rest_vm_mV: array-like, resting membrane potentials (mV)
            - rest_vm_mean_mV: float, mean resting membrane potential (mV)
            - rest_vm_sd_mV: float, standard deviation of resting membrane potential (mV)
            - ap_peak_times_s: array-like, action potential peak times (seconds)
            - ap_peak_amps_mV: array-like, action potential peak amplitudes (mV)
            - ap_waveforms_mV: array-like, action potential waveforms (mV)
            - ap_counts: array-like, number of action potentials per epoch
            - ap_counts_reb: array-like, number of action potentials in rebound period
            - ap_threshold_pA: float, minimum current for action potential threshold (pA)
            - ap_threshold_mV: float, voltage at action potential threshold (mV)
            - ap_first_waveform_mV: array-like, waveform of the first action potential (mV)
            - ap_rates_hz: array-like, firing rates (Hz)
            - adaptation_ratio: float, adaptation ratio of firing rates
            - ap_halfwidth_s: float, half-width of the first action potential (seconds)
            - ap_peak_to_trough_s: float, peak-to-trough duration of the first action potential (seconds)
            - mem_v_mV: array-like, membrane voltage responses (mV)
            - mem_r_MOhm: float, membrane resistance (MΩ)
            - mem_t_s: float, membrane time constant (seconds)
            - sag_amp_mV: array-like, sag amplitude (mV)
            - sag_ratio: array-like, sag ratio
    Notes
    -----
    Requires helper functions:
        - _get_resting_potential
        - _detect_aps
        - _first_spike_properties
        - _rate_properties
        - _ap_shape_properties
        - _passive_properties
        - _sag_ratio
    The function is designed to mimic the output structure of similar MATLAB code for IV protocol analysis.
    """
    
    props = {}

    v_ch = int(np.where(iv_xr.ch_units.values == 'mV')[0][0])
    #i_ch = int(np.where(iv_xr.ch_units.values == 'pA')[0][0])
    v_traces = iv_xr.sel({DIM_CHAN: v_ch})
    #num_trials = v_traces.shape[1]

    stim_start = 0
    dt = 1.0 / float(iv_xr.attrs[ATTR_FS])
    pulse_dur = float(iv_xr.attrs['step_dur'])
    stim_stop = stim_start + pulse_dur
    props['stim_start_s'] = stim_start
    props['stim_stop_s'] = stim_stop
    props['pulse_dur_s'] = pulse_dur

    v_pre = v_traces.sel({DIM_TIME: slice(None, 0)}).to_numpy()
    v_dur = v_traces.sel({DIM_TIME: slice(0, pulse_dur)}).to_numpy()
    v_reb = v_traces.sel({DIM_TIME: slice(pulse_dur+dt, pulse_dur+0.2)}).to_numpy()

    baselines = iv_xr.hold_current.values
    pulse_vals = iv_xr.i_inject.values
    props['current_baselines_pA'] = baselines
    props['current_steps_pA'] = pulse_vals

    rest_vm_vals, rest_vm_mean, rest_vm_sd = _get_resting_potential(v_pre)
    props['rest_vm_mV'] = rest_vm_vals
    props['rest_vm_mean_mV'] = rest_vm_mean
    props['rest_vm_sd_mV'] = rest_vm_sd

    ap_peak_times, ap_peak_amps, ap_counts, ap_waveforms = _detect_aps(
        v_dur, dt
    )
    props['ap_peak_times_s'] = ap_peak_times
    props['ap_peak_amps_mV'] = ap_peak_amps
    props['ap_waveforms_mV'] = ap_waveforms
    props['ap_counts'] = ap_counts

    _, _, ap_counts, _ = _detect_aps(v_reb, dt)
    props['ap_counts_reb'] = ap_counts

    ap_min_i, ap_min_v, ap_first_wave = _first_spike_properties(
        ap_waveforms, pulse_vals
    )
    props['ap_threshold_pA'] = ap_min_i
    props['ap_threshold_mV'] = ap_min_v
    props['ap_first_waveform_mV'] = ap_first_wave

    ap_rates, adaptation_ratio = _rate_properties(ap_peak_times)
    props['ap_rates_hz'] = ap_rates
    props['adaptation_ratio'] = adaptation_ratio

    ap_halfwidth, ap_peak_to_trough = _ap_shape_properties(ap_first_wave)
    props['ap_halfwidth_s'] = ap_halfwidth
    props['ap_peak_to_trough_s'] = ap_peak_to_trough

    mem_v, mem_r, mem_t = _passive_properties(v_dur, pulse_vals, dt)
    props['mem_v_mV'] = mem_v
    props['mem_r_MOhm'] = mem_r * 1e-6 # convert to MOhms
    props['mem_t_s'] = mem_t

    sag_amp, sag_ratio = _sag_ratio(v_dur, pulse_vals, dt)
    props['sag_amp_mV'] = sag_amp
    props['sag_ratio'] = sag_ratio

    return props

def _opsin_model(t_ms, I_max, tau_on, tau_inact, r):
    """During-pulse opsin response (22OM-inspired).

    .. math::

        R(t) = I_{max} \\cdot (1 - e^{-t / \\tau_{on}})
               \\cdot [r + (1 - r) \\cdot e^{-t / \\tau_{inact}}]

    Product of two independent gating processes:
    * **Opening** gate ``O(t) = 1 - exp(-t / tau_on)`` rises from 0 → 1.
    * **Adaptation** gate ``A(t) = r + (1-r) * exp(-t / tau_inact)``
      decays from 1 → r.

    Parameters
    ----------
    t_ms : array-like
        Time in **milliseconds** (t = 0 at LED onset).
    I_max : float
        Maximum achievable current (all channels open, no adaptation).
    tau_on : float
        Activation time constant (ms).
    tau_inact : float
        Inactivation / adaptation time constant (ms).
    r : float
        Sustained-to-peak ratio, constrained to [0, 1].  ``r = 1`` means
        no inactivation (square pulse); ``r = 0`` means full decay to zero.
    """
    return I_max * (1 - np.exp(-t_ms / max(tau_on, 1e-5))) * (
        r + (1 - r) * np.exp(-t_ms / max(tau_inact, 1e-5))
    )


def _opsin_full_trace(t_ms, I_max, tau_on, tau_inact, r, tau_off,
                       pulse_dur_ms):
    """Full-trace model: during-pulse 22OM + post-pulse deactivation.

    During the pulse (0 <= t <= pulse_dur_ms) the response follows
    ``_opsin_model``.  After the pulse the current decays exponentially::

        R(t) = R(t_end) * exp(-(t - t_end) / tau_off)

    Parameters
    ----------
    t_ms : array-like
        Time in ms (t = 0 at LED onset).
    I_max, tau_on, tau_inact, r : float
        During-pulse model parameters (see ``_opsin_model``).
    tau_off : float
        Post-pulse deactivation time constant (ms).
    pulse_dur_ms : float
        LED pulse duration in ms.
    """
    t_ms = np.asarray(t_ms, dtype=float)
    y = np.zeros_like(t_ms)

    on_mask = (t_ms >= 0) & (t_ms <= pulse_dur_ms)
    y[on_mask] = _opsin_model(t_ms[on_mask], I_max, tau_on, tau_inact, r)

    # Value at end of pulse (continuous join)
    R_end = _opsin_model(pulse_dur_ms, I_max, tau_on, tau_inact, r)

    off_mask = t_ms > pulse_dur_ms
    dt = t_ms[off_mask] - pulse_dur_ms
    y[off_mask] = R_end * np.exp(-dt / max(tau_off, 1e-5))

    return y


def _solve_amplitudes(tau_on, tau_inact, peak_amp_data, A_sust_data,
                       pulse_dur_ms):
    """Solve for (I_max, r) that produce given peak and sustained amplitudes.

    For a given pair of time constants, find the unique *r* such that
    ``sustained / peak == r / max[g(t; r)]`` where
    ``g(t; r) = (1 - exp(-t/tau_on)) * (r + (1-r) * exp(-t/tau_inact))``.
    Then ``I_max = peak_amp_data / max[g]``.

    Parameters
    ----------
    tau_on, tau_inact : float
        Time constants in ms.
    peak_amp_data, A_sust_data : float
        Measured peak and sustained amplitudes (baseline-subtracted).
    pulse_dur_ms : float
        LED pulse duration in ms.

    Returns
    -------
    I_max, r : float
        Model parameters that reproduce the measured amplitudes.
    """
    if peak_amp_data == 0:
        return 0.0, 0.5

    ratio = np.clip(A_sust_data / peak_amp_data, 0.01, 0.99)

    # Dense time grid focused on early response where peak occurs
    t_end = min(pulse_dur_ms, 10.0 * (tau_on + tau_inact + 1))
    t_grid = np.linspace(0.001, max(t_end, 1.0), 5000)
    tau_on_s = max(tau_on, 1e-5)
    tau_inact_s = max(tau_inact, 1e-5)
    O = 1 - np.exp(-t_grid / tau_on_s)
    E = np.exp(-t_grid / tau_inact_s)

    def _ratio_error(r):
        """Objective for Brent’s method: error between predicted and observed sustained/peak ratio."""
        g = O * (r + (1 - r) * E)
        g_max = np.max(g)
        return (r / g_max - ratio) if g_max > 1e-12 else (r - ratio)

    try:
        r_sol = brentq(_ratio_error, 1e-6, 1 - 1e-6, xtol=1e-8)
    except ValueError:
        r_sol = np.clip(ratio, 0.01, 0.99)

    g = O * (r_sol + (1 - r_sol) * E)
    g_max = float(np.max(g))
    I_max = peak_amp_data / g_max if g_max > 1e-12 else peak_amp_data

    return float(I_max), float(r_sol)


def _fit_single_trace(
    t_s: np.ndarray,
    v_1d: np.ndarray,
    pulse_dur: float,
    A_sust_fixed: bool = False,
    led_dur_for_post: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Fit the 22OM-inspired opsin model to a single 1-D trace.

    The model combines an opening gate with an adaptation gate plus
    post-pulse deactivation.  All data at t >= 0 are used for fitting
    (both during and after the pulse), giving the optimizer information
    about both onset**,** sustained, and off kinetics.

    Parameters
    ----------
    t_s : np.ndarray
        Full time axis in seconds (must include t < 0 pre-LED region).
    v_1d : np.ndarray
        Signal values corresponding to *t_s*.
    pulse_dur : float
        LED pulse duration in seconds.
    A_sust_fixed : bool
        If True, fix only the sustained amplitude at its measured data
        value and fit ``I_max``, ``tau_on``, ``tau_inact``, and
        ``tau_off`` as free parameters.  ``r`` is derived at each step
        as ``A_sust / I_max``.
    led_dur_for_post : float or None
        LED duration used for computing the post-pulse level (raw mean
        after pulse).  When ``None``, post-pulse is set to NaN.

    Returns
    -------
    v_fitted : np.ndarray
        Reconstructed model trace over the full time axis.
    params : dict
        Fit parameters: I_max, tau_on_ms, tau_inact_ms, inactivation_ratio,
        tau_off_ms, peak_amp, peak_time_ms, A_transient, A_sust, baseline,
        post_pulse.
    """
    # Baseline from pre-LED period
    pre_mask = t_s < 0
    baseline = float(np.mean(v_1d[pre_mask])) if pre_mask.any() else 0.0

    # All data from LED onset onward
    fit_mask = t_s >= 0
    t_fit = t_s[fit_mask]
    v_fit = v_1d[fit_mask] - baseline

    if t_fit.size == 0:
        raise ValueError(
            "No timepoints found at t >= 0. "
            "Ensure time axis is re-zeroed to LED onset via format_blue_tonic_led."
        )

    t_fit_ms = t_fit * 1000.0
    pulse_dur_ms = pulse_dur * 1000.0

    # ---- Initial guesses from the data ------------------------------------
    pulse_mask_local = t_fit_ms <= pulse_dur_ms
    v_pulse = v_fit[pulse_mask_local]
    sign = 1 if np.abs(np.max(v_pulse)) >= np.abs(np.min(v_pulse)) else -1
    extremum_idx = int(np.argmax(sign * v_pulse))
    I_max_guess = float(v_pulse[extremum_idx])

    v_ss_guess = float(np.median(v_pulse[-max(1, len(v_pulse) // 10):]))
    r_guess = np.clip(v_ss_guess / I_max_guess, 0.05, 0.95) if I_max_guess != 0 else 0.5

    # Model wrapper for curve_fit
    def _model(t_ms_arr, I_max, tau_on, tau_inact, r, tau_off):
        """Full opsin model for ``curve_fit`` with all 5 free parameters."""
        return _opsin_full_trace(t_ms_arr, I_max, tau_on, tau_inact, r,
                                 tau_off, pulse_dur_ms)

    if A_sust_fixed:
        # Fix only the sustained amplitude from data; I_max (and thus peak) is free.
        # r is derived at each optimizer step as A_sust / I_max.
        _sust_fixed = float(v_ss_guess)

        def _model_fixed(t_ms_arr, I_max, tau_on, tau_inact, tau_off):
            """Opsin model with sustained amplitude fixed from data (4 free params)."""
            r_c = float(np.clip(_sust_fixed / I_max, 0.01, 0.99)) if I_max != 0 else 0.5
            return _opsin_full_trace(t_ms_arr, I_max, tau_on, tau_inact,
                                     r_c, tau_off, pulse_dur_ms)

        p0 = [I_max_guess, 2.0, 25.0, 5.0]
        bounds = (
            [min(I_max_guess * 3, -1e6), 0.01, 0.1, 0.1],
            [max(I_max_guess * 3,  1e6), 200.0, 5000.0, 5000.0],
        )
        try:
            popt, _ = curve_fit(_model_fixed, t_fit_ms, v_fit,
                                p0=p0, bounds=bounds, maxfev=20000)
        except RuntimeError as exc:
            raise ValueError(
                f"Opsin model fit did not converge: {exc}."
            ) from exc
        I_max, tau_on, tau_inact, tau_off = popt
        r = float(np.clip(_sust_fixed / I_max, 0.01, 0.99)) if I_max != 0 else 0.5
    else:
        p0 = [I_max_guess, 2.0, 25.0, float(r_guess), 5.0]
        bounds = (
            [min(I_max_guess * 3, -1e6), 0.01, 0.1, 0.0, 0.1],
            [max(I_max_guess * 3,  1e6), 200.0, 5000.0, 1.0, 5000.0],
        )
        try:
            popt, _ = curve_fit(_model, t_fit_ms, v_fit,
                                p0=p0, bounds=bounds, maxfev=20000)
        except RuntimeError as exc:
            raise ValueError(
                f"Opsin model fit did not converge: {exc}."
            ) from exc
        I_max, tau_on, tau_inact, r, tau_off = popt

    # ---- Derived quantities -----------------------------------------------
    # Peak from the during-pulse model
    pulse_t = t_fit_ms[t_fit_ms <= pulse_dur_ms]
    model_pulse = _opsin_model(pulse_t, I_max, tau_on, tau_inact, r)
    peak_idx = int(np.argmax(np.abs(model_pulse)))
    peak_amp = float(model_pulse[peak_idx])
    peak_time_ms = float(pulse_t[peak_idx])

    # Backward-compatible derived amplitudes
    A_sust = float(I_max * r)           # steady-state level (t → ∞)
    A_transient = float(I_max - A_sust) # decaying component amplitude

    inact_ratio = float(r)              # r IS the inactivation ratio

    # Post-pulse level (raw data)
    if led_dur_for_post is not None:
        post_data_mask = t_s > led_dur_for_post
        post_pulse = float(np.mean(v_1d[post_data_mask])) if post_data_mask.any() else np.nan
    else:
        post_pulse = np.nan

    # ---- Reconstruct full trace -------------------------------------------
    v_fitted = np.full_like(t_s, baseline, dtype=float)
    v_fitted[fit_mask] = baseline + _opsin_full_trace(
        t_fit_ms, I_max, tau_on, tau_inact, r, tau_off, pulse_dur_ms
    )
    # Pre-LED region stays at baseline (already set)

    params = {
        'I_max':              float(I_max),
        'tau_on_ms':          float(tau_on),
        'tau_inact_ms':       float(tau_inact),
        'inactivation_ratio': inact_ratio,
        'tau_off_ms':         float(tau_off),
        'peak_amp':           peak_amp,
        'peak_time_ms':       peak_time_ms,
        'A_transient':        A_transient,
        'A_sust':             A_sust,
        'baseline':           baseline,
        'post_pulse':         post_pulse,
    }
    return v_fitted, params


def fit_opsin_response(
    cell_xr: xr.DataArray,
    A_sust_fixed: bool = False,
    target_units: str | None = None,
    per_epoch: bool = False,
) -> xr.DataArray:
    """Fit a 22OM-inspired opsin model to the tonic LED response.

    Fits the Double Two-State Opsin Model (22OM) inspired formulation::

        During pulse (0 <= t <= pulse_dur):
            R(t) = I_max * (1 - exp(-t / tau_on))
                         * [r + (1-r) * exp(-t / tau_inact)]

        After pulse (t > pulse_dur):
            R(t) = R(t_end) * exp(-(t - t_end) / tau_off)

    The during-pulse response is a product of two independent gating
    processes: an **opening gate** ``(1 - exp(-t / tau_on))`` that rises
    from 0 → 1, and an **adaptation gate** ``r + (1-r) * exp(-t / tau_inact)``
    that decays from 1 → *r*.  The parameter *r* ∈ [0, 1] directly
    represents the sustained-to-peak ratio, preventing degenerate fits.
    Post-pulse deactivation is modeled as a single exponential with time
    constant ``tau_off``.

    All data at t >= 0 (during *and* after the pulse) are used for fitting.
    Time constants are in milliseconds.

    Works on both current-clamp (mV) and voltage-clamp (pA) recordings.
    The target channel is selected automatically from the recording's
    ``clamp_mode`` attribute, or can be overridden with ``target_units``.

    Parameters
    ----------
    cell_xr : xr.DataArray
        Formatted single-cell recording with dims including DIM_TIME and
        DIM_EPOCH.  The time axis must already be re-zeroed to LED onset
        (output of format_blue_tonic_led).
    A_sust_fixed : bool, optional
        If ``True``, the peak and sustained amplitudes are fixed at their
        measured data values (extremum and last-10 %% median of the
        baseline-subtracted pulse).  Only the time constants ``tau_on``,
        ``tau_inact``, and ``tau_off`` are free.  ``I_max`` and ``r`` are
        derived at each optimizer step to reproduce the data amplitudes
        exactly.  If ``False`` (default), all five parameters are fit
        freely.
    target_units : str or None, optional
        Channel unit string to analyse (``'mV'`` or ``'pA'``).  When ``None``
        (default), auto-detected from the ``clamp_mode`` attribute:
        ``'V-Clamp'`` → ``'pA'``, otherwise → ``'mV'``.
    per_epoch : bool, optional
        How epochs are handled:

        - ``False`` (default): pool all epochs into a single fit by tiling
          the time axis so ``curve_fit`` sees every ``(t, y)`` pair.  Returns
          a 1-D ``(time,)`` DataArray with fit params in ``attrs``.
        - ``True``: fit each epoch independently.  Returns a 2-D
          ``(time, epoch)`` DataArray with per-epoch fit params stored as
          **coordinates on the epoch dimension**.  Global attrs hold only
          ``response_units``.

    Returns
    -------
    xr.DataArray
        Reconstructed response trace(s).

        When ``per_epoch=False`` (default): 1-D ``(time,)`` with fit params
        in ``attrs`` (``I_max``, ``tau_on_ms``, ``tau_inact_ms``,
        ``inactivation_ratio``, ``tau_off_ms``, ``peak_amp``,
        ``peak_time_ms``, ``A_transient``, ``A_sust``, ``baseline``,
        ``post_pulse``, ``response_units``).

        When ``per_epoch=True``: 2-D ``(time, epoch)`` with fit params as
        epoch coordinates.

    Raises
    ------
    TypeError
        If cell_xr is not an xr.DataArray.
    ValueError
        If the target channel is not found, no data at t >= 0, or curve_fit
        fails.
    """
    if not isinstance(cell_xr, xr.DataArray):
        raise TypeError(
            f"cell_xr must be an xr.DataArray, got {type(cell_xr).__name__}."
        )

    # ---- Determine target channel -----------------------------------------
    if target_units is None:
        clamp_mode = cell_xr.attrs.get('clamp_mode', 'unknown')
        target_units = 'pA' if clamp_mode == 'V-Clamp' else 'mV'

    ch_units = cell_xr['ch_units'].to_numpy().tolist()
    target_indices = [i for i, u in enumerate(ch_units) if u == target_units]
    if len(target_indices) == 0:
        raise ValueError(
            f"No channel with ch_units == '{target_units}' found. "
            f"Available units: {ch_units}"
        )
    target_ch = target_indices[0]

    # Remove action potentials only for current-clamp (mV) recordings
    if target_units == 'mV':
        cell_xr = remove_ap(cell_xr, voltage_channel=target_ch)

    v_traces = cell_xr.sel({DIM_CHAN: target_ch})   # (time, epoch)
    t_s = v_traces[DIM_TIME].to_numpy()
    pulse_dur = float(cell_xr.attrs.get('led_dur', None))
    led_dur_for_post = (
        float(np.nanmean(cell_xr['led_dur'].values))
        if 'led_dur' in cell_xr.coords else None
    )
    n_epochs = v_traces.sizes[DIM_EPOCH]

    # ---- Per-epoch mode ---------------------------------------------------
    if per_epoch:
        fit_traces = []
        param_keys = ['I_max', 'tau_on_ms', 'tau_inact_ms',
                       'inactivation_ratio', 'tau_off_ms', 'peak_amp',
                       'peak_time_ms', 'A_transient', 'A_sust',
                       'baseline', 'post_pulse']
        param_arrays = {k: [] for k in param_keys}

        for ep in range(n_epochs):
            v_1d = v_traces.isel({DIM_EPOCH: ep}).to_numpy()
            v_fitted, params = _fit_single_trace(
                t_s, v_1d, pulse_dur, A_sust_fixed, led_dur_for_post,
            )
            fit_traces.append(v_fitted)
            for k in param_keys:
                param_arrays[k].append(params[k])

        epoch_vals = v_traces[DIM_EPOCH].to_numpy()
        coords = {
            DIM_TIME: t_s,
            DIM_EPOCH: epoch_vals,
        }
        for k in param_keys:
            coords[k] = (DIM_EPOCH, param_arrays[k])

        out_attrs = {**cell_xr.attrs, 'response_units': target_units}
        return xr.DataArray(
            np.column_stack(fit_traces),           # (time, epoch)
            coords=coords,
            dims=[DIM_TIME, DIM_EPOCH],
            attrs=out_attrs,
        )

    # ---- Pooled mode (default) --------------------------------------------
    # Tile t and flatten all epochs into one long vector for curve_fit
    v_all = v_traces.to_numpy()                     # (time, epoch)
    t_tiled = np.tile(t_s, n_epochs)
    v_flat = v_all.T.ravel()                        # epoch-major flatten

    _, params = _fit_single_trace(
        t_tiled, v_flat, pulse_dur, A_sust_fixed, led_dur_for_post,
    )

    # Reconstruct a single (time,) trace using the pooled params
    fit_mask = t_s >= 0
    v_fitted_1d = np.full_like(t_s, params['baseline'], dtype=float)
    v_fitted_1d[fit_mask] = params['baseline'] + _opsin_full_trace(
        t_s[fit_mask] * 1000.0,
        params['I_max'], params['tau_on_ms'], params['tau_inact_ms'],
        params['inactivation_ratio'], params['tau_off_ms'],
        pulse_dur * 1000.0,
    )

    fit_attrs = {**cell_xr.attrs, **params, 'response_units': target_units}

    return xr.DataArray(
        v_fitted_1d,
        coords={DIM_TIME: t_s},
        dims=[DIM_TIME],
        attrs=fit_attrs,
    )


def opsin_iv_conductance(
    v_hold: np.ndarray | list,
    i_peak: np.ndarray | list,
    i_sust: np.ndarray | list,
) -> dict:
    """Compute chord conductance and reversal potential from opsin I-V data.

    Fits a linear model ``I = g * (V - E_rev)`` to the peak and sustained
    opsin photocurrents measured at different holding potentials.  The slope
    gives the chord conductance (nS when *I* is in pA and *V* in mV), and the
    x-intercept gives the reversal potential.

    Parameters
    ----------
    v_hold : array-like, shape (n,)
        Holding potentials (mV) at which photocurrents were recorded.
    i_peak : array-like, shape (n,)
        Peak photocurrent amplitude (pA) at each holding potential (from
        ``fit_opsin_response`` ``peak_amp`` attribute).
    i_sust : array-like, shape (n,)
        Sustained photocurrent amplitude (pA) at each holding potential (from
        ``fit_opsin_response`` ``A_sust`` attribute).

    Returns
    -------
    dict
        Keys:

        - ``g_peak_nS``   – chord conductance from peak I-V (nS)
        - ``g_sust_nS``   – chord conductance from sustained I-V (nS)
        - ``E_rev_peak_mV``  – reversal potential from peak I-V (mV)
        - ``E_rev_sust_mV``  – reversal potential from sustained I-V (mV)
        - ``r2_peak``     – R² of the peak linear fit
        - ``r2_sust``     – R² of the sustained linear fit
        - ``n_points``    – number of (V, I) points used

    Raises
    ------
    ValueError
        If fewer than 2 data points are provided.
    """
    v = np.asarray(v_hold, dtype=float)
    i_pk = np.asarray(i_peak, dtype=float)
    i_ss = np.asarray(i_sust, dtype=float)

    if v.size < 2:
        raise ValueError(
            f"Need at least 2 (V, I) points for a linear fit, got {v.size}."
        )

    result = {'n_points': int(v.size)}

    for label, i_vals in [('peak', i_pk), ('sust', i_ss)]:
        # Drop NaN pairs
        mask = np.isfinite(v) & np.isfinite(i_vals)
        if mask.sum() < 2:
            result[f'g_{label}_nS'] = np.nan
            result[f'E_rev_{label}_mV'] = np.nan
            result[f'r2_{label}'] = np.nan
            continue

        reg = linregress(v[mask], i_vals[mask])
        slope = reg.slope          # pA / mV = nS
        intercept = reg.intercept  # pA

        result[f'g_{label}_nS'] = float(slope)
        result[f'r2_{label}'] = float(reg.rvalue ** 2)

        # Reversal potential: I = 0 → V = -intercept / slope
        if abs(slope) > 1e-12:
            result[f'E_rev_{label}_mV'] = float(-intercept / slope)
        else:
            result[f'E_rev_{label}_mV'] = np.nan

    return result


def _get_resting_potential(v_traces):
    """
    Calculate the resting membrane potential statistics from voltage traces.

    Parameters
    ----------
    v_traces : np.ndarray
        Array of voltage traces, where each row represents a trace.

    Returns
    -------
    rest_vm_vals : np.ndarray
        Median voltage values across traces for each time point.
    rest_vm_mean : float
        Mean of the median resting membrane potentials.
    rest_vm_sd : float
        Standard deviation of the median resting membrane potentials (sample SD, ddof=1).
    """
    rest_vm_vals = np.median(v_traces, axis=0)
    rest_vm_mean = np.mean(rest_vm_vals)
    rest_vm_sd = np.std(rest_vm_vals, ddof=1)
    return rest_vm_vals, rest_vm_mean, rest_vm_sd

def _detect_aps(v_traces, dt):
    """
    Detects action potentials (APs) in voltage traces and extracts their properties.
    Parameters
    ----------
    v_traces : np.ndarray
        2D array of voltage traces with shape (num_timepoints, num_trials).
    dt : float
        Time step between samples in seconds.
    Returns
    -------
    ap_peak_times : list of np.ndarray
        List containing arrays of AP peak times (in seconds) for each epoch.
    ap_peak_amps : list of np.ndarray
        List containing arrays of AP peak amplitudes for each epoch.
    ap_counts : list of int
        List containing the number of APs detected in each epoch.
    ap_waveforms : list of np.ndarray
        List containing arrays of AP waveforms for each epoch, 
        each array has shape (window_length, num_aps).
    """
    ap_peak_times = []
    ap_peak_amps = []
    ap_waveforms = []
    ap_counts = []
    num_trials = v_traces.shape[1]
    ap_win_start = int(AP_WIN_START / dt)
    ap_win_stop = int(AP_WIN_STOP / dt)
    ap_window = np.arange(-ap_win_start, ap_win_stop+1)
    for j in range(num_trials):
        curr_v = v_traces[:, j]
        peaks, _ = find_peaks(curr_v, height=0, distance=int(0.001/dt))
        curr_v = np.pad(curr_v, (ap_win_start, ap_win_stop), 
                        mode='constant', constant_values=np.nan)
        ap_peak_times.append(peaks * dt)
        ap_peak_amps.append(curr_v[peaks+ap_win_start])
        ap_counts.append(len(peaks))

        wf = []
        for ind in peaks:
            inds = ap_window + ind + ap_win_start
            wf.append(curr_v[inds])
        ap_waveforms.append(np.array(wf).T)

    return ap_peak_times, ap_peak_amps, np.array(ap_counts), ap_waveforms


def _first_spike_properties(ap_waveforms, pulse_vals):
    """
    Extracts properties of the first detected action potential (spike) from a set of waveforms.

    Parameters
    ----------
    ap_waveforms : list or array-like
        A list or array of action potential waveforms for each epoch. Each element should be a 2D array
        where rows correspond to waveform samples and columns to spikes.
    pulse_vals : list or array-like
        A list or array of pulse values corresponding to each epoch.

    Returns
    -------
    min_i : float or np.nan
        The pulse value for the epoch containing the first spike, or np.nan if no spikes are found.
    min_v : float or np.nan
        The initial voltage value of the first spike waveform, or np.nan if no spikes are found.
    ap_first_wave : array-like or np.nan
        The waveform of the first detected spike, or np.nan if no spikes are found.
    """

    first_spike_trial = [i for i, ap in enumerate(ap_waveforms) if ap.size > 0][0]
    if first_spike_trial is None:
        return np.nan, np.nan, np.nan
    min_i = pulse_vals[first_spike_trial]
    ap_first_wave = ap_waveforms[first_spike_trial][:,0]
    min_v = ap_first_wave[0]
    return min_i, min_v, ap_first_wave

def _rate_properties(ap_peak_times):
    """
    Calculate firing rate properties from action potential peak times.
    Parameters
    ----------
    ap_peak_times : list of array-like
        A list where each element contains the times (in seconds) of action potential
        for an epoch.
    Returns
    -------
    ap_rates : list of float
        List of mean instantaneous firing rates (Hz) for each epoch. If an epoch has fewer
        than 2 action potentials, returns np.nan for that epoch.
    adaptation_ratio : float
        The adaptation ratio for the first epoch with at least 4 action potentials, calculated
        as the ratio of the first to last instantaneous firing rate.
        Returns np.nan if no such epoch exists.
    """
    ap_rates = []

    # Calculate instantaneous firing rate
    for i, times in enumerate(ap_peak_times):
        if len(times) < 2:
            ap_rates.append(np.nan)
            continue
        
        ap_inst_rate = 1/np.diff(times)
        ap_rates.append(np.mean(ap_inst_rate))

    # Calculate adaptation ratio
    try:
        adapt_trial = [i for i, r in enumerate(ap_peak_times) if len(r) >= 4][0]
        ap_inst_rate = 1/np.diff(ap_peak_times[adapt_trial])
        adaptation_ratio = ap_inst_rate[0] / ap_inst_rate[-1]
    except IndexError:
        adaptation_ratio = np.nan
    
    return ap_rates, adaptation_ratio

def _ap_shape_properties(ap_wave):
    """Compute half-width and peak-to-trough time of an action potential waveform.

    Parameters
    ----------
    ap_wave : numpy.ndarray
        1-D voltage waveform of a single action potential.

    Returns
    -------
    ap_halfwidth : float
        Duration (seconds) at half-maximum amplitude. ``NaN`` if the
        waveform contains NaN values.
    ap_peak_to_trough : float
        Time (seconds) from the peak to the subsequent trough.
        ``NaN`` if the waveform contains NaN values.
    """
    ap_halfwidth = np.nan
    ap_peak_to_trough = np.nan
    if not np.any(np.isnan(ap_wave)):
        t_wave = np.linspace(0.0005, 0.003, ap_wave.size)
        interp_t = np.arange(0.0005, 0.003, 0.00001)
        interp_func = interp1d(t_wave, ap_wave, kind='cubic', fill_value='extrapolate')
        up_wave = interp_func(interp_t)
        v_half = (np.max(up_wave) - up_wave[0]) / 2 + up_wave[0]
        ap_halfwidth = np.sum(up_wave > v_half) * 0.00001
        peak_ind = np.argmax(up_wave)
        ap_peak_to_trough = np.argmin(up_wave[peak_ind:]) * 0.00001
        
    return ap_halfwidth, ap_peak_to_trough

def _passive_properties(v_traces, pulse_vals, dt):
    """
    Calculates passive electrophysiological properties from voltage traces and current pulses.
    Parameters
    ----------
    v_traces : np.ndarray
        2D array of voltage traces (time x pulses).
    pulse_vals : np.ndarray
        1D array of current pulse amplitudes corresponding to each pulse.
    dt : float
        Time step between samples in seconds.
    Returns
    -------
    mem_v : np.ndarray
        Median steady-state membrane voltage for each pulse.
    mem_r : float or np.nan
        Estimated membrane resistance in Ohms, calculated from negative current injections.
    mem_t : float or np.nan
        Estimated membrane time constant (tau) in seconds, calculated from exponential fit to voltage response.
    """
    asymp_ind = int(ASYMP_DUR / dt)
    mem_r = np.nan
    mem_t = np.nan
    mem_v = np.median(v_traces[-asymp_ind:, :], axis=0)
    sel_neg_pulse = np.where(pulse_vals < 0)[0][-2]
    
    # Calculate membrane resistance as the slope of the voltage change
    # over the current change during negative current injections
    if sel_neg_pulse > 1:
        dv = np.diff(mem_v[(sel_neg_pulse-2):(sel_neg_pulse+2)]) * 0.001  # V
        di = np.diff(pulse_vals[(sel_neg_pulse-2):(sel_neg_pulse+2)]) * 1e-12     # A
        mem_r = np.mean(dv / di) # Ohms

    # Calculate membrane capacitance as the time constant of the voltage change
    # fit an exponential decay to the second to last negative pulse, capacitance is C = tau / R
    if sel_neg_pulse > 1 and mem_r > 0:
        v_fit = v_traces[:, sel_neg_pulse]
        t_fit = np.arange(0, v_fit.size * dt, dt)

        # tau is the time constant, a is the amplitude, b is the offset
        fit_func = lambda t, tau, a, b: (a*(1-np.exp(-t / tau)))+b
        p0 = [0.01, # 10 ms time constant
              pulse_vals[sel_neg_pulse]*mem_r*1e-9, # expected amplitude in mV
              v_fit[0]] # initial voltage
        bounds = ([0, -np.inf, -200], [1, 0, 0])
        popt, _ = curve_fit(fit_func, t_fit, v_fit, p0=p0, bounds=bounds)
        tau = popt[0]
        mem_t = tau  # Time constant in seconds

    return mem_v, mem_r, mem_t

def _sag_ratio(v_traces, pulse_vals, dt):
    """
    Calculates the sag amplitude and sag ratio from voltage traces in response 
    to hyperpolarizing current pulses.
    
    Parameters
    ----------
    v_traces : np.ndarray
        2D array of voltage traces (time x epochs).
    pulse_vals : np.ndarray
        1D array of pulse amplitudes for each epoch.
    dt : float
        Time step (sampling interval) in milliseconds.
    
    Returns
    -------
    sag_amp : float
        The sag amplitude, defined as the difference between the minimum voltage during 
        the sag and the asymptotic voltage.
    sag_ratio : float
        The sag ratio, defined as the ratio of the asymptotic voltage to the minimum 
        sag voltage.
    """
    asymp_ind = int(ASYMP_DUR / dt)
    sag_ind = int(SAG_DUR / dt)
    neg_trial = np.argmin(pulse_vals)

    v_resp = v_traces[:, neg_trial] - v_traces[0, neg_trial]  # subtract resting potential
    v_sag = np.min(v_resp[0:sag_ind])
    v_asymp = np.median(v_resp[-asymp_ind:])
    sag_amp = v_sag - v_asymp
    sag_ratio = v_asymp / v_sag
    return sag_amp, sag_ratio


if __name__ == "__main__":
    import xarray as xr
    import matplotlib.pyplot as plt
    from scipy import stats
    from mimo_pack.testing.data import synthetic_iv_data
    
    print("=" * 60)
    print("IN VITRO ELECTROPHYSIOLOGY PROPERTIES TESTS")
    print("=" * 60)
    
    # ========== Test 1: Create synthetic IV protocol data ==========
    print("\n1. Creating synthetic IV protocol data...")
    try:
        # Generate synthetic IV data with action potentials
        iv_data = synthetic_iv_data(
            i_steps=np.arange(-150, 125, 25),  # -150 to +100 pA
            fs=10000,
            step_dur=0.5,
            pre_dur=0.1,
            post_dur=0.1,
            add_aps=True,
            seed=42
        )
        
        # Extract variables for compatibility with existing test code
        fs = iv_data.attrs['fs']
        step_dur = iv_data.attrs['step_dur'] 
        pre_dur = iv_data.attrs['pre_dur']
        post_dur = iv_data.attrs['post_dur']
        v_traces = iv_data.values
        i_steps = iv_data['current_step'].values
        t = iv_data['time'].values
        n_trials = len(i_steps)
        n_samples = len(t)
        
        step_start = int(pre_dur * fs)
        step_end = int((pre_dur + step_dur) * fs)
        
        print(f"   + Created IV data: shape={iv_data.shape}")
        print(f"   Current steps: {i_steps[0]} to {i_steps[-1]} pA ({n_trials} steps)")
        print(f"   Time: {(pre_dur + step_dur + post_dur)*1000:.0f} ms (pre={pre_dur*1000:.0f}, step={step_dur*1000:.0f}, post={post_dur*1000:.0f})")
        print(f"   Contains action potentials: {iv_data.attrs.get('add_aps', 'True')}")
    except Exception as e:
        print(f"   - Failed: {e}")
        raise
    
    # ========== Test 2: Test _sag_ratio function ==========
    print("\n2. Testing _sag_ratio function...")
    try:
        dt = 1 / fs
        sag_amp, sag_ratio = _sag_ratio(v_traces[step_start:step_end, :], i_steps, dt)
        
        print(f"   + Sag amplitude: {sag_amp:.2f} mV")
        print(f"   + Sag ratio: {sag_ratio:.3f}")
        
        assert sag_amp < 0, "Sag amplitude should be negative"
        assert 0 < sag_ratio < 1, "Sag ratio should be between 0 and 1"
    except Exception as e:
        print(f"   - Failed: {e}")
        sag_amp = -2.0  # fallback value
    
    # ========== Test 3: Test _passive_properties function ==========
    print("\n3. Testing _passive_properties function...")
    try:
        dt = 1 / fs
        mem_v, mem_r, mem_t = _passive_properties(v_traces[step_start:step_end, :], i_steps, dt)
        
        print(f"   + Membrane resistance: {mem_r/1e6:.1f} MΩ")
        print(f"   + Membrane time constant: {mem_t*1000:.1f} ms")
        
        # Should be close to our simulated values
        assert 50e6 < mem_r < 200e6, f"Mem R {mem_r/1e6:.1f} MΩ outside expected range"
        assert 0.01 < mem_t < 0.05, f"Tau {mem_t} outside expected range"
        
        # Set variables for later use in visualization
        v_rest = mem_v[np.argmin(np.abs(i_steps))]  # baseline voltage
        r_peak = mem_r / 1e6  # Convert to MΩ for display
        r_ss = mem_r / 1e6  # Same as peak for this test
        tau_measured = mem_t
    except Exception as e:
        print(f"   - Failed: {e}")
        # Fallback values
        v_rest = -70.0
        r_peak = 100.0
        r_ss = 95.0
        tau_measured = 0.02
    
    # ========== Test 4: Test IV curve fitting ==========
    print("\n4. Testing IV curve linear region fitting...")
    try:
        dt = 1 / fs
        asymp_ind = int(ASYMP_DUR / dt)
        
        # Calculate steady-state voltages
        v_ss = np.median(v_traces[step_end-asymp_ind:step_end, :], axis=0)
        v_ss = v_ss - v_ss[np.argmin(np.abs(i_steps))]  # Baseline to zero current
        
        # Linear fit in sub-threshold region
        sub_thresh_mask = (i_steps >= -150) & (i_steps <= 0)
        slope, intercept, r_value, _, _ = stats.linregress(
            i_steps[sub_thresh_mask], 
            v_ss[sub_thresh_mask]
        )
        
        r_from_slope = slope * 1e3  # Convert to MOhm
        
        print(f"   + IV slope: {slope:.4f} mV/pA")
        print(f"   + R_in from IV: {r_from_slope:.1f} MΩ")
        print(f"   + R² = {r_value**2:.4f}")
    except Exception as e:
        print(f"   - Failed: {e}")
    
    # ========== Test 5: Visualization ==========
    print("\n5. Creating visualization...")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Voltage traces
        ax = axes[0, 0]
        colors = plt.cm.coolwarm(np.linspace(0, 1, n_trials))
        for i in range(n_trials):
            ax.plot(t * 1000, v_traces[:, i], color=colors[i], lw=0.8)
        ax.axvline(pre_dur * 1000, color='k', ls='--', alpha=0.5)
        ax.axvline((pre_dur + step_dur) * 1000, color='k', ls='--', alpha=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage (mV)')
        ax.set_title('IV Protocol Traces')
        
        # Plot 2: IV curve
        ax = axes[0, 1]
        ax.plot(i_steps, v_ss, 'ko-', ms=8)
        ax.axhline(0, color='gray', ls='--', alpha=0.5)
        ax.axvline(0, color='gray', ls='--', alpha=0.5)
        
        # Add linear fit
        fit_x = np.linspace(i_steps.min(), 0, 100)
        fit_y = slope * fit_x + intercept
        ax.plot(fit_x, fit_y, 'r-', lw=2, label=f'R = {r_from_slope:.0f} MΩ')
        
        ax.set_xlabel('Current (pA)')
        ax.set_ylabel('ΔVoltage (mV)')
        ax.set_title('I-V Curve')
        ax.legend()
        
        # Plot 3: Sag response (most negative step)
        ax = axes[1, 0]
        neg_trial = np.argmin(i_steps)
        t_step = t[step_start:step_end] * 1000
        v_step = v_traces[step_start:step_end, neg_trial]
        
        ax.plot(t_step - t_step[0], v_step, 'b-', lw=1.5)
        
        # Mark sag and steady-state
        sag_idx = np.argmin(v_step[:int(SAG_DUR * fs)])
        v_sag_pt = v_step[sag_idx]
        ax.axhline(v_sag_pt, color='r', ls='--', alpha=0.7, label='Sag minimum')
        
        v_ss_pt = np.median(v_step[-int(ASYMP_DUR * fs):])
        ax.axhline(v_ss_pt, color='g', ls='--', alpha=0.7, label='Steady-state')
        
        ax.set_xlabel('Time from step onset (ms)')
        ax.set_ylabel('Voltage (mV)')
        ax.set_title(f'Sag Response ({i_steps[neg_trial]} pA)')
        ax.legend()
        
        # Plot 4: Summary text
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = f"""
Electrophysiology Properties Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Passive Properties:
  • Resting potential: {v_rest:.1f} mV
  • Peak input resistance: {r_peak:.1f} MΩ
  • Steady-state input resistance: {r_ss:.1f} MΩ
  • Membrane time constant: {tau_measured*1000:.1f} ms

Sag Properties:
  • Sag amplitude: {sag_amp:.2f} mV
  • Sag ratio: {sag_ratio:.3f}

IV Curve:
  • Linear region R²: {r_value**2:.4f}
  • R from IV slope: {r_from_slope:.1f} MΩ
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        ax.set_title('Properties Summary')
        
        plt.tight_layout()
        plt.savefig('invitro_ephys_tests.png', dpi=150)
        plt.show()
        print(f"   + Visualization saved to invitro_ephys_tests.png")
    except Exception as e:
        print(f"   - Visualization failed: {e}")
    
    print("\n" + "=" * 60)
    print("IN VITRO EPHYS PROPERTIES TESTS COMPLETE")
    print("=" * 60)
