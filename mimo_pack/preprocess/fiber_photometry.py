"""Process and correct fiber photometry data
Created by: Drew Headley
"""

import xarray as xr
import numpy as np
from mimo_pack.consts import ATTR_UNITS, DIM_TIME, DIM_CHAN
from mimo_pack.util.xarray import get_fs, validate_dims
from mimo_pack.math.curvefitting import fit_exp_decay
from mimo_pack.math.regression import linear_regression, linear_regression_robust, iir_regression


def _fit_exp_decay_per_channel(t: np.ndarray, values: np.ndarray, n_exp: int,
                               robust: bool) -> np.ndarray:
    """Fit an exponential decay to each column of a (time, channel) array."""
    fit = np.empty_like(values)
    for ch in range(values.shape[1]):
        fit[:, ch], _ = fit_exp_decay(t, values[:, ch], n_exp=n_exp, robust=robust)
    return fit


def exponential_decay_baseline(signal: xr.DataArray, model: str = 'biexponential',
                               robust: bool = False) -> xr.DataArray:
    """
    Fit an exponential decay curve to a signal for use as a baseline.

    Fits an exponential decay (photobleaching) curve to the signal, per
    channel if a channel dimension is present.

    Parameters
    ----------
    signal : xr.DataArray
        The fiber photometry signal. Must have a `DIM_TIME` dimension and
        an `ATTR_FS` attribute.
    model : str, optional
        Either 'monoexponential' or 'biexponential'. Default is
        'biexponential'.
    robust : bool, optional
        If True, fits using robust (soft L1 loss) optimization instead of
        standard least squares. Default is False.

    Returns
    -------
    xr.DataArray
        The fitted baseline F0, with the same coordinates as `signal`.

    Raises
    ------
    TypeError
        If `signal` is not an xr.DataArray.
    ValueError
        If `signal` is missing the `DIM_TIME` dimension, or `model` is
        not recognized.
    """
    if not isinstance(signal, xr.DataArray):
        raise TypeError(f"Expected xr.DataArray, got {type(signal).__name__}.")
    validate_dims(signal, [DIM_TIME])

    if model == 'monoexponential':
        n_exp = 1
    elif model == 'biexponential':
        n_exp = 2
    else:
        raise ValueError(
            f"model must be 'monoexponential' or 'biexponential', got '{model}'."
        )

    t = signal[DIM_TIME].values

    if DIM_CHAN in signal.dims:
        transposed = signal.transpose(DIM_TIME, DIM_CHAN)
        fit = _fit_exp_decay_per_channel(t, transposed.values, n_exp, robust)
        fit_da = transposed.copy(data=fit).transpose(*signal.dims)
    else:
        fit_vals, _ = fit_exp_decay(t, signal.values, n_exp=n_exp, robust=robust)
        fit_da = signal.copy(data=fit_vals)

    # Propagate coordinates and attrs, then update provenance
    fit_da.attrs = signal.attrs.copy()
    desc_base = signal.attrs.get('description', 'fiber photometry signal')
    fit_da.attrs['description'] = f"{desc_base} -> {model} baseline fit"

    return fit_da


def estimate_baseline(signal: xr.DataArray, method: str = 'median', window: float = None,
                      percentile: float = 8.0, exp_model: str = 'biexponential',
                      exp_robust: bool = False) -> xr.DataArray:
    """
    Estimate a baseline F0 for a signal using a chosen method.

    Parameters
    ----------
    signal : xr.DataArray
        The fiber photometry signal. Must have a `DIM_TIME` dimension and
        an `ATTR_FS` attribute.
    method : str, optional
        Baseline method: 'median', 'percentile', or 'exponential'. When
        'exponential', delegates to `exponential_decay_baseline` using
        `exp_model` and `exp_robust`. Default is 'median'.
    window : float, optional
        Rolling window size in seconds for 'median'/'percentile' baselines.
        If None (default), a single global baseline is computed over the
        entire time axis.
    percentile : float, optional
        Percentile (0-100) used when `method='percentile'`. Default is 8.0.
    exp_model : str, optional
        Exponential decay model used when `method='exponential'`, passed
        through to `exponential_decay_baseline`. Default is 'biexponential'.
    exp_robust : bool, optional
        Robust fitting flag used when `method='exponential'`, passed
        through to `exponential_decay_baseline`. Default is False.

    Returns
    -------
    xr.DataArray
        The estimated baseline F0. Has the same coordinates as `signal`
        for 'exponential' or windowed methods, or is reduced over
        `DIM_TIME` for global 'median'/'percentile' baselines.

    Raises
    ------
    TypeError
        If `signal` is not an xr.DataArray.
    ValueError
        If `signal` is missing the `DIM_TIME` dimension, `method` is not
        recognized, or `window` is longer than the signal duration.
    """
    if not isinstance(signal, xr.DataArray):
        raise TypeError(f"Expected xr.DataArray, got {type(signal).__name__}.")
    validate_dims(signal, [DIM_TIME])

    if method == 'exponential':
        return exponential_decay_baseline(signal, model=exp_model, robust=exp_robust)

    if method not in ('median', 'percentile'):
        raise ValueError(
            f"method must be 'median', 'percentile', or 'exponential', got '{method}'."
        )

    if window is not None:
        fs = get_fs(signal)
        window_samples = int(round(window * fs))
        if window_samples > signal.sizes[DIM_TIME]:
            raise ValueError(
                f"window ({window}s, {window_samples} samples) is longer than "
                f"the signal duration ({signal.sizes[DIM_TIME]} samples)."
            )
        rolling = signal.rolling({DIM_TIME: window_samples}, center=True,
                                 min_periods=1)
        if method == 'median':
            baseline = rolling.median()
        else:
            baseline = rolling.reduce(np.nanpercentile, q=percentile)
        window_desc = f"{window}s rolling window"
    else:
        if method == 'median':
            baseline = signal.median(dim=DIM_TIME)
        else:
            baseline = signal.quantile(percentile / 100.0, dim=DIM_TIME)
            baseline = baseline.drop_vars('quantile', errors='ignore')
        window_desc = "global"

    # Propagate attrs, then update provenance
    baseline.attrs = signal.attrs.copy()
    desc_base = signal.attrs.get('description', 'fiber photometry signal')
    method_desc = f"{method} ({percentile}th pctl)" if method == 'percentile' else method
    baseline.attrs['description'] = (
        f"{desc_base} -> {method_desc} baseline ({window_desc})"
    )

    return baseline


def isosbestic_baseline(signal: xr.DataArray, isosbestic: xr.DataArray,
                        method: str = 'robust', order: int = 2) -> xr.DataArray:
    """
    Estimate a baseline by regressing the isosbestic channel onto the signal.

    Parameters
    ----------
    signal : xr.DataArray
        The fiber photometry signal channel(s). Must have a `DIM_TIME`
        dimension.
    isosbestic : xr.DataArray
        The isosbestic control channel(s), matching `signal` in shape.
        Ordering of channels must match that of `signal`.
    method : str, optional
        One of 'ols' (ordinary least squares), 'robust' (Huber regression),
        or 'iir' (acausal IIR model). Default is 'robust'.
    order : int, optional
        The order of the IIR model, used only when `method='iir'`. Default
        is 2.

    Returns
    -------
    xr.DataArray
        The isosbestic-derived baseline F0, having the same coordinates
        as `signal`.

    Raises
    ------
    TypeError
        If `signal` or `isosbestic` is not an xr.DataArray.
    ValueError
        If either input is missing the `DIM_TIME` dimension, the shapes
        do not match, or `method` is not recognized.
    """
    if not isinstance(signal, xr.DataArray):
        raise TypeError(f"Expected xr.DataArray for signal, got {type(signal).__name__}.")
    if not isinstance(isosbestic, xr.DataArray):
        raise TypeError(
            f"Expected xr.DataArray for isosbestic, got {type(isosbestic).__name__}."
        )
    validate_dims(signal, [DIM_TIME])
    validate_dims(isosbestic, [DIM_TIME])
    if signal.shape != isosbestic.shape:
        raise ValueError(
            f"signal and isosbestic must have matching shapes, got "
            f"{signal.shape} and {isosbestic.shape}."
        )
    if method == 'ols':
        regress_fn = linear_regression
    elif method == 'robust':
        regress_fn = linear_regression_robust
    elif method == 'iir':
        regress_fn = lambda y, x: iir_regression(y, x, order=order)
    else:
        raise ValueError(f"method must be 'ols', 'robust', or 'iir', got '{method}'.")

    if DIM_CHAN in signal.dims:
        sig_t = signal.transpose(DIM_TIME, DIM_CHAN)
        iso_t = isosbestic.transpose(DIM_TIME, DIM_CHAN)
        fitted = np.empty_like(sig_t.values)
        for ch in range(fitted.shape[1]):
            _, _, fitted[:, ch] = regress_fn(sig_t.values[:, ch], iso_t.values[:, ch])
        fitted_da = sig_t.copy(data=fitted).transpose(*signal.dims)
    else:
        _, _, fitted = regress_fn(signal.values, isosbestic.values)
        fitted_da = signal.copy(data=fitted)

    # Propagate attrs, then update provenance
    fitted_da.attrs = signal.attrs.copy()
    desc_base = signal.attrs.get('description', 'fiber photometry signal')
    fitted_da.attrs['description'] = f"{desc_base} -> isosbestic regression baseline ({method})"

    return fitted_da


def df_over_f(signal: xr.DataArray, baseline: xr.DataArray) -> xr.DataArray:
    """
    Compute ΔF/F0 from a signal and its baseline F0.

    Parameters
    ----------
    signal : xr.DataArray
        The fiber photometry signal.
    baseline : xr.DataArray
        The baseline F0, with coordinates aligned to (or broadcastable
        against) `signal`, e.g. as returned by `exponential_decay_baseline`,
        `estimate_baseline`, or `isosbestic_baseline`.

    Returns
    -------
    xr.DataArray
        The ΔF/F0 signal, (signal - baseline) / baseline, with the same
        coordinates as `signal`.

    Raises
    ------
    TypeError
        If `signal` or `baseline` is not an xr.DataArray.

    Examples
    --------
    >>> from mimo_pack.testing.data import synthetic_fiber_photometry
    >>> signal, _ = synthetic_fiber_photometry(duration=60.0, fs=100.0)
    >>> baseline = estimate_baseline(signal, method='median')
    >>> dff = df_over_f(signal, baseline)
    """
    if not isinstance(signal, xr.DataArray):
        raise TypeError(f"Expected xr.DataArray for signal, got {type(signal).__name__}.")
    if not isinstance(baseline, xr.DataArray):
        raise TypeError(f"Expected xr.DataArray for baseline, got {type(baseline).__name__}.")

    dff = (signal - baseline) / baseline

    # Propagate attrs, then update provenance
    dff.attrs = signal.attrs.copy()
    dff.attrs[ATTR_UNITS] = 'ΔF/F0'
    desc_base = signal.attrs.get('description', 'fiber photometry signal')
    dff.attrs['description'] = f"{desc_base} -> \u0394F/F0"

    return dff
