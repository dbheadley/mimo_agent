# Functions for curve fitting
# Author: Drew Headley
# Date: 2024-06-13

from scipy.optimize import curve_fit, least_squares
import numpy as np
from rdp import rdp


def fit_sine_freq(x, xp, yp, freq):
    """
    Fit a sine wave of specified frequency to data using the Fourier equation approach.

    Parameters
    ----------
    x : array-like
        The x values to evaluate the curve at.
    xp : array-like
        The x values of the data to fit.
    yp : array-like
        The y values of the data to fit.
    freq : float
        The frequency of the sine wave (Hz).

    Returns
    -------
    fit : array-like
        The fitted sine wave values at x.
    params : dict
        Dictionary with amplitude, phase, and offset.
    """
    xp = np.asarray(xp)
    yp = np.asarray(yp)
    omega = 2 * np.pi * freq
    sin_basis = np.sin(omega * xp)
    cos_basis = np.cos(omega * xp)
    X = np.column_stack([sin_basis, cos_basis, np.ones_like(xp)])
    beta, _, _, _ = np.linalg.lstsq(X, yp, rcond=None)
    amp = np.hypot(beta[0], beta[1])
    phase = np.arctan2(beta[1], beta[0])
    offset = beta[2]
    fit = amp * np.sin(omega * x + phase) + offset
    params = {'frequency': freq, 'amplitude': amp, 'phase': phase, 'offset': offset}
    return fit, params
    
# half-gaussian fit
def half_gauss(x, amp, sd, offset):
    """
    Curve for a positive sided gaussian curve.
    
    Parameters
    ----------
    x : array-like
        The x values to evaludate the curve at.
    amp : float
        The amplitude of the curve.
    sd : float
        The standard deviation of the curve.
    offset : float
        The offset of the curve (mean).
        
    Returns
    -------
    y : array-like
        The y value of the curve at the given x values.
    """

    y = amp*np.exp(-((x-offset)**2)/(2*sd**2))*(x>offset)
    return y

def fit_half_gauss(x, xp, yp, **kwargs):
    """
    Fit a half-gaussian curve to the data.

    Parameters
    ----------
    x : array-like
        The x values to evaludate the curve at.
    xp : array-like
        The x values of the data to fit.
    yp : array-like
        The y values of the data to fit.
    kwargs : dict
        Additional arguments to pass to curve_fit.

    Returns
    -------
    fit : array-like
        The y values of the curve at the given x values.
    popt : array-like
        The optimized parameters of the curve.
    """

    popt, _ = curve_fit(half_gauss, xp, yp, **kwargs)
    fit = half_gauss(x, *popt)
    return fit, popt

def piecewise_table_monotonic(vals, thresh=0.001):
    """
    Reduce a list of monotonically increasing values to a table of key 
    points that define a piecewise function. The rate of increase is assumed 
    to be the median difference between adjacent values. Error in the values that 
    would be returned by a piecewise linear interpolation function is kept 
    below an error threshold.
    
    Parameters
    ----------
    vals : N-length numpy array
        The values to reduce. 
        
    Optional
    --------
    threshold : float
        Maximum allowable error in the piecewise interpolation. Default
        is half the median difference between adjacent values.
        
    Returns
    -------
    table : Kx2 numpy array
        The key points of the piecewise function. The first column is the index
        of the keypoint in vals, and the second column is the value in vals at 
        that index. K is the number of key points found. There is always at least
        two, for the start and end of vals.


    Example
    -------
    import matplotlib.pyplot as plt
    test = np.array([np.nan, 1,2,3,4,5,6,8,9,11,12,13,14,15,16,17,
                     np.nan,np.nan,19,20])
    table = piecewise_table_monotonic(test)
    plt.plot(test)
    plt.plot(table[:,0], table[:,1], 'ro')
    """

    # Use Ramer-Douglas-Peucker algorithm to reduce the points
    nan_mask = np.isnan(vals)
    valid_inds = np.where(~nan_mask)[0]
    arr = np.column_stack((valid_inds, vals[~nan_mask]))
    table = rdp(arr, epsilon=thresh)

    
    # vals_num = len(vals)
    # table = np.array([[0, vals[0]]])
    # slope = np.nanmedian(np.diff(vals))
    # if thresh is None:
    #     thresh = slope/2

    # # Find the starts and ends of NaN values so that all of them can be included in the table
    # nan_edges = np.argwhere(np.abs(np.diff(np.isnan(vals)))==1)
    # if len(nan_edges) > 0:
    #     nan_edges = nan_edges.flatten()[np.newaxis,:]+1
    #     table = np.vstack((table, np.vstack((nan_edges, vals[nan_edges])).T, 
    #                        np.vstack((nan_edges-1, vals[nan_edges-1])).T))
                
    # ind = 0
    # while ind < vals_num-1:
    #     if np.isnan(vals[ind]):
    #         next_ind = np.argwhere(~np.isnan(vals[ind:]))
    #     else:
    #         pred_vals = vals[ind] + slope*np.arange(0, ((vals_num-1)-ind))
    #         cum_err = np.abs(np.nancumsum(vals[ind:-1]-pred_vals))
    #         next_ind = np.argwhere(cum_err>thresh)

    #     if len(next_ind) == 0:
    #         break
    #     else:
    #         next_ind = next_ind[0][0]

    #     if (ind>0):
    #         table = np.vstack((table, [ind+next_ind-1, vals[ind+next_ind-1]]))
    #     table = np.vstack((table, [ind+next_ind, vals[ind+next_ind]]))
    #     ind += next_ind
    
    # table = np.vstack((table, [vals_num-1, vals[-1]]))
    # uniq_inds = np.unique(table[:,0], return_index=True)[1]
    # table = table[uniq_inds]
    return table


def exp_knee(x, offset, knee, exp):
    """
    Exponential knee function.
    Based on function from FOOOF package.
    
    Parameters
    ----------
    x : array-like
        The x values to evaluate the curve at.
    offset : float
        The offset of the curve.
    knee : float
        The knee point of the curve.
    exp : float
        The exponential factor of the curve.
        
    Returns
    -------
    y : array-like
        The y value of the curve at the given x values.
    """
    
    return offset - np.log10(knee + x**exp)

def fit_exp_knee(x, xp, yp, suppress_error=False, **kwargs):
    """
    Fit an exponential knee curve to the data.
    When working with spectral data, x values should
    be linearly spaced frequencies, and y values should
    be the log10 power spectral density values.

    Parameters
    ----------
    x : array-like
        The x values to evaluate the curve at.
    xp : array-like
        The x values of the data to fit.
    yp : array-like
        The y values of the data to fit.
    suppress_error : bool, optional
        If True, suppresses errors during fitting. 
        Default is False, which raises an error if fitting fails.
    kwargs : dict
        Additional arguments to pass to curve_fit.

    Returns
    -------
    fit : array-like
        The y values of the curve at the given x values.
    popt : array-like
        The optimized parameters of the curve.
    """

    # initial guess for the parameters
    if 'p0' not in kwargs:
        offset_guess = yp[np.where(xp == np.min(xp))[0][0]]
        knee_guess = 0.1
        exp_guess = 1.0
        kwargs['p0'] = [offset_guess, knee_guess, exp_guess]

    if suppress_error:
        try:
            popt, _ = curve_fit(exp_knee, xp, yp, **kwargs)
        except Exception as e:
            print(f"Error fitting exp_knee: {e}")
            popt = np.array([np.nan, np.nan, np.nan])
    else:
        popt, _ = curve_fit(exp_knee, xp, yp, **kwargs)

    fit = exp_knee(x, *popt)
    return fit, popt


def exp_decay(t, *params):
    """
    Mono- or bi-exponential decay basis function.

    Parameters
    ----------
    t : array-like
        The time values to evaluate the curve at.
    *params : float
        Either 3 parameters (amp, tau, offset) for a mono-exponential decay,
        or 5 parameters (amp1, tau1, amp2, tau2, offset) for a
        bi-exponential decay.

    Returns
    -------
    y : array-like
        The value of the decay curve at the given time values.

    Raises
    ------
    ValueError
        If the number of parameters is not 3 or 5.
    """
    t = np.asarray(t)
    if len(params) == 3:
        amp, tau, offset = params
        return amp * np.exp(-t / tau) + offset
    elif len(params) == 5:
        amp1, tau1, amp2, tau2, offset = params
        return amp1 * np.exp(-t / tau1) + amp2 * np.exp(-t / tau2) + offset
    else:
        raise ValueError(
            f"exp_decay expects 3 (mono-exponential) or 5 (bi-exponential) "
            f"parameters, got {len(params)}."
        )


def fit_exp_decay(t, y, n_exp: int = 1, robust: bool = False, **kwargs) -> tuple:
    """
    Fit a mono- or bi-exponential decay curve to the data.

    Parameters
    ----------
    t : array-like
        The time values of the data to fit.
    y : array-like
        The y values of the data to fit.
    n_exp : int, optional
        Number of exponential terms: 1 for mono-exponential (default), or
        2 for bi-exponential decay.
    robust : bool, optional
        If True, fits using `scipy.optimize.least_squares` with a
        `soft_l1` loss for robustness to outliers, instead of
        `scipy.optimize.curve_fit`. Default is False.
    kwargs : dict
        Additional arguments to pass to `curve_fit` (when `robust=False`)
        or `least_squares` (when `robust=True`).

    Returns
    -------
    fit : array-like
        The fitted decay curve values at t.
    popt : array-like
        The optimized parameters of the curve: (amp, tau, offset) for
        mono-exponential, or (amp1, tau1, amp2, tau2, offset) for
        bi-exponential.

    Raises
    ------
    ValueError
        If `n_exp` is not 1 or 2.
    """
    if n_exp not in (1, 2):
        raise ValueError(f"n_exp must be 1 or 2, got {n_exp}.")

    t = np.asarray(t)
    y = np.asarray(y)

    # Initial guesses based on the data range
    amp_guess = y[0] - y[-1]
    tau_guess = (t[-1] - t[0]) / 4 if t[-1] != t[0] else 1.0
    offset_guess = y[-1]

    if n_exp == 1:
        p0 = kwargs.pop('p0', [amp_guess, tau_guess, offset_guess])
    else:
        p0 = kwargs.pop(
            'p0',
            [amp_guess / 2, tau_guess, amp_guess / 2, tau_guess * 4, offset_guess],
        )

    if robust:
        residual = lambda p: exp_decay(t, *p) - y
        result = least_squares(residual, p0, loss='soft_l1', **kwargs)
        popt = result.x
    else:
        kwargs.setdefault('maxfev', 10000)
        popt, _ = curve_fit(exp_decay, t, y, p0=p0, **kwargs)

    fit = exp_decay(t, *popt)
    return fit, popt
