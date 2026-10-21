# Functions for cross-correlation analyses
# Author: Drew Headley
# Date: 2025-07-11

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pynapple as nap
from numba import njit
from scipy.stats import poisson as sp_poisson
from tqdm import tqdm

def cc_to_prob(cc, counts=None, norm=True):
    """
    Convert cross-correlogram to probability conditioned on the reference spike.

    Parameters
    ----------
    cc : pandas.DataFrame
        Cross-correlogram DataFrame returned by pynapple.compute_correlogram.
    counts : pandas.DataFrame, optional
        Spike counts, indexed by spike group IDs. Only used if norm is True.
    norm : bool, optional
        Indicates whether the computed cross-correlogram was normalized by the
        target unit's firing rate. By default, Pynapple normalizes the cross-correlogram.

    Returns
    -------
    pandas.DataFrame
        Probability conditioned on the reference spike.
    """

    cc_prob = cc.copy()
    
    bin_width = np.median(np.diff(cc.index))
    if norm:
        if counts is None:
            raise ValueError("counts must be provided if norm is True.")

        for i in range(len(cc.columns)):
            target_id = cc.columns[i][1]
            cc_prob.iloc[:, i] = cc.iloc[:, i] * counts[target_id].values * bin_width
    else:
        for i in range(len(cc.columns)):
            cc_prob.iloc[:, i] = cc.iloc[:, i] * bin_width

    return cc_prob

def cc_correct_poly(cc, exclude_lag=0.005, order=12):
    """
    Corrects the cross-correlogram by fitting a polynomial to the baseline and subtracting it.

    Parameters
    ----------
    cc : nap.Ts
        Cross-correlogram to be corrected.
    exclude_lag : float, optional
        Time lag to exclude from the polynomial fit.
    order : int, optional
        Order of the polynomial to fit.

    Returns
    -------
    cc_corr : nap.Ts
        Corrected cross-correlogram.
    cc_baseline : nap.Ts
        Baseline polynomial fit.
    """
    lags = cc.index.values
    w = (np.abs(lags) > exclude_lag).astype(float)
    cent_lags = np.abs(lags) <= 0.0006

    # Overweight weights for central near 0 ms lags, accounting for the 
    # excluded lags in the monosynaptic window
    w[cent_lags] = (w.size-np.sum(w))/(4*np.sum(cent_lags))

    cc_corr = cc.copy()
    cc_baseline = cc_corr.copy()
    for i in range(len(cc_corr.columns)):        
        # Fit polynomial to the baseline
        p = np.polyfit(lags, cc_corr.iloc[:, i], order, w=w)
        baseline = np.polyval(p, lags)
        cc_corr.iloc[:, i] -= baseline
        cc_baseline.iloc[:, i] = baseline
    
    return cc_corr, cc_baseline

def ccg_convolution_significance(cc, kernel_half_width=5):
    """
    Compute per-bin Poisson p-values for a cross-correlogram using a
    moving-median baseline.

    Separate baseline widths are used for excitatory (narrow window) and
    inhibitory (2x-wide window) deviations. For each bin, the excitatory
    p-value is P(X >= count) and the inhibitory p-value is P(X <= count)
    under a Poisson distribution with mean equal to the moving-median
    baseline.

    Parameters
    ----------
    cc : pandas.DataFrame
        Cross-correlogram of raw spike counts, indexed by lag (seconds),
        with one column per unit pair, as returned by
        pynapple.compute_crosscorrelogram (converted to counts).
    kernel_half_width : int, optional
        Half-width of the moving-median kernel in bins. The excitatory
        baseline window is ``2 * kernel_half_width + 1`` bins wide and the
        inhibitory window is twice that. The two bins closest to zero lag are
        excluded from baseline estimation. Default is 5.

    Returns
    -------
    p_vals_ex : pandas.DataFrame
        Per-bin excitatory p-values, P(X >= count).
    p_vals_in : pandas.DataFrame
        Per-bin inhibitory p-values, P(X <= count).
    baseline_ex : pandas.DataFrame
        Moving-median baseline used for the excitatory test.
    baseline_in : pandas.DataFrame
        Moving-median baseline used for the inhibitory test.
    """
    if not isinstance(cc, pd.DataFrame):
        raise TypeError("cc must be a pandas.DataFrame of cross-correlogram counts.")

    win = 2 * kernel_half_width + 1
    lag_values = cc.index.to_numpy(dtype=float)
    excluded_bins = np.argsort(np.abs(lag_values), kind="stable")[:2]

    def moving_median_baseline(counts, size):
        pad_left = size // 2
        pad_right = size - 1 - pad_left
        padded_counts = np.pad(counts, (pad_left, pad_right), mode="symmetric")
        padded_bins = np.pad(
            np.arange(counts.size), (pad_left, pad_right), mode="symmetric"
        )
        windows = np.lib.stride_tricks.sliding_window_view(padded_counts, size)
        bin_windows = np.lib.stride_tricks.sliding_window_view(padded_bins, size)
        excluded = np.isin(bin_windows, excluded_bins)
        return np.nanmedian(np.where(excluded, np.nan, windows), axis=1)

    baseline_ex = cc.copy().astype(float)
    baseline_in = cc.copy().astype(float)
    p_vals_ex   = pd.DataFrame(np.nan, index=cc.index, columns=cc.columns)
    p_vals_in   = pd.DataFrame(np.nan, index=cc.index, columns=cc.columns)

    for col in tqdm(cc.columns, desc='CCG significance', leave=False):
        counts = cc[col].values.astype(float)
        bl_ex  = moving_median_baseline(counts, win)
        bl_in  = moving_median_baseline(counts, 2 * win)
        baseline_ex[col] = bl_ex
        baseline_in[col] = bl_in
        mu_ex = np.maximum(bl_ex, 1e-10)
        mu_in = np.maximum(bl_in, 1e-10)
        p_vals_ex[col] = sp_poisson.sf(counts - 1, mu_ex)   # P(X >= count)
        p_vals_in[col] = sp_poisson.cdf(counts,    mu_in)   # P(X <= count)

    return p_vals_ex, p_vals_in, baseline_ex, baseline_in

def _mad_significant_run(window_mask, signif_ex, signif_in, excess, undershoot):
    """
    Test the maximum-absolute-deviation (MAD) bin of a lag window and, if it
    is significant, grow the run of significant bins outward from it.

    Only one bin is tested, so no multiple-comparison correction is needed.
    The MAD bin is the bin with the largest absolute deviation from baseline
    among the excitatory (largest positive ``excess``) and inhibitory
    (largest negative ``undershoot``) candidates; ties go to excitatory.
    The run extends toward earlier and later lags while adjacent bins remain
    significant (same sign, same uncorrected alpha) and inside the window.

    Parameters
    ----------
    window_mask : numpy.ndarray
        Boolean mask over lag bins selecting the (contiguous) lag window.
    signif_ex, signif_in : numpy.ndarray
        Boolean arrays flagging bins significant for the excitatory /
        inhibitory test.
    excess, undershoot : numpy.ndarray
        Deviation of counts from the excitatory / inhibitory baseline.

    Returns
    -------
    tuple or None
        ``(peak, sign, start, end)`` bin indices (``start <= peak <= end``) and
        sign (+1/-1), or None if the window is empty or its MAD bin is not
        significant.
    """
    idx = np.flatnonzero(window_mask)
    if idx.size == 0:
        return None
    lo, hi = int(idx[0]), int(idx[-1])

    dev_ex = np.nan_to_num(excess[lo:hi + 1], nan=-np.inf)
    dev_in = np.nan_to_num(-undershoot[lo:hi + 1], nan=-np.inf)
    ex_peak = int(np.argmax(dev_ex))
    in_peak = int(np.argmax(dev_in))
    if dev_ex[ex_peak] >= dev_in[in_peak]:
        sign, peak, signif = 1, lo + ex_peak, signif_ex
    else:
        sign, peak, signif = -1, lo + in_peak, signif_in

    if not signif[peak]:
        return None

    start = end = peak
    while start > lo and signif[start - 1]:
        start -= 1
    while end < hi and signif[end + 1]:
        end += 1
    return peak, sign, start, end

def find_significant_connections(p_vals_ex, p_vals_in, baseline_ex, baseline_in, cc,
                                  lag_min_abs, lag_max_abs,
                                  min_contiguous_bins=2, alpha=0.05,
                                  max_contiguous_bins=10):
    """
    Return significant monosynaptic connections from CCG p-value DataFrames.

    For each pair column and each lag window (positive: ``lag_min_abs < lag <=
    lag_max_abs``; negative: the mirrored window), only the bin of maximum
    absolute deviation (MAD) of the spike count from baseline is tested. The
    MAD bin is the larger of the largest excitatory deviation (count minus
    excitatory baseline) and the largest inhibitory deviation (baseline_in
    minus count) in the window. Because a single bin is tested per window and
    direction, no multiple-comparison (Bonferroni) correction is applied; the
    bin is significant if its p-value is below ``alpha``.

    If the MAD bin is significant, the run of significant bins is grown
    outward from it toward earlier and later lags, staying within the window.
    A neighbor joins the run if it has the same sign and is significant at the
    same uncorrected ``alpha``; growth stops at the first non-significant bin
    or at the window edge. The run is reported if its width in bins is between
    ``min_contiguous_bins`` and ``max_contiguous_bins`` (inclusive). This
    yields up to one connection per ordered pair direction.

    Parameters
    ----------
    p_vals_ex : pandas.DataFrame
        Per-bin excitatory p-values from ccg_convolution_significance.
    p_vals_in : pandas.DataFrame
        Per-bin inhibitory p-values from ccg_convolution_significance.
    baseline_ex : pandas.DataFrame
        Excitatory moving-median baseline from ccg_convolution_significance.
    baseline_in : pandas.DataFrame
        Inhibitory moving-median baseline from ccg_convolution_significance.
    cc : pandas.DataFrame
        Cross-correlogram of raw spike counts, indexed by lag (seconds).
    lag_min_abs : float
        Minimum absolute lag (seconds) for a monosynaptic connection,
        excluding the near-zero artifact window.
    lag_max_abs : float
        Maximum absolute lag (seconds) for a monosynaptic transmission delay.
    min_contiguous_bins : int, optional
        Minimum width (bins) of the significant run around the MAD bin; the
        run is confined to the lag window. Default is 2.
    alpha : float, optional
        Uncorrected significance level applied to the MAD bin and to each
        bin added to the run. Default is 0.05.
    max_contiguous_bins : int, optional
        Maximum width (bins) of the significant run; wider runs are treated
        as broad co-modulation and rejected. Default is 10.

    Returns
    -------
    list of dict
        One dict per detected connection with keys:

        direction : tuple
            (pre_unit_id, post_unit_id).
        lag_s : float
            Absolute transmission delay (seconds) of the MAD bin.
        sign : int
            +1 (excitatory) or -1 (inhibitory).
        amplitude : float
            (count - baseline) / baseline at the MAD bin.
        run_start_s : float
            Smallest absolute lag (seconds) in the significant run.
        run_end_s : float
            Largest absolute lag (seconds) in the significant run.
        run_width_bins : int
            Number of bins in the significant run.

    Raises
    ------
    TypeError
        If ``cc`` is not a pandas.DataFrame.

    Examples
    --------
    >>> p_ex, p_in, bl_ex, bl_in = ccg_convolution_significance(cc_counts)
    >>> conns = find_significant_connections(
    ...     p_ex, p_in, bl_ex, bl_in, cc_counts,
    ...     lag_min_abs=0.001, lag_max_abs=0.010)
    >>> conns[0]['run_width_bins']  # doctest: +SKIP
    3
    """
    if not isinstance(cc, pd.DataFrame):
        raise TypeError("cc must be a pandas.DataFrame of cross-correlogram counts.")

    lags       = cc.index.values
    pos_window = (lags >  lag_min_abs) & (lags <= lag_max_abs)
    neg_window = (lags >= -lag_max_abs) & (lags < -lag_min_abs)

    connections = []
    for col in tqdm(cc.columns, desc='Finding connections', leave=False):
        unit_i, unit_j = col
        counts     = cc[col].values.astype(float)
        bl_ex      = baseline_ex[col].values.astype(float)
        bl_in      = baseline_in[col].values.astype(float)
        excess     = counts - bl_ex
        undershoot = counts - bl_in

        # Uncorrected alpha: only the MAD bin (and its neighbors, tested
        # one at a time while growing the run) is ever evaluated.
        signif_ex = (p_vals_ex[col].values < alpha) & (excess > 0) & (bl_ex > 0)
        signif_in = (p_vals_in[col].values < alpha) & (undershoot < 0)

        for window_mask, direction in [
            (pos_window, (unit_i, unit_j)),
            (neg_window, (unit_j, unit_i)),
        ]:
            run = _mad_significant_run(window_mask, signif_ex, signif_in,
                                       excess, undershoot)
            if run is None:
                continue
            sel, sign, start, end = run
            width = end - start + 1
            if not min_contiguous_bins <= width <= max_contiguous_bins:
                continue
            bl_peak = bl_ex[sel] if sign == 1 else bl_in[sel]
            run_lags = np.abs(lags[[start, end]])
            connections.append({
                'direction':      direction,
                'lag_s':          abs(lags[sel]),
                'sign':           sign,
                'amplitude':      (counts[sel] - bl_peak) / bl_peak if bl_peak > 0 else np.nan,
                'run_start_s':    float(run_lags.min()),
                'run_end_s':      float(run_lags.max()),
                'run_width_bins': int(width),
            })

    return connections


@njit(nogil=True, cache=True)
def _ccg_counts(t_ref, t_tgt, binsize, n_lags, out):
    """Accumulate CCG counts of t_tgt relative to t_ref into ``out`` in place.

    Bins match pynapple's ``compute_crosscorrelogram``: with half-width
    ``w = n_lags / 2 * binsize``, bin j spans
    ``[-w + j * binsize, -w + (j + 1) * binsize)``. Both inputs must be sorted.
    """
    w = (n_lags / 2.0) * binsize
    n_tgt = t_tgt.size
    i2 = 0
    for i1 in range(t_ref.size):
        lbound = t_ref[i1] - w
        while i2 < n_tgt and t_tgt[i2] < lbound:
            i2 += 1
        rbound = t_ref[i1] + w
        k = i2
        while k < n_tgt and t_tgt[k] < rbound:
            j = int((t_tgt[k] - lbound) / binsize)
            if j >= n_lags:
                j = n_lags - 1
            out[j] += 1.0
            k += 1


@njit(nogil=True, cache=True)
def _interval_jitter(win_start, jitter_window, out):
    """Redraw each spike uniformly within its jitter window, then sort ``out``.

    ``win_start`` is sorted, so the jittered times are only out of order within
    a window and an insertion sort is close to linear.
    """
    n = win_start.size
    for i in range(n):
        out[i] = win_start[i] + np.random.random() * jitter_window
    for i in range(1, n):
        v = out[i]
        k = i - 1
        while k >= 0 and out[k] > v:
            out[k + 1] = out[k]
            k -= 1
        out[k + 1] = v


@njit(nogil=True, cache=True)
def _jitter_spike_times_kernel(win_start, jitter_window, n_surrogates, seed):
    np.random.seed(seed)
    out = np.empty((n_surrogates, win_start.size))
    for s in range(n_surrogates):
        _interval_jitter(win_start, jitter_window, out[s])
    return out


@njit(nogil=True, cache=True)
def _jitter_ccg_kernel(t_ref, win_start, binsize, n_lags, jitter_window,
                       n_surrogates, seed):
    """Surrogate CCG counts (n_surrogates, n_lags) with the target jittered."""
    np.random.seed(seed)
    out = np.zeros((n_surrogates, n_lags))
    jit = np.empty(win_start.size)
    for s in range(n_surrogates):
        _interval_jitter(win_start, jitter_window, jit)
        _ccg_counts(t_ref, jit, binsize, n_lags, out[s])
    return out


def _spike_times_array(spike_times, name='spike_times'):
    """Return sorted float64 spike times from a nap.Ts/Tsd or array-like."""
    if isinstance(spike_times, (nap.Ts, nap.Tsd)):
        t = np.asarray(spike_times.t, dtype=np.float64)
    elif isinstance(spike_times, (np.ndarray, list, tuple)):
        t = np.asarray(spike_times, dtype=np.float64)
    else:
        raise TypeError(
            f"{name} must be a pynapple.Ts/Tsd or a 1D array of times in seconds, "
            f"got {type(spike_times).__name__}.")
    if t.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {t.shape}.")
    return np.sort(t)


def _jitter_window_starts(t, jitter_window):
    return np.floor(t / jitter_window) * jitter_window


def jitter_spike_times(spike_times, jitter_window=0.004, n_surrogates=1, seed=None):
    """
    Generate interval-jittered surrogates of a spike train.

    Time is partitioned into fixed, non-overlapping windows
    ``[k * jitter_window, (k + 1) * jitter_window)`` and each spike is redrawn
    uniformly within its own window (Amarasingham et al., 2012). Spike counts
    per window, and therefore firing-rate fluctuations slower than the window,
    are preserved while finer spike timing is destroyed.

    Parameters
    ----------
    spike_times : pynapple.Ts or array-like
        Spike times in seconds.
    jitter_window : float, optional
        Jitter window width in seconds. Default is 0.004.
    n_surrogates : int, optional
        Number of surrogate spike trains. Default is 1.
    seed : int, optional
        Seed for reproducible surrogates.

    Returns
    -------
    numpy.ndarray
        Jittered spike times, shape (n_surrogates, n_spikes). Each row is sorted.

    Examples
    --------
    >>> t = np.sort(np.random.uniform(0, 10, 100))
    >>> surr = jitter_spike_times(t, jitter_window=0.004, n_surrogates=10, seed=0)
    >>> surr.shape
    (10, 100)
    """
    t = _spike_times_array(spike_times)
    if not np.isscalar(jitter_window) or jitter_window <= 0:
        raise ValueError(f"jitter_window must be a positive number, got {jitter_window}.")
    if int(n_surrogates) < 1:
        raise ValueError(f"n_surrogates must be >= 1, got {n_surrogates}.")
    seed_int = int(np.random.SeedSequence(seed).generate_state(1, dtype=np.uint32)[0])
    return _jitter_spike_times_kernel(
        _jitter_window_starts(t, float(jitter_window)), float(jitter_window),
        int(n_surrogates), seed_int)


def ccg_jitter_significance(spks, cc, jitter_window=0.004, n_surrogates=1000,
                            n_jobs=1, seed=None, p_method='poisson'):
    """
    Compute per-bin p-values for cross-correlograms against interval-jitter
    surrogates of the target unit.

    For each ``(ref, tgt)`` column of ``cc`` the target spike train is
    interval-jittered (see :func:`jitter_spike_times`) ``n_surrogates`` times
    and the surrogate CCG counts are computed against the unjittered reference
    spikes, using bins aligned to ``cc.index``. The surrogate mean is the
    baseline that captures co-modulation slower than ``jitter_window``. The
    outputs have the same format as :func:`ccg_convolution_significance`, so
    they can be passed directly to :func:`find_significant_connections`.

    Parameters
    ----------
    spks : pynapple.TsGroup
        Spike trains in seconds; must contain every unit referenced by the
        columns of ``cc``. Use the same (restricted) group that produced ``cc``.
    cc : pandas.DataFrame
        Cross-correlogram spike counts indexed by lag (seconds) with
        ``(ref, tgt)`` tuple columns, as derived from
        ``pynapple.compute_crosscorrelogram(norm=False)``. Typically a subset
        of columns that passed :func:`ccg_convolution_significance`.
    jitter_window : float, optional
        Interval-jitter window width in seconds. Default is 0.004.
    n_surrogates : int, optional
        Number of jittered surrogates per pair. Default is 1000.
    n_jobs : int, optional
        Number of threads used across pairs; -1 uses all CPUs. Default is 1.
    seed : int, optional
        Seed for reproducible surrogates. Results do not depend on ``n_jobs``.
    p_method : {'poisson', 'empirical'}, optional
        ``'poisson'`` (default) computes Poisson tail probabilities with mean
        equal to the surrogate mean. Under target jitter each bin count is a
        sum of independent Bernoulli variables (Poisson-binomial), whose
        variance does not exceed its mean, so Poisson tails are conservative.
        Unlike empirical p-values, they can resolve small significance
        thresholds used by :func:`find_significant_connections`.
        ``'empirical'`` returns ``(1 + #{surrogate >= count}) / (1 + N)``
        (excitatory) and ``(1 + #{surrogate <= count}) / (1 + N)``
        (inhibitory); these cannot fall below ``1 / (1 + n_surrogates)``.

    Returns
    -------
    p_vals_ex : pandas.DataFrame
        Per-bin excitatory p-values, P(X >= count).
    p_vals_in : pandas.DataFrame
        Per-bin inhibitory p-values, P(X <= count).
    baseline_ex : pandas.DataFrame
        Surrogate mean counts per bin.
    baseline_in : pandas.DataFrame
        Surrogate mean counts per bin (identical to ``baseline_ex``).

    Examples
    --------
    >>> p_ex, p_in, bl_ex, bl_in = ccg_jitter_significance(
    ...     su_spks, cc_counts[tested_cols], jitter_window=0.004,
    ...     n_surrogates=1000, n_jobs=-1, seed=0)
    >>> conns = find_significant_connections(
    ...     p_ex, p_in, bl_ex, bl_in, cc_counts[tested_cols],
    ...     lag_min_abs=0.001, lag_max_abs=0.010)
    """
    if not isinstance(spks, nap.TsGroup):
        raise TypeError(f"spks must be a pynapple.TsGroup, got {type(spks).__name__}.")
    if not isinstance(cc, pd.DataFrame):
        raise TypeError("cc must be a pandas.DataFrame of cross-correlogram counts.")
    if p_method not in ('poisson', 'empirical'):
        raise ValueError(f"p_method must be 'poisson' or 'empirical', got {p_method!r}.")
    if not np.isscalar(jitter_window) or jitter_window <= 0:
        raise ValueError(f"jitter_window must be a positive number, got {jitter_window}.")
    if int(n_surrogates) < 1:
        raise ValueError(f"n_surrogates must be >= 1, got {n_surrogates}.")
    if len(cc.index) < 2:
        raise ValueError("cc must have at least two lag bins.")
    for col in cc.columns:
        if not (isinstance(col, tuple) and len(col) == 2):
            raise ValueError(f"cc columns must be (ref, tgt) tuples, got {col!r}.")
        missing = [u for u in col if u not in spks.index]
        if missing:
            raise ValueError(f"Units {missing} from cc column {col} are not in spks.")

    lags = cc.index.to_numpy(dtype=float)
    n_lags = lags.size
    binsize = float(np.median(np.diff(lags)))
    half_width = (n_lags / 2.0) * binsize
    jitter_window = float(jitter_window)
    n_surrogates = int(n_surrogates)

    columns = list(cc.columns)
    child_seeds = np.random.SeedSequence(seed).spawn(len(columns))
    pair_seeds = [int(s.generate_state(1, dtype=np.uint32)[0]) for s in child_seeds]
    times = {}

    def unit_times(uid):
        if uid not in times:
            times[uid] = np.ascontiguousarray(_spike_times_array(spks[uid]))
        return times[uid]

    for col in columns:
        unit_times(col[0])
        unit_times(col[1])

    def run_pair(i):
        ref, tgt = columns[i]
        t_ref = times[ref]
        win_start = _jitter_window_starts(times[tgt], jitter_window)
        # Exact pruning: drop target windows that cannot reach any reference spike's CCG window.
        if t_ref.size and win_start.size:
            idx = np.searchsorted(t_ref, win_start - half_width, side='left')
            reachable = idx < t_ref.size
            reachable[reachable] = (t_ref[idx[reachable]]
                                    < win_start[reachable] + jitter_window + half_width)
            win_start = win_start[reachable]
        if win_start.size == 0 or t_ref.size == 0:
            return np.zeros((n_surrogates, n_lags))
        return _jitter_ccg_kernel(t_ref, np.ascontiguousarray(win_start), binsize,
                                  n_lags, jitter_window, n_surrogates, pair_seeds[i])

    n_workers = os.cpu_count() if n_jobs in (-1, None) else max(1, int(n_jobs))
    surrogates = [None] * len(columns)
    with tqdm(total=len(columns), desc='CCG jitter', leave=False) as pbar:
        if n_workers == 1:
            for i in range(len(columns)):
                surrogates[i] = run_pair(i)
                pbar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for i, surr in zip(range(len(columns)),
                                   pool.map(run_pair, range(len(columns)))):
                    surrogates[i] = surr
                    pbar.update(1)

    baseline = pd.DataFrame(np.nan, index=cc.index, columns=cc.columns)
    p_vals_ex = pd.DataFrame(np.nan, index=cc.index, columns=cc.columns)
    p_vals_in = pd.DataFrame(np.nan, index=cc.index, columns=cc.columns)
    for i, col in enumerate(columns):
        counts = cc[col].to_numpy(dtype=float)
        surr = surrogates[i]
        mu = surr.mean(axis=0)
        baseline[col] = mu
        if p_method == 'poisson':
            mu_safe = np.maximum(mu, 1e-10)
            p_vals_ex[col] = sp_poisson.sf(counts - 1, mu_safe)
            p_vals_in[col] = sp_poisson.cdf(counts, mu_safe)
        else:
            p_vals_ex[col] = (1 + (surr >= counts).sum(axis=0)) / (1 + n_surrogates)
            p_vals_in[col] = (1 + (surr <= counts).sum(axis=0)) / (1 + n_surrogates)

    return p_vals_ex, p_vals_in, baseline, baseline.copy()


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from mimo_pack.math.curvefitting import fit_sine_freq

    def simulate_and_test(mod_freqs, trans_probs, max_time=1000, dt=0.00003, f_rate=10):
        """Simulate spike trains with known modulation and test cross-correlogram correction.

        Generates synthetic pre/post-synaptic spike trains modulated at
        each frequency in *mod_freqs* with each transmission probability
        in *trans_probs*, computes cross-correlograms, and quantifies
        how well polynomial correction removes oscillatory artefacts.

        Parameters
        ----------
        mod_freqs : array-like of float
            Modulation frequencies (Hz) to test.
        trans_probs : array-like of float
            Monosynaptic transmission probabilities to test.
        max_time : float, optional
            Total simulation duration in seconds. Default is 1000.
        dt : float, optional
            Simulation time step in seconds. Default is 3×10⁻⁵.
        f_rate : float, optional
            Mean baseline firing rate in Hz. Default is 10.

        Returns
        -------
        None
            Results are printed and plotted inline.
        """
        results_attenuation = []
        results_peak_prob = []
        results_actual_prob = []

        for mod_freq in mod_freqs:
            for trans_prob in trans_probs:
                print(f"Simulating for modulation frequency {mod_freq} Hz and transmission probability {trans_prob}")
                # Generate spike trains
                t = np.arange(0, max_time, dt)
                rate1 = f_rate * dt * np.ones_like(t)
                spks1 = t[np.where(np.random.poisson(rate1) > 0)[0]]

                rate2 = f_rate * (1 + np.sin(2 * np.pi * mod_freq * t)) * dt
                spks2 = t[np.where(np.random.poisson(rate2) > 0)[0]]

                spks1_sub = np.random.choice(spks1, size=int(len(spks1) * trans_prob), replace=False)
                spks3 = np.sort(np.concatenate((spks2, spks1_sub + 0.002)))

                spks1_pos = spks1[np.where(np.mod(spks1, 1/mod_freq) < 1/(2*mod_freq))[0]]
                spks1_neg = spks1[np.where(np.mod(spks1, 1/mod_freq) >= 1/(2*mod_freq))[0]]

                # Convert to nap.Ts objects
                spks1_ts = nap.Ts(t=spks1, time_units='s')
                spks2_ts = nap.Ts(t=spks2, time_units='s')
                spks3_ts = nap.Ts(t=spks3, time_units='s')
                spks1_pos_ts = nap.Ts(t=spks1_pos, time_units='s')
                spks1_neg_ts = nap.Ts(t=spks1_neg, time_units='s')

                spks_grp = nap.TsGroup({0: spks1_ts, 1: spks1_pos_ts, 2: spks1_neg_ts, 3: spks3_ts})

                # Compute cross-correlogram
                cc = nap.compute_crosscorrelogram(spks_grp, binsize=0.0005, windowsize=0.02, norm=False)

                # Convert to probability and correct
                cc_prob = cc_to_prob(cc, norm=False)
                cc_corr, _ = cc_correct_poly(cc_prob, order=12)

                # Power at modulation frequency before and after correction
                lags = cc.index.values
                exclude_lag = 0.005  # Exclude lags within this range for polynomial fitting
                mask = np.abs(lags) > exclude_lag
                sin_orig, p_orig = fit_sine_freq(lags, lags[mask], cc_prob[(2,3)].values[mask], mod_freq)
                sin_corr, p_corr = fit_sine_freq(lags, lags[mask], cc_corr[(2,3)].values[mask], mod_freq)
                attenuation = p_corr['amplitude'] / p_orig['amplitude']
                results_attenuation.append((mod_freq, trans_prob, attenuation))

                # Peak transmission probability in corrected cc
                peak_prob = np.max(cc_corr[(2,3)].values)
                results_peak_prob.append((mod_freq, trans_prob, peak_prob))
                results_actual_prob.append(trans_prob)

        # Plot results
        results_attenuation = np.array(results_attenuation)
        results_peak_prob = np.array(results_peak_prob)
        results_actual_prob = np.array(results_actual_prob)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # Plot attenuation at modulation frequency
        for mod_freq in mod_freqs:
            mask = results_attenuation[:, 0] == mod_freq
            axs[0].plot(results_attenuation[mask, 1], results_attenuation[mask, 2], label=f"Mod Freq {mod_freq} Hz")
        axs[0].set_ylim(-0.1, 1.1)
        axs[0].set_xlabel("Transmission Probability")
        axs[0].set_ylabel("Attenuation at Modulation Frequency")
        axs[0].legend()
        axs[0].set_title("Attenuation at Modulation Frequency")

        # Plot peak transmission probability vs actual
        for mod_freq in mod_freqs:
            mask = results_peak_prob[:, 0] == mod_freq
            axs[1].plot(results_actual_prob[mask], results_peak_prob[mask, 2], label=f"Mod Freq {mod_freq} Hz")
        axs[1].plot([-1, 1], [-1, 1], 'k--')  # y=x line for reference
        axs[1].set_xlim(-0.01, 0.11)
        axs[1].set_ylim(-0.01, 0.11)
        axs[1].set_xlabel("Actual Transmission Probability")
        axs[1].set_ylabel("Peak Transmission Probability (Corrected CC)")
        axs[1].legend()
        axs[1].set_title("Peak Transmission Probability vs Actual")

        plt.tight_layout()
        plt.show()

    mod_freqs = [10, 16, 20, 50, 60, 80, 100]
    trans_probs = np.linspace(0, 0.1, 6)
    simulate_and_test(mod_freqs, trans_probs)
