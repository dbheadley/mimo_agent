# Functions for cross-correlation analyses
# Author: Drew Headley
# Date: 2025-07-11

import numpy as np
import pandas as pd
import pynapple as nap
from scipy.ndimage import label
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

def _significant_run_peaks(signif, deviation, min_len, max_len):
    """
    Find runs of consecutive significant bins and the peak bin of each run.

    Parameters
    ----------
    signif : numpy.ndarray
        Boolean array flagging significant bins.
    deviation : numpy.ndarray
        Signed deviation from baseline (count - baseline); the peak of a run
        is the bin with the largest absolute deviation.
    min_len, max_len : int
        Inclusive bounds on the run length (in bins).

    Returns
    -------
    list of tuple
        (peak_index, absolute_deviation) for each run within the length bounds.
    """
    labels, n_runs = label(signif)
    peaks = []
    for run_id in range(1, n_runs + 1):
        idx = np.flatnonzero(labels == run_id)
        if not min_len <= idx.size <= max_len:
            continue
        peak = idx[np.argmax(np.abs(deviation[idx]))]
        peaks.append((peak, abs(deviation[peak])))
    return peaks

def find_significant_connections(p_vals_ex, p_vals_in, baseline_ex, baseline_in, cc,
                                  lag_min_abs, lag_max_abs,
                                  min_contiguous_bins=2, alpha=0.05,
                                  max_contiguous_bins=10):
    """
    Return significant monosynaptic connections from CCG p-value DataFrames.

    For each pair column, runs of consecutive significant bins (same sign,
    Bonferroni-corrected threshold of alpha over the number of lag bins) are
    identified across the full CCG. A run is kept if its length is between
    min_contiguous_bins and max_contiguous_bins. The peak of a run is the bin
    with the largest absolute deviation of the spike count from baseline. A
    run is accepted only if its peak lies within the positive or negative lag
    window; the remaining bins of the run may lie outside the windows. Within
    each window, the accepted run with the largest absolute deviation is
    reported, yielding up to one connection per ordered pair direction.

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
        Minimum number of consecutive significant bins in a run, counting
        bins outside the lag windows. Default is 2.
    alpha : float, optional
        Per-pair significance level; Bonferroni correction over lag bins is
        applied internally. Default is 0.05.
    max_contiguous_bins : int, optional
        Maximum number of consecutive significant bins in a run; longer runs
        are treated as broad co-modulation and rejected. Default is 10.

    Returns
    -------
    list of dict
        One dict per detected connection with keys:

        direction : tuple
            (pre_unit_id, post_unit_id).
        lag_s : float
            Absolute transmission delay in seconds.
        sign : int
            +1 (excitatory) or -1 (inhibitory).
        amplitude : float
            (count - baseline) / baseline at the peak bin.
    """
    if not isinstance(cc, pd.DataFrame):
        raise TypeError("cc must be a pandas.DataFrame of cross-correlogram counts.")

    lags       = cc.index.values
    pos_window = (lags >  lag_min_abs) & (lags <= lag_max_abs)
    neg_window = (lags >= -lag_max_abs) & (lags < -lag_min_abs)
    alpha_corr = alpha / len(cc)  # Bonferroni over lag bins

    connections = []
    for col in tqdm(cc.columns, desc='Finding connections', leave=False):
        unit_i, unit_j = col
        counts     = cc[col].values.astype(float)
        bl_ex      = baseline_ex[col].values.astype(float)
        bl_in      = baseline_in[col].values.astype(float)
        excess     = counts - bl_ex
        undershoot = counts - bl_in

        signif_ex = (p_vals_ex[col].values < alpha_corr) & (excess > 0) & (bl_ex > 0)
        signif_in = (p_vals_in[col].values < alpha_corr) & (undershoot < 0)

        # (peak index, absolute deviation, sign, baseline at peak)
        candidates = []
        for signif, dev, sign, bl in [(signif_ex, excess, 1, bl_ex),
                                      (signif_in, undershoot, -1, bl_in)]:
            for peak, mag in _significant_run_peaks(
                    signif, dev, min_contiguous_bins, max_contiguous_bins):
                candidates.append((peak, mag, sign, bl[peak]))

        for window_mask, direction in [
            (pos_window, (unit_i, unit_j)),
            (neg_window, (unit_j, unit_i)),
        ]:
            in_win = [c for c in candidates if window_mask[c[0]]]
            if not in_win:
                continue
            sel, _, sign, bl_peak = max(in_win, key=lambda c: c[1])
            connections.append({
                'direction': direction,
                'lag_s':     abs(lags[sel]),
                'sign':      sign,
                'amplitude': (counts[sel] - bl_peak) / bl_peak if bl_peak > 0 else np.nan,
            })

    return connections
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
