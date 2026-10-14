"""Gaussian mixture model fitting for point process (spike train) data.

Fits a Gaussian Mixture Model directly to pooled event-relative spike times
to detect latency components (e.g., short-latency, low-jitter response
peaks) relative to a set of reference event times, and computes
per-component spike probability statistics.

Spike times are collected in a single window around each reference event
(optionally asymmetric) and are not separated into "before" and "after"
groups; the mixture is fit to all of them together. Application-specific
callers can inspect the sign of ``rel_spike_times`` themselves (negative
values precede the reference event) if that distinction matters for their
domain, and can use the per-event ``rel_spike_time_parts`` to reason about
individual trials (e.g. collision corrections).

This module is application-agnostic: it operates on any spike train and any
set of reference (trigger) event times. Application-specific wrappers, such
as antidromic activation detection, live alongside their domain logic (see
:mod:`mimo_pack.analysis.antidromic`).
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union
import warnings

import numpy as np
from sklearn.mixture import GaussianMixture


def _resolve_window(window_s: Union[float, Sequence[float]]) -> Tuple[float, float]:
    """Normalize ``window_s`` to an explicit ``(pre_s, post_s)`` pair.

    A scalar is treated as a symmetric window (the same duration before and
    after the reference event). A two-element array-like is treated as
    ``(pre_s, post_s)``, an asymmetric window.
    """
    window_arr = np.asarray(window_s, dtype=float)
    if window_arr.ndim == 0:
        pre_s = post_s = float(window_arr)
    elif window_arr.shape == (2,):
        pre_s, post_s = float(window_arr[0]), float(window_arr[1])
    else:
        raise ValueError(
            "window_s must be a scalar or a two-element array-like "
            f"(pre_s, post_s); got shape {window_arr.shape}."
        )
    return pre_s, post_s


@dataclass
class GaussianComponent:
    """One Gaussian component from a point process mixture fit.

    Attributes
    ----------
    mu_s : float
        Mean latency relative to the reference event, in seconds.
    sigma_s : float
        Standard deviation in seconds.
    weight : float
        Mixing proportion of this component in the fitted mixture.
    amplitude : float
        Peak spike density in spikes/event/s.
        Equals ``rate * weight / (sigma * sqrt(2*pi))``.
    raw_prob : float
        Mean spikes per reference event falling in [mu - 2*sigma, mu + 2*sigma].
    """

    mu_s: float
    sigma_s: float
    weight: float
    amplitude: float
    raw_prob: float


@dataclass
class PointProcessGMMFitResult:
    """Gaussian mixture fit result for one spike train / reference-event set.

    Attributes
    ----------
    components : list of GaussianComponent
        Fitted components sorted ascending by mu_s.
    offset : float
        Always 0.0 for the pure GMM (retained for compatibility).
    n_gaussians : int
        Number of Gaussian components in the selected model.
    bic : float
        Bayesian Information Criterion of the selected model.
    n_spikes : int
        Total pooled relative spike events across all reference events.
    n_events : int
        Number of reference events.
    window_s : tuple of (float, float)
        ``(pre_s, post_s)`` durations of the peri-event window used, in
        seconds. Symmetric windows have ``pre_s == post_s``.
    rel_spike_times : numpy.ndarray
        All relative spike times in (-pre_s, post_s], pooled across
        reference events and not separated by sign. Negative values
        precede the reference event; positive values follow it.
    rel_spike_time_parts : list of numpy.ndarray
        Relative spike times for each reference event individually, in the
        same order as the input ``ref_times``. ``rel_spike_times`` is the
        concatenation of this list. Exposed so application-specific callers
        can reason about individual trials (e.g. whether a given reference
        event has a spike in some sub-window) without recollecting spike
        times.
    """

    components: List[GaussianComponent]
    offset: float
    n_gaussians: int
    bic: float
    n_spikes: int
    n_events: int
    window_s: Tuple[float, float]
    rel_spike_times: np.ndarray
    rel_spike_time_parts: List[np.ndarray]


def fit_point_process_gmm(
    spike_times,
    ref_times,
    window_s: Union[float, Sequence[float]] = 0.030,
    max_gaussians: int = 4,
    min_spikes: int = 10,
    n_init: int = 5,
    reg_covar: float = 1e-8,
    random_state: int = 0,
) -> Optional[PointProcessGMMFitResult]:
    """Fit a Gaussian mixture model to event-relative spike latencies.

    Fits a Gaussian Mixture Model directly to pooled peri-event relative
    spike times (unbinned), selects the number of components by BIC, and
    computes a raw per-component spike probability statistic. Spike times
    before and after each reference event are pooled together and fit as a
    single mixture; they are not separated into distinct groups. Callers
    that care about the distinction can filter ``rel_spike_times`` by sign
    (negative values precede the reference event), or use
    ``rel_spike_time_parts`` for per-trial reasoning, without recollecting
    spike times.

    This is a general point-process latency analysis: ``ref_times`` may be
    stimulus onsets, behavioral events, or any other set of reference times
    against which ``spike_times`` should be aligned.

    Algorithm
    ---------
    1. For each reference event, collect relative spike times in
       (-pre_s, post_s].
    2. Fit 1-max_gaussians Gaussian mixtures directly to all of the pooled
       relative spike times (unbinned, sklearn EM); select the number of
       components by BIC.
    3. For each mixture component, compute raw_prob: the mean spikes per
       reference event falling within mu +/- 2*sigma.

    Parameters
    ----------
    spike_times : array-like or pynapple.Ts
        Absolute spike times in seconds for a single unit. If a pynapple Ts
        object is passed, its ``.t`` attribute is used.
    ref_times : array-like or pynapple.Ts
        Reference (trigger) event times in seconds. If a pynapple Ts object
        is passed, its ``.t`` attribute is used.
    window_s : float or array-like of 2 floats, optional
        Peri-event window for collecting relative spike times, in seconds.
        A scalar gives a symmetric window (the same duration before and
        after each reference event). A two-element array-like
        ``(pre_s, post_s)`` gives an asymmetric window. Default 0.030.
    max_gaussians : int, optional
        Maximum number of Gaussian components to consider. Default 4.
    min_spikes : int, optional
        Minimum pooled relative spikes required to attempt fitting. Default 10.
    n_init : int, optional
        Number of EM initializations per mixture, passed to
        ``GaussianMixture``. Default 5.
    reg_covar : float, optional
        Non-negative regularization added to the diagonal of the covariance,
        passed to ``GaussianMixture``. Default 1e-8.
    random_state : int, optional
        Random seed for ``GaussianMixture`` initialization. Default 0.

    Returns
    -------
    PointProcessGMMFitResult or None
        ``None`` only if ``ref_times`` is empty. If fewer than
        ``min_spikes`` relative spike events were found, returns a
        :class:`PointProcessGMMFitResult` with ``n_gaussians=0`` and an
        empty ``components`` list, but ``rel_spike_times`` still contains
        the collected latencies. Otherwise returns a fully fitted
        :class:`PointProcessGMMFitResult` with ``.components`` sorted by
        ascending latency.

    Examples
    --------
    >>> spike_t = np.random.uniform(0, 3600, 5000)
    >>> ref_t   = np.arange(0, 3600, 1.0)
    >>> result  = fit_point_process_gmm(spike_t, ref_t)
    """
    # Accept pynapple Ts objects
    if hasattr(spike_times, 't'):
        spike_times = spike_times.t
    if hasattr(ref_times, 't'):
        ref_times = ref_times.t

    spike_times = np.asarray(spike_times, dtype=float)
    ref_times   = np.asarray(ref_times,   dtype=float)
    n_events    = len(ref_times)
    if n_events == 0:
        return None

    pre_s, post_s = _resolve_window(window_s)

    # -- Step 1: collect relative spike times ---------------------------------
    parts: List[np.ndarray] = []
    for t0 in ref_times:
        mask = (spike_times > t0 - pre_s) & (spike_times <= t0 + post_s)
        parts.append(spike_times[mask] - t0)

    rel_times = np.concatenate(parts)

    if len(rel_times) < min_spikes:
        return PointProcessGMMFitResult(
            components=[],
            offset=0.0,
            n_gaussians=0,
            bic=np.nan,
            n_spikes=len(rel_times),
            n_events=n_events,
            window_s=(pre_s, post_s),
            rel_spike_times=rel_times,
            rel_spike_time_parts=parts,
        )

    # -- Step 2: fit 1..max_gaussians GMMs directly to spike times, select by BIC --
    X        = rel_times.reshape(-1, 1)
    rate     = len(rel_times) / n_events   # mean spikes per event in window
    best_bic = np.inf
    best_gmm = None
    best_n   = 1

    for n_gauss in range(1, max_gaussians + 1):
        if n_gauss > len(rel_times):
            break
        gmm = GaussianMixture(
            n_components=n_gauss,
            covariance_type='full',
            n_init=n_init,
            reg_covar=reg_covar,
            random_state=random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                gmm.fit(X)
            except Exception:
                continue
        bic = gmm.bic(X)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_n   = n_gauss

    if best_gmm is None:
        return None

    # -- Step 3: per-component probabilities ----------------------------------
    components: List[GaussianComponent] = []
    for k in range(best_n):
        mu     = float(best_gmm.means_[k, 0])
        sigma  = float(np.sqrt(best_gmm.covariances_[k, 0, 0]))
        weight = float(best_gmm.weights_[k])

        # Peak spike density: rate * weight / (sigma * sqrt(2pi))
        amplitude = rate * weight / (sigma * np.sqrt(2.0 * np.pi))

        lo = mu - 2.0 * sigma
        hi = mu + 2.0 * sigma

        # Raw probability: mean spikes per event in [lo, hi]
        raw_prob = float(np.sum((rel_times >= lo) & (rel_times <= hi))) / n_events

        components.append(GaussianComponent(
            mu_s=mu,
            sigma_s=sigma,
            weight=weight,
            amplitude=amplitude,
            raw_prob=raw_prob,
        ))

    components.sort(key=lambda c: c.mu_s)

    return PointProcessGMMFitResult(
        components=components,
        offset=0.0,
        n_gaussians=best_n,
        bic=float(best_bic),
        n_spikes=len(rel_times),
        n_events=n_events,
        window_s=(pre_s, post_s),
        rel_spike_times=rel_times,
        rel_spike_time_parts=parts,
    )
