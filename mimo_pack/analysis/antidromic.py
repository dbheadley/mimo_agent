"""Antidromic activation detection for Neuropixels spike data.

Thin antidromic-specific wrapper around the application-agnostic point
process Gaussian mixture model fit in
:mod:`mimo_pack.analysis.point_process`. Detects short-latency, low-jitter
antidromic activation peaks locked to electrical stimulation pulses and
computes per-component spike probability statistics, including a
collision-corrected probability derived from the classic antidromic
collision test.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union
import warnings

import numpy as np

from mimo_pack.analysis.point_process import fit_point_process_gmm

__all__ = ["GaussianComponent", "AntidromicFitResult", "fit_antidromic_gaussians"]


@dataclass
class GaussianComponent:
    """One Gaussian component from an antidromic mixture fit.

    Attributes
    ----------
    mu_s : float
        Mean latency in seconds.
    sigma_s : float
        Standard deviation in seconds.
    weight : float
        Mixing proportion of this component in the fitted mixture.
    amplitude : float
        Peak spike density in spikes/stim/s.
        Equals ``rate * weight / (sigma * sqrt(2*pi))``.
    raw_prob : float
        Mean spikes per stim falling in [mu - 2*sigma, mu + 2*sigma].
    corrected_prob : float
        Fraction of stims with a response spike in [mu - 2*sigma, mu + 2*sigma],
        plus the fraction of stims with a collision-window spike in
        [-mu, 0) *and no* response spike (a collision-window spike only
        explains a missing response; it is not added when a response spike
        is also present). See ``n_collision_violations``.
    n_collision_violations : int
        Number of stims where a spike was present in *both* the
        collision window [-mu, 0) and the response window
        [mu - 2*sigma, mu + 2*sigma]. This violates the antidromic collision
        principle (a colliding spike should block the antidromic response)
        and should be reviewed; it is flagged here and via a warning.
    """

    mu_s: float
    sigma_s: float
    weight: float
    amplitude: float
    raw_prob: float
    corrected_prob: float
    n_collision_violations: int


@dataclass
class AntidromicFitResult:
    """Gaussian mixture fit result for one unit / one stim-group pair.

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
        Total pooled relative spike events across all stims.
    n_stims : int
        Number of stimulation pulses.
    window_s : float
        Post-stimulus window used in seconds (the ``post_s`` half of the
        underlying peri-event window; see
        :func:`mimo_pack.analysis.point_process.fit_point_process_gmm`).
    rel_spike_times : numpy.ndarray
        All relative spike times in (0, window_s].
    """

    components: List[GaussianComponent]
    offset: float
    n_gaussians: int
    bic: float
    n_spikes: int
    n_stims: int
    window_s: float
    rel_spike_times: np.ndarray


def fit_antidromic_gaussians(
    spike_times,
    stim_times,
    window_s: Union[float, Sequence[float]] = 0.030,
    max_gaussians: int = 4,
    min_spikes: int = 10,
    n_init: int = 5,
    reg_covar: float = 1e-8,
    random_state: int = 0,
) -> Optional[AntidromicFitResult]:
    """Detect antidromic activation via a Gaussian mixture model fit to spike latencies.

    Thin antidromic-specific wrapper around
    :func:`mimo_pack.analysis.point_process.fit_point_process_gmm`, using
    stimulus pulses as the reference events. On top of the generic fit, this
    function computes a collision-corrected probability for each component
    using the classic antidromic collision test: for each stim, a spike in
    the pre-stimulus collision window [-mu, 0) only explains the *absence*
    of a response spike in [mu - 2*sigma, mu + 2*sigma]; it is not counted
    as additional evidence when a response spike is also present for that
    stim. Stims where both are present violate the collision principle and
    are flagged via ``GaussianComponent.n_collision_violations``. 
    See :func:`fit_point_process_gmm` for the
    underlying fitting algorithm.

    Parameters
    ----------
    spike_times : array-like or pynapple.Ts
        Absolute spike times in seconds for a single unit. If a pynapple Ts
        object is passed, its ``.t`` attribute is used.
    stim_times : array-like or pynapple.Ts
        Stimulus onset times in seconds. If a pynapple Ts object is passed,
        its ``.t`` attribute is used.
    window_s : float or array-like of 2 floats, optional
        Peri-stimulus window for collecting relative spike times, in
        seconds. A scalar gives a symmetric window (the same duration
        before and after each stim). A two-element array-like
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
    AntidromicFitResult or None
        ``None`` only if ``stim_times`` is empty.  If fewer than
        ``min_spikes`` relative spike events were found, returns an
        :class:`AntidromicFitResult` with ``n_gaussians=0`` and an empty
        ``components`` list, but ``rel_spike_times`` still contains the
        collected latencies.  Otherwise returns a fully fitted
        :class:`AntidromicFitResult` with ``.components`` sorted by
        ascending latency.

    Examples
    --------
    >>> spike_t = np.random.uniform(0, 3600, 5000)
    >>> stim_t  = np.arange(0, 3600, 1.0)
    >>> result  = fit_antidromic_gaussians(spike_t, stim_t)
    """
    result = fit_point_process_gmm(
        spike_times,
        stim_times,
        window_s=window_s,
        max_gaussians=max_gaussians,
        min_spikes=min_spikes,
        n_init=n_init,
        reg_covar=reg_covar,
        random_state=random_state,
    )
    if result is None:
        return None

    n_stims = result.n_events
    parts   = result.rel_spike_time_parts

    components: List[GaussianComponent] = []
    for comp in result.components:
        mu    = comp.mu_s
        sigma = comp.sigma_s
        lo    = mu - 2.0 * sigma
        hi    = mu + 2.0 * sigma

        n_response      = 0
        n_collision_only = 0
        n_violations    = 0
        for part in parts:
            has_response  = bool(np.any((part >= lo) & (part <= hi)))
            has_collision = bool(np.any((part >= -hi) & (part < lo)))
            if has_response:
                n_response += 1
                if has_collision:
                    n_violations += 1
            elif has_collision:
                n_collision_only += 1


        corrected_prob = (n_response + n_collision_only) / n_stims

        components.append(GaussianComponent(
            mu_s=mu,
            sigma_s=sigma,
            weight=comp.weight,
            amplitude=comp.amplitude,
            raw_prob=comp.raw_prob,
            corrected_prob=corrected_prob,
            n_collision_violations=n_violations,
        ))

    return AntidromicFitResult(
        components=components,
        offset=result.offset,
        n_gaussians=result.n_gaussians,
        bic=result.bic,
        n_spikes=result.n_spikes,
        n_stims=n_stims,
        window_s=result.window_s[1],
        rel_spike_times=result.rel_spike_times,
    )
