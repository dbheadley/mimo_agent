# Functions for spectral analysis
# Author: Drew Headley
# Date: 2025-06-16

import warnings
from typing import Optional, Tuple, Union, Any
import pywt
import xarray as xr
import numpy as np
import pandas as pd
import scipy.signal as ss
import pynapple as nap
from mimo_pack.math.curvefitting import fit_exp_knee, exp_knee
from mimo_pack.consts import DIM_TIME, DIM_FREQ, DIM_CHAN, ATTR_FS, ATTR_UNITS

def stft(signal: xr.DataArray, window: float = 1.0, step: Optional[float] = None, **kwargs: Any) -> xr.DataArray:
    """
    Apply Short-Time Fourier Transform (STFT) to an xarray signal 
    and return an xarray with a frequency dimension.

    Parameters
    ----------
    signal : xarray.DataArray
        Must have a time dimension. STFT will be applied along 
        this dimension. Sampling frequency is extracted from the
        time coordinate.
    window : float
        Duration (in seconds) of each STFT window. Default is 1 second.
    step : float or None, optional
        Step size in seconds between successive windows. Defaults to
        ``None``, which sets the step equal to the window length (no overlap).
    kwargs : dict, optional
        Additional arguments to pass to scipy.signal.stft 
        (e.g., nperseg, noverlap, etc.).

    Returns
    -------
    xarray.DataArray
        STFT result with same non-time dimensions and an added 
        'frequency' dimension. Power spectral density with same units 
        as input signal squared.
        
    Raises
    ------
    TypeError
        If input is not an xarray.DataArray.
    ValueError
        If input does not have a 'time' dimension.
        
    Examples
    --------
    >>> from mimo_pack.testing.data import synthetic_lfp
    >>> lfp = synthetic_lfp(duration=10.0, fs=1000, n_chans=4)
    >>> spec = stft(lfp.isel(channel=0), window=1.0)
    >>> spec.dims
    ('time', 'frequency')
    >>> spec.shape
    (10, 501)
    """
    if not isinstance(signal, xr.DataArray):
        raise TypeError(
            f"Input must be an xarray.DataArray, got {type(signal).__name__}. "
            "Use fileio functions to load data into xarray format."
        )

    if DIM_TIME not in signal.dims:
        raise ValueError(
            f"Input xarray must have a '{DIM_TIME}' dimension. "
            f"Found dimensions: {signal.dims}. "
            "Ensure your signal has a time dimension."
        )
    
    # Ensure window is a scalar
    window = float(window)

    # Ensure step is a scalar or None
    if step is not None:
        step = float(step)

    # Extract sampling frequency from attributes or derive from time coordinate
    if ATTR_FS in signal.attrs:
        fs = float(signal.attrs[ATTR_FS])
    else:
        time = signal[DIM_TIME].values
        dt = np.nanmedian(np.diff(time))
        fs = 1.0 / dt

    arr = signal.to_numpy()
    dims = signal.dims
    time_axis = np.where(np.array(dims) == DIM_TIME)[0][0]
    
    hop = int(step * fs) if step is not None else int(window * fs)
    mfft = kwargs.get('mfft', int(window * fs))
    kwargs['mfft'] = mfft

    win_func = ss.windows.gaussian(mfft, std=mfft/5.0, sym=True)
    stft = ss.ShortTimeFFT(win_func, hop, fs, **kwargs)
    
    f = stft.f
    t = stft.t(arr.shape[time_axis])

    Zxx = stft.spectrogram(arr, axis=time_axis)
        
    # rearrange spectrogram array to have
    # original dimenions and with frequency last
    Zxx = Zxx.swapaxes(-1,time_axis)

    # Prepare output dims and coords
    dims = list(signal.dims)
    dims = dims + [DIM_FREQ]

    # Fix time coordinates and add frequency
    coords = dict(signal.coords)
    new_coords = {key: values for key, values in coords.items() 
                  if values.dims != (DIM_TIME,)}
    new_coords[DIM_FREQ] = f
    new_coords[DIM_TIME] = t
    new_coords['s0'] = (DIM_TIME, np.arange(Zxx.shape[time_axis]))

    # Propagate all attributes from input, then update transformation-specific keys
    attrs = signal.attrs.copy()
    attrs[ATTR_FS] = fs
    if ATTR_UNITS in signal.attrs:
        attrs[ATTR_UNITS] = f"({signal.attrs[ATTR_UNITS]})^2"
    else:
        attrs.pop(ATTR_UNITS, None)
    if step is not None and step != window:
        attrs['description'] = f"Short-time Fourier transform (window={window}s, step={step}s)"
    else:
        attrs['description'] = f"Short-time Fourier transform (window={window}s)"

    return xr.DataArray(Zxx, dims=dims, coords=new_coords, attrs=attrs)


def wavelet(signal: xr.DataArray, 
           freqs: Union[np.ndarray, list] = np.power(2, np.arange(0, 7, 0.25)), 
           wavelet_type: str = 'cmor2.5-1.0',
           remove_dc: bool = True, 
           time_dim: str = DIM_TIME, 
           verbose: bool = True) -> xr.DataArray:
    """
    Apply a wavelet transform to an xarray signal.

    Parameters
    ----------
    signal : xarray.DataArray
        Must have a time dimension. Sampling frequency is extracted
        from the time coordinate.
    freqs : np.ndarray or list, optional
        Frequencies to use for the wavelet transform. 
        Default is from 1 to 128 Hz on a logarithmic scale.
    wavelet_type : str, optional
        Wavelet type to use for the transform.
        Default is 'cmor2.5-1.0' (complex Morlet wavelet).
    remove_dc : bool, optional
        Whether to remove the DC offset from the signal before
        applying the wavelet transform. Default is True.
    time_dim : str, optional
        Name of the time dimension in the input xarray. Default is DIM_TIME.
    verbose : bool, optional
        Whether to print warnings about the size of the resulting
        wavelet transform. Default is True.

    Returns
    -------
    xarray.DataArray
        Wavelet result with same non-time dimensions and an 
        added 'frequency' dimension appended. Complex-valued output.
        
    Raises
    ------
    TypeError
        If input is not an xarray.DataArray.
    ValueError
        If input does not have the specified time dimension.
        
    Examples
    --------
    >>> from mimo_pack.testing.data import synthetic_lfp
    >>> lfp = synthetic_lfp(duration=10.0, fs=1000, n_chans=1)
    >>> freqs = np.power(2, np.arange(1, 7, 0.5))  # 2-64 Hz
    >>> wave = wavelet(lfp.isel(channel=0), freqs=freqs)
    >>> wave.dims
    ('time', 'frequency')
    """
    if not isinstance(signal, xr.DataArray):
        raise TypeError(
            f"Input must be an xarray.DataArray, got {type(signal).__name__}. "
            "Use fileio functions to load data into xarray format."
        )

    if time_dim not in signal.dims:
        raise ValueError(
            f"Input xarray must have a '{time_dim}' dimension. "
            f"Found dimensions: {signal.dims}. "
            "Ensure your signal has a time dimension."
        )
    
    if not isinstance(freqs, np.ndarray):
        freqs = np.array(freqs)

    # Calculate size of the resulting array from the wavelet transform
    # and give a warning if the size exceeds 20 GB
    est_size = (signal.size * freqs.size * 
                np.dtype(np.complex64).itemsize) / 1e9  # in GB
    if verbose and est_size > 20:
        warnings.warn(
            f"The resulting wavelet transform will be {est_size:.1f} GB. "
            "Consider reducing frequency resolution or signal length.",
            UserWarning
        )

    # Extract sampling frequency from attributes or derive from time coordinate
    time = signal[time_dim].values
    dt = np.nanmedian(np.diff(time))
    if ATTR_FS in signal.attrs:
        fs = float(signal.attrs[ATTR_FS])
    else:
        fs = 1.0 / dt

    arr = signal.to_numpy()
    dims = signal.dims

    time_axis = np.where(np.array(dims) == time_dim)[0][0]

    # Create a continuous wavelet transform object
    # We will use the complex Morlet wavelet
    w = pywt.ContinuousWavelet(wavelet_type)

    # convert frequencies to scales
    scales = pywt.frequency2scale(w, freqs) * fs 

    # remove DC offset to minimize edge effects
    if remove_dc:
        arr = arr - np.mean(arr, axis=time_axis, keepdims=True) 

    # Continuous wavelet transform
    Wxx, _ = pywt.cwt(arr, scales, w, sampling_period=dt, 
                      axis=time_axis, method='fft')
        
    # rearrange spectrogram array to have original organization,
    # and with frequency last
    Wxx = Wxx.transpose(np.roll(np.arange(Wxx.ndim), -1))

    # Prepare output dims and coords
    dims = list(signal.dims)
    dims = dims + [DIM_FREQ]

    # Fix time coordinates and add frequency
    new_coords = dict(signal.coords)
    new_coords[DIM_FREQ] = freqs

    # Propagate all attributes from input, then update transformation-specific keys
    attrs = signal.attrs.copy()
    attrs[ATTR_FS] = fs
    # Complex wavelet coefficients have no physical units from the input signal
    attrs.pop(ATTR_UNITS, None)
    attrs['description'] = f"Wavelet transform ({wavelet_type})"

    return xr.DataArray(Wxx, dims=dims, coords=new_coords, attrs=attrs)


def wavelet_phamp(wavelet: xr.DataArray) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate the phase and amplitude of a wavelet transform stored as an xarray.DataArray.

    Parameters
    ----------
    wavelet : xarray.DataArray
        Wavelet transform with a 'frequency' dimension and other dimensions to broadcast over.

    Returns
    -------
    amplitude_xr : xarray.DataArray
        Amplitude of the wavelet transform, with same shape as input.
        Preserves original signal units.
    phase_xr : xarray.DataArray
        Phase of the wavelet transform, with same shape as input.
        Units are radians.
        
    Raises
    ------
    ValueError
        If input does not have a 'frequency' dimension.
        
    Examples
    --------
    >>> from mimo_pack.testing.data import synthetic_lfp
    >>> lfp = synthetic_lfp(duration=1.0, fs=1000, n_chans=1)
    >>> wave = wavelet(lfp.isel(channel=0), freqs=[10, 20, 40])
    >>> amp, phase = wavelet_phamp(wave)
    >>> amp.shape == wave.shape
    True
    >>> phase.min() >= -np.pi and phase.max() <= np.pi
    True
    """

    # Check that 'frequency' dimension exists
    if DIM_FREQ not in wavelet.dims:
        raise ValueError(
            f"Input xarray must have a '{DIM_FREQ}' dimension. "
            f"Found dimensions: {wavelet.dims}. "
            "Use wavelet() to compute frequency decomposition first."
        )

    # Calculate phase and amplitude
    wave = wavelet.to_numpy()
    phase = np.angle(wave)
    amplitude = np.abs(wave)

    # Propagate all attributes from input, then update transformation-specific keys
    desc_base = wavelet.attrs.get('description', 'wavelet transform')

    amp_attrs = wavelet.attrs.copy()
    amp_attrs['description'] = f"{desc_base} -> wavelet amplitude"

    phase_attrs = wavelet.attrs.copy()
    phase_attrs['description'] = f"{desc_base} -> wavelet phase"
    phase_attrs[ATTR_UNITS] = 'radians'

    amplitude_xr = xr.DataArray(amplitude, dims=wavelet.dims, coords=wavelet.coords,
                               attrs=amp_attrs)
    phase_xr = xr.DataArray(phase, dims=wavelet.dims, coords=wavelet.coords,
                           attrs=phase_attrs)

    return amplitude_xr, phase_xr


def _fit_spectrum_aperiodic_np(freqs, spec, f_range=None, f_step=1, **kwargs):
    """
    Fit a model of the aperiodic component of the power spectrum (numpy version).
    
    This is an internal helper function. Use fit_spectrum_aperiodic() for xarray inputs.
    
    Parameters
    ----------
    freqs : np.ndarray
        Frequencies at which the power spectrum is evaluated.
    spec : np.ndarray
        Power spectrum values corresponding to the frequencies.
    f_range : [fmin, fmax], optional
        Frequency range to fit the aperiodic component.
        Default is None, which uses the full range.
    f_step : float, optional
        Frequency step size to interpolate the spectrum for fitting. 
        Default is 1 Hz.
    kwargs : dict, optional
        Additional keyword arguments for the fitting function.

    Returns
    -------
    fit : np.ndarray
        Fitted aperiodic component of the power spectrum.
    popt : np.ndarray
        Optimized parameters of the aperiodic fit.
    """

    orig_freqs = freqs
    if f_step is not None:
        # Interpolate the spectrum to a regular frequency grid for fitting
        f_min = freqs.min() 
        f_max = freqs.max()
        freqs_interp = np.arange(f_min, f_max + f_step, f_step)
        spec_interp = np.interp(freqs_interp, freqs, spec)
        freqs = freqs_interp
        spec = spec_interp

    if f_range is None:
        f_inds = np.arange(freqs.size)
    else:
        f_inds = np.where((freqs >= f_range[0]) & 
                          (freqs <= f_range[1]))[0]


    # Fit the aperiodic component
    _, popt = fit_exp_knee(freqs[f_inds], freqs[f_inds], 
                           np.log10(spec[f_inds]), **kwargs)

    fit = exp_knee(orig_freqs, *popt)
    fit = 10**fit  # Convert back to linear scale

    
    return fit, popt


def fit_spectrum_aperiodic(spec: xr.DataArray, 
                          f_range: Optional[Tuple[float, float]] = None, 
                          **kwargs: Any) -> xr.DataArray:
    """
    Calculates the aperiodic component of a power spectrogram 
    stored as an xarray.DataArray, fitting along the 'frequency' dimension and broadcasting over remaining dimensions.

    Parameters
    ----------
    spec : xarray.DataArray
        Power spectrogram with a 'frequency' dimension and other dimensions to broadcast over.
    f_range : tuple of (float, float), optional
        Frequency range (fmin, fmax) in Hz to fit the aperiodic component.
        Default is None, which uses the full range. 
    kwargs : dict, optional
        Additional keyword arguments for the fitting function.

    Returns
    -------
    spec_ap : xarray.DataArray
        Spectrogram with aperiodic component, with same shape as input.
        Preserves fs and units from input.
        
    Raises
    ------
    ValueError
        If input does not have a 'frequency' dimension.
        
    Examples
    --------
    >>> from mimo_pack.testing.data import synthetic_lfp
    >>> lfp = synthetic_lfp(duration=10.0, fs=1000, n_chans=1)
    >>> spec = stft(lfp.isel(channel=0), window=2.0)
    >>> aper = fit_spectrum_aperiodic(spec, f_range=(1, 100))
    >>> aper.shape == spec.shape
    True
    """

    # Check that 'frequency' dimension exists
    if DIM_FREQ not in spec.dims:
        raise ValueError(
            f"Input xarray must have a '{DIM_FREQ}' dimension. "
            f"Found dimensions: {spec.dims}. "
            "Compute power spectrum using stft() or similar first."
        )

    freqs = spec[DIM_FREQ].values

    # Define a function to apply along the 'frequency' dimension
    def _fit_func(x):
        """Apply aperiodic spectrum fit to a single frequency vector, returning NaN on failure."""
        try:
            fit, _ = _fit_spectrum_aperiodic_np(freqs, x, f_range=f_range, **kwargs)
        except Exception as e:
            warnings.warn(
                f"Error fitting spectrum: {e}. Returning NaN.",
                UserWarning
            )
            fit = np.full_like(x, np.nan)
        return fit
    
    # Apply along 'frequency', broadcasting over other dimensions
    spec_ap = xr.apply_ufunc(
        _fit_func,
        spec,
        input_core_dims=[[DIM_FREQ]],
        output_core_dims=[[DIM_FREQ]],
        vectorize=True,
        dask='parallelized'
    )
    
    # Propagate all attributes from input, then update description
    desc_base = spec.attrs.get('description', 'spectrum')
    spec_ap.attrs = spec.attrs.copy()
    spec_ap.attrs['description'] = f"{desc_base} -> aperiodic fit"

    return spec_ap

def compute_spike_phamp_hist(spks: nap.TsGroup, 
                            lfp: xr.DataArray, 
                            bins: Tuple[int, int] = (4, 9), 
                            freqs: Union[float, int, list, np.ndarray] = [52]) -> nap.TsGroup:
    """
    Compute 2D histograms of spike phase and amplitude percentiles for 
    each unit grouped by LFP channel, for one or more frequencies.
    
    Returns the input TsGroup with a new 'phamp_hist' metadata column
    containing xarray.DataArray histograms for each unit.

    Parameters
    ----------
    spks : pynapple.TsGroup
        Spike units object with 'lfp_channel' metadata field.
    lfp : xarray.DataArray
        LFP data with channel dimension.
    bins : tuple of int
        Number of bins for amplitude percentile and phase, e.g. (4, 9).
    freqs : float, int, or list/array
        Frequency or frequencies (Hz) to extract from wavelet. Default is [52].

    Returns
    -------
    pynapple.TsGroup
        Input TsGroup with added 'phamp_hist' metadata column containing
        xarray.DataArray histograms with dims ('amplitude', 'phase', 'frequency')
        for each unit.
        
    Raises
    ------
    TypeError
        If spks is not a pynapple.TsGroup or lfp is not an xarray.DataArray.
    ValueError
        If lfp does not have a channel dimension.
        
    Examples
    --------
    >>> import pynapple as nap
    >>> # Assuming spks and lfp are loaded
    >>> spks_with_hist = compute_spike_phamp_hist(spks, lfp, bins=(4, 9), freqs=[52, 78])
    >>> spks_with_hist.get_info('phamp_hist').iloc[0].shape
    (4, 9, 2)
    >>> # Access histogram for specific unit
    >>> hist_unit5 = spks_with_hist.get_info('phamp_hist').loc[5]
    """
    if not isinstance(spks, nap.TsGroup):
        raise TypeError(
            f"spks must be a pynapple.TsGroup, got {type(spks).__name__}. "
            "Load spike data using fileio functions."
        )
    
    if not isinstance(lfp, xr.DataArray):
        raise TypeError(
            f"lfp must be an xarray.DataArray, got {type(lfp).__name__}. "
            "Load LFP data using fileio functions."
        )
    
    if DIM_CHAN not in lfp.dims:
        raise ValueError(
            f"lfp must have a '{DIM_CHAN}' dimension. "
            f"Found dimensions: {lfp.dims}."
        )
    
    if not isinstance(freqs, (list, np.ndarray)):
        freqs = [freqs]
    freqs = np.array(freqs)

    spks_grp = spks.getby_category('lfp_channel')
    hist_grp = []
    for ch in spks_grp:
        spk_sel = spks_grp[ch]
        lfp_ch = lfp.sel({DIM_CHAN: ch})
        lfp_wave = wavelet(lfp_ch, freqs=freqs)
        t_dim = np.where(np.array(lfp_wave.dims) == DIM_TIME)[0][0]
        f_dim = np.where(np.array(lfp_wave.dims) == DIM_FREQ)[0][0]
        wave_amp, wave_ph = wavelet_phamp(lfp_wave)

        # convert wave_amp to percentiles along time for each frequency
        wave_amp_pr = wave_amp.copy(deep=True)
        # argsort twice to get ranks, then normalize to percentiles
        amp_np = wave_amp.to_numpy()
        amp_rank = np.apply_along_axis(lambda x: np.argsort(np.argsort(x)), t_dim, amp_np)
        amp_pr = amp_rank / (amp_np.shape[t_dim] - 1 + 1e-12)
        wave_amp_pr.data = amp_pr

        hist = []
        for spk_i in spk_sel:
            times_i = spk_sel[spk_i].times()
            # Select nearest time points for all frequencies
            amp_pr_sel = wave_amp_pr.sel({DIM_TIME: times_i}, method='nearest').to_numpy()
            ph_sel = wave_ph.sel({DIM_TIME: times_i}, method='nearest').to_numpy()
            # amp_pr_sel, ph_sel shape: (n_times, n_freqs) or (n_freqs, n_times) depending on dims
            # Ensure shape is (n_times, n_freqs)
            if amp_pr_sel.shape[0] != len(times_i):
                amp_pr_sel = amp_pr_sel.T
                ph_sel = ph_sel.T

            # Remove spikes with amplitude percentile > 0.999 for each frequency
            hists = []
            for fi in range(len(freqs)):
                amp_f = amp_pr_sel[:, fi]
                ph_f = ph_sel[:, fi]
                mask = amp_f <= 0.999
                amp_f = amp_f[mask]
                ph_f = ph_f[mask]
                h2d, amp_edges, ph_edges = np.histogram2d(
                    amp_f, ph_f, bins=bins, range=[[0, 1], [-np.pi, np.pi]]
                )
                hists.append(h2d)

            # Stack along frequency dimension
            hists = np.stack(hists, axis=-1)
            # Create xarray for this unit: dims (amplitude, phase, frequency)
            hist_xr = xr.DataArray(
                hists,
                dims=('amplitude', 'phase', DIM_FREQ),
                coords={
                    'amplitude': 0.5 * (amp_edges[:-1] + amp_edges[1:]),
                    'phase': 0.5 * (ph_edges[:-1] + ph_edges[1:]),
                    DIM_FREQ: freqs
                },
                name='count'
            )
            hist.append(hist_xr)

        hist_grp.extend(hist)

    # Create pandas Series indexed by unit with histogram xarrays
    hist_series = pd.Series(hist_grp, index=spks.keys(), dtype=object)
    
    # Add histogram as metadata column to TsGroup (pass as kwarg so column name is the string)
    spks.set_info(phamp_hist=hist_series)
    
    return spks


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mimo_pack.testing.data import synthetic_lfp
    
    print("=" * 60)
    print("SPECTRAL ANALYSIS MODULE TESTS")
    print("=" * 60)
    
    # Generate synthetic LFP data
    print("\n1. Generating synthetic LFP data...")
    duration = 10.0  # seconds
    fs = 1000  # Hz
    n_chans = 4
    
    lfp = synthetic_lfp(duration, fs, n_chans, oscillations=True, seed=42)
    print(f"   LFP shape: {lfp.shape}")
    print(f"   Dimensions: {lfp.dims}")
    print(f"   Sample rate: {lfp.attrs.get(ATTR_FS)} Hz")
    
    # Test 1: STFT
    print("\n2. Testing stft()...")
    try:
        spec = stft(lfp.isel({DIM_CHAN: 0}), window=1.0)
        print(f"   ✓ STFT successful")
        print(f"   Output shape: {spec.shape}")
        print(f"   Output dims: {spec.dims}")
        print(f"   Frequency range: {spec[DIM_FREQ].values.min():.1f} - {spec[DIM_FREQ].values.max():.1f} Hz")
        print(f"   Time points: {spec[DIM_TIME].size}")
    except Exception as e:
        print(f"   ✗ STFT failed: {e}")
    
    # Test 2: STFT on multi-channel data
    print("\n3. Testing stft() on multi-channel data...")
    try:
        spec_multi = stft(lfp, window=0.5)
        print(f"   ✓ Multi-channel STFT successful")
        print(f"   Output shape: {spec_multi.shape}")
        print(f"   Output dims: {spec_multi.dims}")
    except Exception as e:
        print(f"   ✗ Multi-channel STFT failed: {e}")
    
    # Test 3: Wavelet transform
    print("\n4. Testing wavelet()...")
    try:
        # Use single channel to keep memory reasonable
        freqs = np.power(2, np.arange(1, 7, 0.5))  # 2-64 Hz
        wave = wavelet(lfp.isel({DIM_CHAN: 0}), freqs=freqs, verbose=False)
        print(f"   ✓ Wavelet transform successful")
        print(f"   Output shape: {wave.shape}")
        print(f"   Output dims: {wave.dims}")
        print(f"   Frequencies: {freqs.min():.1f} - {freqs.max():.1f} Hz")
    except Exception as e:
        print(f"   ✗ Wavelet failed: {e}")
    
    # Test 4: Wavelet phase/amplitude
    print("\n5. Testing wavelet_phamp()...")
    try:
        amp, phase = wavelet_phamp(wave)
        print(f"   ✓ Phase/amplitude extraction successful")
        print(f"   Amplitude range: {amp.values.min():.3f} - {amp.values.max():.3f}")
        print(f"   Phase range: {phase.values.min():.3f} - {phase.values.max():.3f} rad")
    except Exception as e:
        print(f"   ✗ Phase/amplitude failed: {e}")
    
    # Test 5: Aperiodic fit
    print("\n6. Testing fit_spectrum_aperiodic()...")
    try:
        spec_single = stft(lfp.isel({DIM_CHAN: 0}), window=2.0)
        spec_mean = spec_single.mean(dim=DIM_TIME)
        
        # Fit aperiodic component
        aper_fit_mean = fit_spectrum_aperiodic(spec_mean, f_range=(1, 100))
        print(f"   ✓ Aperiodic fit successful")
        print(f"   Output shape: {aper_fit_mean.shape}")
    except Exception as e:
        print(f"   ✗ Aperiodic fit failed: {e}")
    
    # Test 6: Aperiodic fit on spectrogram
    print("\n7. Testing fit_spectrum_aperiodic() on spectrogram...")
    try:
        aper_xr = fit_spectrum_aperiodic(spec_single, f_range=(1, 100))
        print(f"   ✓ Aperiodic spectrogram fit successful")
        print(f"   Output shape: {aper_xr.shape}")
    except Exception as e:
        print(f"   ✗ Aperiodic spectrogram fit failed: {e}")
    
    # Test 7: Error handling
    print("\n8. Testing error handling...")
    try:
        stft(np.array([1, 2, 3]))  # Should fail - not xarray
        print(f"   ✗ Should have raised TypeError")
    except TypeError as e:
        print(f"   ✓ Correctly raised TypeError for non-xarray input")
    
    try:
        bad_xr = xr.DataArray([1, 2, 3], dims=['x'])
        stft(bad_xr)  # Should fail - no time dimension
        print(f"   ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError for missing time dimension")
    
    # Visualization
    print("\n9. Creating visualization...")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Raw LFP
        ax = axes[0, 0]
        t = lfp[DIM_TIME].values
        ax.plot(t[:2000], lfp.isel({DIM_CHAN: 0}).values[:2000], 'b-', lw=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Synthetic LFP (first 2 seconds)')
        
        # Plot 2: Spectrogram
        ax = axes[0, 1]
        spec_plot = spec_single.sel({DIM_FREQ: slice(0, 100)})
        im = ax.pcolormesh(spec_plot[DIM_TIME].values, 
                          spec_plot[DIM_FREQ].values,
                          10 * np.log10(spec_plot.values.T + 1e-10),
                          shading='auto', cmap='viridis')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('STFT Spectrogram (dB)')
        plt.colorbar(im, ax=ax, label='Power (dB)')
        
        # Plot 3: Mean spectrum with aperiodic fit
        ax = axes[1, 0]
        freqs_plot = spec_mean[DIM_FREQ].values
        ax.semilogy(freqs_plot, spec_mean.values, 'b-', label='Spectrum')
        ax.semilogy(freqs_plot, aper_fit_mean.values, 'r--', label='Aperiodic fit')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_title('Mean Spectrum with Aperiodic Fit')
        ax.legend()
        ax.set_xlim(0, 100)
        
        # Plot 4: Wavelet amplitude
        ax = axes[1, 1]
        wave_plot = amp.sel({DIM_FREQ: slice(2, 64)})
        im = ax.pcolormesh(wave_plot[DIM_TIME].values,
                          wave_plot[DIM_FREQ].values,
                          wave_plot.values.T,
                          shading='auto', cmap='hot')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('Wavelet Amplitude')
        ax.set_yscale('log')
        plt.colorbar(im, ax=ax, label='Amplitude')
        
        plt.tight_layout()
        plt.savefig('spectral_tests.png', dpi=150)
        # plt.show()  # Comment out to avoid blocking in automated tests
        print(f"   ✓ Visualization saved to spectral_tests.png")
    except Exception as e:
        print(f"   ✗ Visualization failed: {e}")
    
    print("\n" + "=" * 60)
    print("SPECTRAL ANALYSIS TESTS COMPLETE")
    print("=" * 60)
