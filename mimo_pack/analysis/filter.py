# Functions for filtering
# Author: Drew Headley
# Date: 2025-07-22

import xarray as xr
from scipy.signal import butter, filtfilt, iirnotch, lfilter

from mimo_pack.consts import DIM_TIME, ATTR_FS

def lp_filter(signal_xr, cutoff_freq=10, order=4, causal=False):
    """
    Applies a low-pass Butterworth filter to the input signal.

    Parameters
    ----------
    signal_xr : xr.DataArray
        The input signal to be filtered.
    cutoff_freq : float
        The cutoff frequency for the low-pass filter in Hz.
    order : int, optional
        The order of the Butterworth filter. Default is 4.
    causal : bool, optional
        If False (default), applies a zero-phase acausal filter using
        ``scipy.signal.filtfilt``.  If True, applies a causal
        forward-only filter using ``scipy.signal.lfilter``.

    Returns
    -------
    xr.DataArray
        The filtered signal.
        
    Raises
    ------
    TypeError
        If input is not an xarray.DataArray.
    ValueError
        If input does not have a 'time' dimension.
    """
    if not isinstance(signal_xr, xr.DataArray):
        raise TypeError(
            f"Input must be an xarray.DataArray, got {type(signal_xr).__name__}. "
            "Use fileio functions to load data into xarray format."
        )
        
    if DIM_TIME not in signal_xr.dims:
        raise ValueError(f"Input xarray must have a '{DIM_TIME}' dimension.")

    # if signal_xr doesn't have sample_rate attribute, calculate from time dimension
    if ATTR_FS not in signal_xr.attrs:
        s_rate = 1/signal_xr[DIM_TIME].diff(DIM_TIME).mean().item()
    else:
        s_rate = signal_xr.attrs.get(ATTR_FS)
        
    nyquist = 0.5 * s_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    dims = signal_xr.dims
    time_idx = dims.index(DIM_TIME)
    
    if causal:
        signal_filt = lfilter(b, a, signal_xr.to_numpy(), axis=time_idx)
    else:
        signal_filt = filtfilt(b, a, signal_xr.to_numpy(), axis=time_idx)

    # Add description for data provenance
    causality = 'causal' if causal else 'acausal'
    attrs = signal_xr.attrs.copy()
    prev_desc = attrs.get('description', '').strip()
    attrs['description'] = prev_desc + (' -> ' if prev_desc else '') + f'Low-pass filtered ({cutoff_freq} Hz, {causality})'

    return xr.DataArray(
        signal_filt,
        dims=dims,
        coords=signal_xr.coords,
        attrs=attrs
    )

def hp_filter(signal_xr, cutoff_freq=0.02, order=4, causal=False):
    """
    Applies a high-pass Butterworth filter to the input signal.

    Parameters
    ----------
    signal_xr : xr.DataArray
        The input signal to be filtered.
    cutoff_freq : float
        The cutoff frequency for the high-pass filter in Hz.
    order : int, optional
        The order of the Butterworth filter. Default is 4.
    causal : bool, optional
        If False (default), applies a zero-phase acausal filter using
        ``scipy.signal.filtfilt``.  If True, applies a causal
        forward-only filter using ``scipy.signal.lfilter``.

    Returns
    -------
    xr.DataArray
        The filtered signal.
    """
    s_rate = signal_xr.attrs.get(ATTR_FS, 1.0)
    nyquist = 0.5 * s_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    
    dims = signal_xr.dims
    time_idx = dims.index(DIM_TIME)
    
    if causal:
        signal_filt = lfilter(b, a, signal_xr.to_numpy(), axis=time_idx)
    else:
        signal_filt = filtfilt(b, a, signal_xr.to_numpy(), axis=time_idx)

    # Add description for data provenance
    causality = 'causal' if causal else 'acausal'
    attrs = signal_xr.attrs.copy()
    prev_desc = attrs.get('description', '').strip()
    attrs['description'] = prev_desc + (' -> ' if prev_desc else '') + f'High-pass filtered ({cutoff_freq} Hz, {causality})'

    return xr.DataArray(
        signal_filt,
        dims=dims,
        coords=signal_xr.coords,
        attrs=attrs
    )


def notch_filter(signal_xr, notch_freq=60.0, quality=30.0, causal=False):
    """Apply a notch (band-reject) filter at the specified frequency.

    Uses a second-order IIR notch filter designed with
    ``scipy.signal.iirnotch``.

    Parameters
    ----------
    signal_xr : xr.DataArray
        The input signal to be filtered.  Must have a ``'time'`` dimension.
    notch_freq : float, optional
        The center frequency to reject, in Hz.  Default is 60.0 (US mains).
    quality : float, optional
        Quality factor that controls the bandwidth of the notch.
        ``bandwidth = notch_freq / quality``.  Higher values produce a
        narrower notch.  Default is 30.0.
    causal : bool, optional
        If False (default), applies a zero-phase acausal filter using
        ``scipy.signal.filtfilt``.  If True, applies a causal
        forward-only filter using ``scipy.signal.lfilter``.

    Returns
    -------
    xr.DataArray
        The filtered signal with the same dimensions, coordinates, and
        attributes as the input (plus updated ``description``).

    Raises
    ------
    TypeError
        If input is not an xarray.DataArray.
    ValueError
        If input does not have a ``'time'`` dimension.

    Examples
    --------
    >>> from mimo_pack.testing import synthetic_lfp
    >>> lfp = synthetic_lfp(duration=2.0, fs=1000, n_chans=4)
    >>> lfp_clean = notch_filter(lfp, notch_freq=60.0)
    """
    if not isinstance(signal_xr, xr.DataArray):
        raise TypeError(
            f"Input must be an xarray.DataArray, got {type(signal_xr).__name__}. "
            "Use fileio functions to load data into xarray format."
        )

    if DIM_TIME not in signal_xr.dims:
        raise ValueError(f"Input xarray must have a '{DIM_TIME}' dimension.")

    # Determine sample rate
    if ATTR_FS not in signal_xr.attrs:
        s_rate = 1 / signal_xr[DIM_TIME].diff(DIM_TIME).mean().item()
    else:
        s_rate = float(signal_xr.attrs[ATTR_FS])

    b, a = iirnotch(notch_freq, quality, fs=s_rate)

    dims = signal_xr.dims
    time_idx = dims.index(DIM_TIME)

    if causal:
        signal_filt = lfilter(b, a, signal_xr.to_numpy(), axis=time_idx)
    else:
        signal_filt = filtfilt(b, a, signal_xr.to_numpy(), axis=time_idx)

    # Add description for data provenance
    causality = 'causal' if causal else 'acausal'
    attrs = signal_xr.attrs.copy()
    prev_desc = attrs.get('description', '').strip()
    attrs['description'] = (
        prev_desc + (' -> ' if prev_desc else '')
        + f'Notch filtered ({notch_freq} Hz, Q={quality}, {causality})'
    )

    return xr.DataArray(
        signal_filt,
        dims=dims,
        coords=signal_xr.coords,
        attrs=attrs
    )


if __name__ == "__main__":
    import numpy as np
    from mimo_pack.consts import DIM_CHAN, DIM_EPOCH
    
    print("=" * 60)
    print("FILTER MODULE TESTS")
    print("=" * 60)
    
    # Test parameters
    fs = 1000  # Hz
    duration = 5.0  # seconds
    t = np.arange(0, duration, 1/fs)
    n_samples = len(t)
    
    # ========== Test 1: Low-pass removes high frequencies ==========
    print("\n1. Testing low-pass filter removes high frequencies...")
    try:
        # Create signal with low (5 Hz) and high (100 Hz) frequency components
        low_freq = 5  # Hz
        high_freq = 100  # Hz
        
        signal = np.sin(2 * np.pi * low_freq * t) + np.sin(2 * np.pi * high_freq * t)
        
        signal_xr = xr.DataArray(
            signal,
            dims=[DIM_TIME],
            coords={DIM_TIME: t},
            attrs={ATTR_FS: fs}
        )
        
        # Low-pass at 20 Hz should remove 100 Hz component
        filtered = lp_filter(signal_xr, cutoff_freq=20, order=4)
        
        # Check power at low and high frequencies using FFT
        fft_orig = np.abs(np.fft.rfft(signal))
        fft_filt = np.abs(np.fft.rfft(filtered.values))
        freqs = np.fft.rfftfreq(n_samples, 1/fs)
        
        # Find power at target frequencies
        low_idx = np.argmin(np.abs(freqs - low_freq))
        high_idx = np.argmin(np.abs(freqs - high_freq))
        
        low_power_ratio = fft_filt[low_idx] / fft_orig[low_idx]
        high_power_ratio = fft_filt[high_idx] / (fft_orig[high_idx] + 1e-10)
        
        assert low_power_ratio > 0.9, f"Low freq attenuated too much: {low_power_ratio:.3f}"
        assert high_power_ratio < 0.1, f"High freq not attenuated enough: {high_power_ratio:.3f}"
        
        print(f"   ✓ Low freq (5 Hz) preserved: {low_power_ratio:.3f}")
        print(f"   ✓ High freq (100 Hz) attenuated: {high_power_ratio:.4f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 2: High-pass removes low frequencies ==========
    print("\n2. Testing high-pass filter removes low frequencies...")
    try:
        # Create signal with low (2 Hz) and high (50 Hz) frequency components
        low_freq = 2  # Hz
        high_freq = 50  # Hz
        
        signal = np.sin(2 * np.pi * low_freq * t) + np.sin(2 * np.pi * high_freq * t)
        
        signal_xr = xr.DataArray(
            signal,
            dims=[DIM_TIME],
            coords={DIM_TIME: t},
            attrs={ATTR_FS: fs}
        )
        
        # High-pass at 10 Hz should remove 2 Hz component
        filtered = hp_filter(signal_xr, cutoff_freq=10, order=4)
        
        # Check power at low and high frequencies using FFT
        fft_orig = np.abs(np.fft.rfft(signal))
        fft_filt = np.abs(np.fft.rfft(filtered.values))
        freqs = np.fft.rfftfreq(n_samples, 1/fs)
        
        low_idx = np.argmin(np.abs(freqs - low_freq))
        high_idx = np.argmin(np.abs(freqs - high_freq))
        
        low_power_ratio = fft_filt[low_idx] / (fft_orig[low_idx] + 1e-10)
        high_power_ratio = fft_filt[high_idx] / fft_orig[high_idx]
        
        assert low_power_ratio < 0.1, f"Low freq not attenuated enough: {low_power_ratio:.3f}"
        assert high_power_ratio > 0.9, f"High freq attenuated too much: {high_power_ratio:.3f}"
        
        print(f"   ✓ Low freq (2 Hz) attenuated: {low_power_ratio:.4f}")
        print(f"   ✓ High freq (50 Hz) preserved: {high_power_ratio:.3f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 3: Multi-channel support ==========
    print("\n3. Testing multi-channel support...")
    try:
        n_chans = 4
        
        multi_signal = np.zeros((n_samples, n_chans))
        for ch in range(n_chans):
            # Each channel has 5 Hz + 100 Hz
            multi_signal[:, ch] = (np.sin(2 * np.pi * 5 * t) + 
                                   np.sin(2 * np.pi * 100 * t))
        
        multi_xr = xr.DataArray(
            multi_signal,
            dims=[DIM_TIME, DIM_CHAN],
            coords={DIM_TIME: t, DIM_CHAN: np.arange(n_chans)},
            attrs={ATTR_FS: fs}
        )
        
        filtered = lp_filter(multi_xr, cutoff_freq=20)
        
        assert filtered.shape == multi_xr.shape, "Shape mismatch"
        assert DIM_CHAN in filtered.dims, "Channel dimension lost"
        
        print(f"   ✓ Multi-channel: shape={filtered.shape}, dims={filtered.dims}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 4: 3D data (time, channel, trial) ==========
    print("\n4. Testing 3D data (time, channel, trial)...")
    try:
        n_chans = 2
        n_trials = 3
        
        data_3d = np.random.randn(n_samples, n_chans, n_trials)
        
        xr_3d = xr.DataArray(
            data_3d,
            dims=[DIM_TIME, DIM_CHAN, DIM_EPOCH],
            coords={
                DIM_TIME: t,
                DIM_CHAN: np.arange(n_chans),
                DIM_EPOCH: np.arange(n_trials)
            },
            attrs={ATTR_FS: fs}
        )
        
        filtered = lp_filter(xr_3d, cutoff_freq=50)
        
        assert filtered.shape == xr_3d.shape, "Shape mismatch"
        assert filtered.dims == xr_3d.dims, "Dims mismatch"
        print(f"   ✓ 3D data: shape={filtered.shape}, dims={filtered.dims}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 5: Coordinate/attribute preservation ==========
    print("\n5. Testing coordinate and attribute preservation...")
    try:
        signal_xr = xr.DataArray(
            np.sin(2 * np.pi * 10 * t),
            dims=[DIM_TIME],
            coords={DIM_TIME: t},
            attrs={ATTR_FS: fs, 'units': 'mV', 'custom_attr': 'test_value'}
        )
        
        filtered = lp_filter(signal_xr, cutoff_freq=50)
        
        assert ATTR_FS in filtered.attrs, "fs attribute lost"
        assert filtered.attrs[ATTR_FS] == fs, "fs attribute value changed"
        assert 'units' in filtered.attrs, "units attribute lost"
        assert 'custom_attr' in filtered.attrs, "custom attribute lost"
        assert np.allclose(filtered[DIM_TIME].values, t), "Time coordinates changed"
        
        print(f"   ✓ Attributes preserved: {list(filtered.attrs.keys())}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 6: Sample rate from time coordinates ==========
    print("\n6. Testing sample rate inference from time coordinates...")
    try:
        # Create signal WITHOUT fs attribute
        signal_xr = xr.DataArray(
            np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 100 * t),
            dims=[DIM_TIME],
            coords={DIM_TIME: t}
            # No attrs - should infer fs from time
        )
        
        filtered = lp_filter(signal_xr, cutoff_freq=20)
        
        # Verify filtering worked correctly (high freq removed)
        fft_filt = np.abs(np.fft.rfft(filtered.values))
        freqs = np.fft.rfftfreq(n_samples, 1/fs)
        high_idx = np.argmin(np.abs(freqs - 100))
        
        # If fs was inferred correctly, 100 Hz should be attenuated
        assert fft_filt[high_idx] < 100, "High freq not attenuated - fs inference failed"
        
        print(f"   ✓ Sample rate correctly inferred from time coordinates")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 7: Different filter orders ==========
    print("\n7. Testing different filter orders...")
    try:
        signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 50 * t)
        
        signal_xr = xr.DataArray(
            signal,
            dims=[DIM_TIME],
            coords={DIM_TIME: t},
            attrs={ATTR_FS: fs}
        )
        
        # Compare attenuation with different orders
        filt_order2 = lp_filter(signal_xr, cutoff_freq=20, order=2)
        filt_order6 = lp_filter(signal_xr, cutoff_freq=20, order=6)
        
        fft_o2 = np.abs(np.fft.rfft(filt_order2.values))
        fft_o6 = np.abs(np.fft.rfft(filt_order6.values))
        freqs = np.fft.rfftfreq(n_samples, 1/fs)
        
        high_idx = np.argmin(np.abs(freqs - 50))
        
        # Higher order should have sharper cutoff (more attenuation at 50 Hz)
        atten_o2 = fft_o2[high_idx]
        atten_o6 = fft_o6[high_idx]
        
        assert atten_o6 < atten_o2, "Higher order should attenuate more"
        
        print(f"   ✓ Order 2 attenuation at 50 Hz: {atten_o2:.4f}")
        print(f"   ✓ Order 6 attenuation at 50 Hz: {atten_o6:.4f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 8: Zero-phase filtering (filtfilt) ==========
    print("\n8. Testing zero-phase filtering...")
    try:
        # Create a pulse signal
        pulse = np.zeros(n_samples)
        pulse[n_samples//2] = 1.0
        
        pulse_xr = xr.DataArray(
            pulse,
            dims=[DIM_TIME],
            coords={DIM_TIME: t},
            attrs={ATTR_FS: fs}
        )
        
        filtered = lp_filter(pulse_xr, cutoff_freq=50, order=4)
        
        # Zero-phase filter should produce symmetric response around the pulse
        # Find peak location
        peak_idx = np.argmax(np.abs(filtered.values))
        
        # Check symmetry around peak (should be centered at original pulse location)
        assert np.abs(peak_idx - n_samples//2) < 5, "Phase shift detected"
        
        print(f"   ✓ Zero-phase confirmed: peak at {peak_idx}, original at {n_samples//2}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 9: Bandpass equivalent (HP + LP) ==========
    print("\n9. Testing bandpass via HP + LP cascade...")
    try:
        # Create signal with 2 Hz, 20 Hz, and 100 Hz components
        signal = (np.sin(2 * np.pi * 2 * t) + 
                  np.sin(2 * np.pi * 20 * t) + 
                  np.sin(2 * np.pi * 100 * t))
        
        signal_xr = xr.DataArray(
            signal,
            dims=[DIM_TIME],
            coords={DIM_TIME: t},
            attrs={ATTR_FS: fs}
        )
        
        # Bandpass 10-50 Hz (should keep 20 Hz, remove 2 Hz and 100 Hz)
        filtered = hp_filter(signal_xr, cutoff_freq=10, order=4)
        filtered = lp_filter(filtered, cutoff_freq=50, order=4)
        
        fft_filt = np.abs(np.fft.rfft(filtered.values))
        freqs = np.fft.rfftfreq(n_samples, 1/fs)
        
        idx_2 = np.argmin(np.abs(freqs - 2))
        idx_20 = np.argmin(np.abs(freqs - 20))
        idx_100 = np.argmin(np.abs(freqs - 100))
        
        fft_orig = np.abs(np.fft.rfft(signal))
        
        ratio_2 = fft_filt[idx_2] / (fft_orig[idx_2] + 1e-10)
        ratio_20 = fft_filt[idx_20] / fft_orig[idx_20]
        ratio_100 = fft_filt[idx_100] / (fft_orig[idx_100] + 1e-10)
        
        assert ratio_2 < 0.1, f"2 Hz not attenuated: {ratio_2:.3f}"
        assert ratio_20 > 0.9, f"20 Hz too attenuated: {ratio_20:.3f}"
        assert ratio_100 < 0.1, f"100 Hz not attenuated: {ratio_100:.3f}"
        
        print(f"   ✓ 2 Hz attenuated: {ratio_2:.4f}")
        print(f"   ✓ 20 Hz preserved: {ratio_20:.3f}")
        print(f"   ✓ 100 Hz attenuated: {ratio_100:.4f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 10: Visualization ==========
    print("\n10. Creating visualization...")
    try:
        import matplotlib.pyplot as plt
        
        # Create test signal
        signal = (np.sin(2 * np.pi * 5 * t) + 
                  0.5 * np.sin(2 * np.pi * 50 * t) +
                  0.3 * np.sin(2 * np.pi * 150 * t))
        
        signal_xr = xr.DataArray(
            signal,
            dims=[DIM_TIME],
            coords={DIM_TIME: t},
            attrs={ATTR_FS: fs}
        )
        
        lp_filtered = lp_filter(signal_xr, cutoff_freq=20)
        hp_filtered = hp_filter(signal_xr, cutoff_freq=100)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Time domain - first 0.5 seconds
        ax = axes[0, 0]
        t_plot = t[:500]
        ax.plot(t_plot, signal[:500], 'b-', alpha=0.7, label='Original')
        ax.plot(t_plot, lp_filtered.values[:500], 'r-', lw=2, label='LP (20 Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('Low-pass Filter Effect')
        ax.legend()
        
        ax = axes[0, 1]
        ax.plot(t_plot, signal[:500], 'b-', alpha=0.7, label='Original')
        ax.plot(t_plot, hp_filtered.values[:500], 'g-', lw=2, label='HP (100 Hz)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title('High-pass Filter Effect')
        ax.legend()
        
        # Frequency domain
        freqs = np.fft.rfftfreq(n_samples, 1/fs)
        fft_orig = np.abs(np.fft.rfft(signal))
        fft_lp = np.abs(np.fft.rfft(lp_filtered.values))
        fft_hp = np.abs(np.fft.rfft(hp_filtered.values))
        
        ax = axes[1, 0]
        ax.semilogy(freqs[:250], fft_orig[:250], 'b-', alpha=0.7, label='Original')
        ax.semilogy(freqs[:250], fft_lp[:250], 'r-', lw=2, label='LP (20 Hz)')
        ax.axvline(20, color='r', ls='--', alpha=0.5)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_title('LP Filter Frequency Response')
        ax.legend()
        
        ax = axes[1, 1]
        ax.semilogy(freqs[:250], fft_orig[:250], 'b-', alpha=0.7, label='Original')
        ax.semilogy(freqs[:250], fft_hp[:250], 'g-', lw=2, label='HP (100 Hz)')
        ax.axvline(100, color='g', ls='--', alpha=0.5)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Magnitude')
        ax.set_title('HP Filter Frequency Response')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('filter_tests.png', dpi=150)
        plt.show()
        print(f"   ✓ Visualization saved to filter_tests.png")
    except Exception as e:
        print(f"   ✗ Visualization failed: {e}")
    
    print("\n" + "=" * 60)
    print("FILTER TESTS COMPLETE")
    print("=" * 60)
