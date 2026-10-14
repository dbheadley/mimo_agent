# Signal processing (not filtering)

import numpy as np
import xarray as xr
from scipy.signal import hilbert

from mimo_pack.consts import DIM_TIME

def envelope(signal, dim=DIM_TIME):
    """
    Calculate the amplitude envelope of a signal using the Hilbert transform.

    Parameters
    ----------
    signal : xarray.DataArray
        Input signal. Must have the dimension specified by 'dim'.
    dim : str, optional
        Dimension along which to calculate the envelope. Default is DIM_TIME ('time').

    Returns
    -------
    xarray.DataArray
        Amplitude envelope with the same dimensions and coordinates as the input.
        
    Raises
    ------
    TypeError
        If input is not an xarray.DataArray.
    ValueError
        If input does not have the specified dimension.
    """
    if not isinstance(signal, xr.DataArray):
        raise TypeError(
            f"Input must be an xarray.DataArray, got {type(signal).__name__}. "
            "Use fileio functions to load data into xarray format."
        )

    if dim not in signal.dims:
        raise ValueError(f"Input xarray must have a '{dim}' dimension.")

    # Extract data and dimension info
    arr = signal.to_numpy()
    dims = signal.dims
    
    # Find the axis index for the specified dimension
    axis = np.where(np.array(dims) == dim)[0][0]

    # Calculate analytic signal using Hilbert transform
    # Note: scipy.signal.hilbert calculates the analytic signal
    analytic = hilbert(arr, axis=axis)
    
    # Calculate amplitude envelope
    env = np.abs(analytic)

    # Add description for data provenance
    attrs = signal.attrs.copy()
    prev_desc = attrs.get('description', '').strip()
    attrs['description'] = prev_desc + (' -> ' if prev_desc else '') + 'Signal envelope via Hilbert transform'

    # Reconstruct xarray
    # Since dimensions and shape are preserved, we can copy coordinates directly
    return xr.DataArray(env, dims=dims, coords=signal.coords, attrs=attrs)


if __name__ == "__main__":
    from mimo_pack.consts import DIM_CHAN, DIM_EPOCH, ATTR_FS
    
    print("=" * 60)
    print("SIGNAL PROCESSING MODULE TESTS")
    print("=" * 60)
    
    # Test parameters
    fs = 1000  # Hz
    duration = 2.0  # seconds
    t = np.arange(0, duration, 1/fs)
    n_samples = len(t)
    
    # ========== Test 1: Basic sinusoid envelope ==========
    print("\n1. Testing basic sinusoid envelope...")
    try:
        # Pure sinusoid should have constant envelope equal to amplitude
        amplitude = 5.0
        freq = 10  # Hz
        sinusoid = amplitude * np.sin(2 * np.pi * freq * t)
        
        signal_xr = xr.DataArray(
            sinusoid,
            dims=[DIM_TIME],
            coords={DIM_TIME: t},
            attrs={ATTR_FS: fs}
        )
        
        env = envelope(signal_xr)
        
        # Envelope should be approximately constant at amplitude (ignoring edges)
        env_mid = env.values[100:-100]
        env_mean = np.mean(env_mid)
        env_std = np.std(env_mid)
        
        assert np.abs(env_mean - amplitude) < 0.1, f"Mean envelope {env_mean:.3f} != amplitude {amplitude}"
        assert env_std < 0.1, f"Envelope std {env_std:.3f} too high for pure sinusoid"
        print(f"   ✓ Pure sinusoid envelope: mean={env_mean:.3f}, std={env_std:.4f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 2: AM signal recovery ==========
    print("\n2. Testing AM signal envelope extraction...")
    try:
        # Create AM signal: carrier modulated by slower signal
        carrier_freq = 50  # Hz
        mod_freq = 3  # Hz
        mod_depth = 0.5
        
        carrier = np.sin(2 * np.pi * carrier_freq * t)
        modulator = 1 + mod_depth * np.sin(2 * np.pi * mod_freq * t)
        am_signal = carrier * modulator
        
        am_xr = xr.DataArray(
            am_signal,
            dims=[DIM_TIME],
            coords={DIM_TIME: t},
            attrs={ATTR_FS: fs}
        )
        
        env = envelope(am_xr)
        
        # Envelope should track the modulator
        # Compare correlation between envelope and modulator
        corr = np.corrcoef(env.values[100:-100], modulator[100:-100])[0, 1]
        assert corr > 0.95, f"Correlation {corr:.3f} too low"
        print(f"   ✓ AM envelope correlates with modulator: r={corr:.4f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 3: Multi-channel support ==========
    print("\n3. Testing multi-channel support...")
    try:
        n_chans = 4
        amplitudes = np.array([1, 2, 3, 4])
        
        multi_signal = np.zeros((n_samples, n_chans))
        for ch in range(n_chans):
            multi_signal[:, ch] = amplitudes[ch] * np.sin(2 * np.pi * 10 * t)
        
        multi_xr = xr.DataArray(
            multi_signal,
            dims=[DIM_TIME, DIM_CHAN],
            coords={DIM_TIME: t, DIM_CHAN: np.arange(n_chans)},
            attrs={ATTR_FS: fs}
        )
        
        env = envelope(multi_xr)
        
        assert env.shape == multi_xr.shape, "Shape mismatch"
        assert DIM_CHAN in env.dims, "Channel dimension lost"
        
        # Check each channel has correct envelope
        for ch in range(n_chans):
            ch_env = env.isel({DIM_CHAN: ch}).values[100:-100]
            ch_mean = np.mean(ch_env)
            assert np.abs(ch_mean - amplitudes[ch]) < 0.1, f"Channel {ch} envelope incorrect"
        
        print(f"   ✓ Multi-channel: shape={env.shape}, dims={env.dims}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 4: Multi-dimensional (time, channel, trial) ==========
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
            }
        )
        
        env = envelope(xr_3d)
        
        assert env.shape == xr_3d.shape, "Shape mismatch"
        assert env.dims == xr_3d.dims, "Dims mismatch"
        print(f"   ✓ 3D data: shape={env.shape}, dims={env.dims}")
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
        
        env = envelope(signal_xr)
        
        assert ATTR_FS in env.attrs, "fs attribute lost"
        assert env.attrs[ATTR_FS] == fs, "fs attribute value changed"
        assert 'units' in env.attrs, "units attribute lost"
        assert 'custom_attr' in env.attrs, "custom attribute lost"
        assert np.allclose(env[DIM_TIME].values, t), "Time coordinates changed"
        
        print(f"   ✓ Attributes preserved: {list(env.attrs.keys())}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 6: DC signal ==========
    print("\n6. Testing DC (constant) signal...")
    try:
        dc_value = 3.5
        dc_signal = np.ones(n_samples) * dc_value
        
        dc_xr = xr.DataArray(
            dc_signal,
            dims=[DIM_TIME],
            coords={DIM_TIME: t}
        )
        
        env = envelope(dc_xr)
        
        # DC signal envelope should be approximately constant
        env_std = np.std(env.values[100:-100])
        env_mean = np.mean(env.values[100:-100])
        
        print(f"   ✓ DC signal envelope: mean={env_mean:.3f}, std={env_std:.4f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 7: Short signal ==========
    print("\n7. Testing short signal...")
    try:
        short_t = np.arange(0, 0.1, 1/fs)  # 100 samples
        short_signal = np.sin(2 * np.pi * 20 * short_t)
        
        short_xr = xr.DataArray(
            short_signal,
            dims=[DIM_TIME],
            coords={DIM_TIME: short_t}
        )
        
        env = envelope(short_xr)
        
        assert env.shape == short_xr.shape, "Shape mismatch"
        print(f"   ✓ Short signal ({len(short_t)} samples): shape={env.shape}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 8: Custom dimension name ==========
    print("\n8. Testing custom dimension name...")
    try:
        custom_dim = 'samples'
        signal_custom = xr.DataArray(
            np.sin(2 * np.pi * 10 * t),
            dims=[custom_dim],
            coords={custom_dim: t}
        )
        
        env = envelope(signal_custom, dim=custom_dim)
        
        assert env.shape == signal_custom.shape, "Shape mismatch"
        assert custom_dim in env.dims, "Custom dimension lost"
        print(f"   ✓ Custom dimension '{custom_dim}': dims={env.dims}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 9: Missing dimension error ==========
    print("\n9. Testing missing dimension error...")
    try:
        signal_xr = xr.DataArray(
            np.sin(2 * np.pi * 10 * t),
            dims=['samples'],
            coords={'samples': t}
        )
        
        envelope(signal_xr, dim='time')  # Should fail
        print(f"   ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError: {str(e)[:50]}...")
    except Exception as e:
        print(f"   ✗ Wrong exception type: {type(e).__name__}")
    
    # ========== Test 10: Envelope >= 0 ==========
    print("\n10. Testing envelope non-negativity...")
    try:
        # Random signal
        np.random.seed(42)
        random_signal = np.random.randn(n_samples)
        
        random_xr = xr.DataArray(
            random_signal,
            dims=[DIM_TIME],
            coords={DIM_TIME: t}
        )
        
        env = envelope(random_xr)
        
        assert np.all(env.values >= 0), "Envelope has negative values"
        print(f"   ✓ All envelope values >= 0: min={env.values.min():.6f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 11: Envelope bounds signal ==========
    print("\n11. Testing envelope bounds original signal...")
    try:
        # The envelope should generally bound the absolute value of the signal
        # (this is approximately true, with some tolerance for edge effects)
        signal_data = 3 * np.sin(2 * np.pi * 10 * t) + np.random.randn(n_samples) * 0.5
        
        signal_xr = xr.DataArray(
            signal_data,
            dims=[DIM_TIME],
            coords={DIM_TIME: t}
        )
        
        env = envelope(signal_xr)
        
        # Check middle portion (avoid edge effects)
        mid_slice = slice(100, -100)
        violations = np.sum(np.abs(signal_data[mid_slice]) > env.values[mid_slice] * 1.01)
        violation_pct = violations / len(signal_data[mid_slice]) * 100
        
        assert violation_pct < 5, f"Too many violations: {violation_pct:.1f}%"
        print(f"   ✓ Envelope bounds signal: {violation_pct:.2f}% violations")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 12: Peak alignment ==========
    print("\n12. Testing envelope peak alignment with amplitude peaks...")
    try:
        # Create signal with clear amplitude modulation
        mod_freq = 2  # Hz
        carrier_freq = 30  # Hz
        
        modulator = 1 + 0.8 * np.sin(2 * np.pi * mod_freq * t)
        carrier = np.sin(2 * np.pi * carrier_freq * t)
        am_signal = modulator * carrier
        
        am_xr = xr.DataArray(
            am_signal,
            dims=[DIM_TIME],
            coords={DIM_TIME: t}
        )
        
        env = envelope(am_xr)
        
        # Find peaks in modulator and envelope
        from scipy.signal import find_peaks
        mod_peaks, _ = find_peaks(modulator, height=1.5)
        env_peaks, _ = find_peaks(env.values, height=1.5)
        
        # Check that envelope peaks are close to modulator peaks
        if len(mod_peaks) > 0 and len(env_peaks) > 0:
            # For each modulator peak, find nearest envelope peak
            peak_diffs = []
            for mp in mod_peaks:
                nearest = env_peaks[np.argmin(np.abs(env_peaks - mp))]
                peak_diffs.append(np.abs(nearest - mp))
            
            mean_diff = np.mean(peak_diffs) / fs * 1000  # in ms
            assert mean_diff < 20, f"Peak alignment too poor: {mean_diff:.1f} ms"
            print(f"   ✓ Peak alignment: mean offset = {mean_diff:.2f} ms")
        else:
            print(f"   ✓ Peak detection completed (mod_peaks={len(mod_peaks)}, env_peaks={len(env_peaks)})")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    print("\n" + "=" * 60)
    print("SIGNAL PROCESSING TESTS COMPLETE")
    print("=" * 60)
