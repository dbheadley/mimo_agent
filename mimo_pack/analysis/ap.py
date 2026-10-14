# Analysis function for AP data

import xarray as xr
from mimo_pack.util.xarray import interp_stack
from mimo_pack.analysis.signal_proc import envelope
from mimo_pack.consts import DIM_TIME, DIM_EPOCH
import numpy as np

def evoked_ap(ap_trials, align_dim=DIM_EPOCH, spacing=20, clip_min=5, clip_max=20):
    """Calculate the mean evoked action potential envelope from epoch-windowed AP data.

    This function takes epoch-windowed action potential (AP) data, interpolates
    spatially, computes the mean evoked potential across all epochs, calculates
    the temporal envelope, and clips to a specified amplitude range.

    Parameters
    ----------
    ap_trials : xarray.DataArray
        Epoch-windowed AP data with dimensions including time and an epoch
        dimension. Must be an xarray.DataArray with proper coordinates.
        Use fileio.ap.load_ap_windows to load epoch windows.
    align_dim : str, optional
        The dimension name to average over. Defaults to 'epoch'.
    spacing : int, optional
        The spatial interpolation spacing in microns. Defaults to 20.
    clip_min : float, optional
        The minimum value to clip the envelope amplitude to. Defaults to 5.
    clip_max : float, optional
        The maximum value to clip the envelope amplitude to. Defaults to 20.

    Returns
    -------
    ap_env : xarray.DataArray
        The clipped envelope of the mean evoked action potential. 
        Time coordinate is converted to milliseconds.

    Raises
    ------
    TypeError
        If input is not an xarray.DataArray.
    ValueError
        If required dimensions are missing from input.

    Examples
    --------
    >>> from mimo_pack.fileio.ap import load_ap_windows
    >>> # Load windowed data using fileio (done separately)
    >>> ap_data = load_ap_windows(ap_path, centers=times, pre=0.01, post=0.03)
    >>> # Then pass to analysis function
    >>> ap_env = evoked_ap(ap_data, spacing=20)
    """
    # Type checking
    if not isinstance(ap_trials, xr.DataArray):
        raise TypeError(
            f"Input must be an xarray.DataArray, got {type(ap_trials).__name__}. "
            "Use fileio.ap.load_ap_windows to load windowed AP data first."
        )
    
    # Validate dimensions
    if align_dim not in ap_trials.dims:
        raise ValueError(
            f"Input must have '{align_dim}' dimension. "
            f"Available dimensions: {list(ap_trials.dims)}"
        )
    
    if DIM_TIME not in ap_trials.dims:
        raise ValueError(
            f"Input must have '{DIM_TIME}' dimension. "
            f"Available dimensions: {list(ap_trials.dims)}"
        )
    
    # Perform spatial interpolation
    ap_interp = interp_stack(ap_trials, spacing=spacing)
    
    # Average across epochs
    ap_mean = ap_interp.mean(dim=align_dim)
    
    # Calculate envelope and clip
    ap_env = envelope(ap_mean, dim=DIM_TIME)
    ap_env = np.clip(ap_env, clip_min, clip_max)
    
    # Add description attribute
    desc_base = ap_trials.attrs.get('description', 'AP trials')
    ap_env.attrs['description'] = f"{desc_base} -> evoked AP envelope (clipped {clip_min}-{clip_max})"
    
    return ap_env


if __name__ == "__main__":
    import xarray as xr
    import matplotlib.pyplot as plt
    from mimo_pack.consts import ATTR_FS, DIM_CHAN, COORD_Y, COORD_X
    from mimo_pack.plot.map import wave_map
    
    print("=" * 60)
    print("AP ANALYSIS MODULE TESTS")
    print("=" * 60)
    
    # ========== Helper: Generate spike waveform ==========
    def spike_waveform(n_samples, fs, amplitude=100):
        """Generate a realistic synthetic extracellular spike waveform.

        Creates a triphasic waveform: small positive → large negative
        trough → positive recovery, using a sum of Gaussians.

        Parameters
        ----------
        n_samples : int
            Total number of samples (unused internally; spike length is
            derived from *fs* and a fixed 1 ms duration).
        fs : float
            Sampling rate in Hz.
        amplitude : float, optional
            Peak-to-peak scaling factor. Default is 100.

        Returns
        -------
        numpy.ndarray
            1-D waveform array of length ``int(0.001 * fs)``.
        """
        # Spike duration ~1ms
        spike_dur = 0.001  # 1 ms
        spike_samples = int(spike_dur * fs)
        t_spike = np.linspace(0, spike_dur, spike_samples)
        
        # Triphasic waveform using sum of Gaussians
        # Small initial positive, large negative trough, positive recovery
        sigma = spike_dur / 6
        t_center = spike_dur / 2
        
        # Negative phase (main spike)
        neg_phase = -np.exp(-((t_spike - t_center) ** 2) / (2 * (sigma * 0.8) ** 2))
        # Positive recovery phase
        pos_phase = 0.4 * np.exp(-((t_spike - t_center - sigma * 1.5) ** 2) / (2 * sigma ** 2))
        # Small initial positive
        init_phase = 0.15 * np.exp(-((t_spike - t_center + sigma * 1.2) ** 2) / (2 * (sigma * 0.6) ** 2))
        
        waveform = (neg_phase + pos_phase + init_phase) * amplitude
        return waveform
    
    # ========== Test 1: Create synthetic AP trial data ==========
    print("\n1. Creating synthetic AP trial data with spike waveforms...")
    try:
        # Generate synthetic "AP band" data (high frequency content)
        fs = 30000  # AP band sampling rate
        pre = 0.01  # 10 ms pre
        post = 0.03  # 30 ms post
        duration = pre + post
        n_chans = 32
        n_trials = 20
        t = np.arange(0, duration, 1/fs) - pre
        n_samples = len(t)
        
        # Create synthetic AP data with evoked spike bursts
        np.random.seed(42)
        trial_data = np.zeros((n_samples, n_chans, n_trials))
        
        # Generate a template spike waveform
        spike_template = spike_waveform(n_samples, fs, amplitude=1)
        spike_len = len(spike_template)
        
        for trial in range(n_trials):
            for ch in range(n_chans):
                # Background noise (filtered to AP band characteristics)
                trial_data[:, ch, trial] = np.random.randn(n_samples) * 15
                
                # Add evoked spikes after t=0
                # Number of spikes varies by channel (deeper = more activity)
                n_spikes = int(3 + (n_chans - ch) * 0.3 + np.random.randint(0, 3))
                
                # Spike times: clustered after stimulus with some jitter
                base_latency = 0.002  # 2 ms latency
                for spike_i in range(n_spikes):
                    # Spike time with jitter
                    spike_time = base_latency + spike_i * 0.002 + np.random.rand() * 0.001
                    spike_idx = int((pre + spike_time) * fs)
                    
                    if spike_idx + spike_len < n_samples:
                        # Amplitude varies by channel (larger near soma) and trial
                        amp = (80 + (n_chans//2 - abs(ch - n_chans//2)) * 5) * (0.6 + 0.8 * np.random.rand())
                        trial_data[spike_idx:spike_idx + spike_len, ch, trial] += spike_template * amp
        
        ap_trials = xr.DataArray(
            trial_data,
            dims=[DIM_TIME, DIM_CHAN, DIM_EPOCH],
            coords={
                DIM_TIME: t,
                DIM_CHAN: np.arange(n_chans),
                DIM_EPOCH: np.arange(n_trials),
                COORD_X: (DIM_CHAN, np.zeros(n_chans)),  # Single shank
                COORD_Y: (DIM_CHAN, np.arange(n_chans) * 20),  # 20 µm spacing
            },
            attrs={ATTR_FS: fs}
        )
        
        print(f"   ✓ Created AP trials: shape={ap_trials.shape}, dims={ap_trials.dims}")
        print(f"   Sample rate: {fs} Hz")
        print(f"   Time range: {t.min()*1000:.1f} to {t.max()*1000:.1f} ms")
        print(f"   Spike waveform length: {spike_len} samples ({spike_len/fs*1000:.2f} ms)")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        raise
    
    # ========== Test 2: Test evoked_ap function ==========
    print("\n2. Testing evoked_ap() function...")
    try:
        ap_env = evoked_ap(ap_trials, align_dim=DIM_EPOCH, spacing=40, clip_min=5, clip_max=50)
        
        assert DIM_TIME in ap_env.dims, "Time dimension missing from output"
        assert ap_env.values.min() >= 5, "Clipping min failed"
        assert ap_env.values.max() <= 50, "Clipping max failed"
        
        print(f"   ✓ evoked_ap successful")
        print(f"   Output shape: {ap_env.shape}, dims: {ap_env.dims}")
        print(f"   Time now in ms: {ap_env[DIM_TIME].values.min():.1f} to {ap_env[DIM_TIME].values.max():.1f} ms")
        print(f"   Envelope range: {ap_env.values.min():.1f} to {ap_env.values.max():.1f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 3: Test error handling ==========
    print("\n3. Testing error handling...")
    try:
        evoked_ap(np.random.randn(100, 10))  # numpy array instead of xarray
        print(f"   ✗ Should have raised TypeError")
    except TypeError as e:
        print(f"   ✓ Correctly raised TypeError for non-xarray input")
    
    try:
        # Missing required dimension
        bad_xr = xr.DataArray(np.random.randn(100, 10), dims=['time', 'channel'])
        evoked_ap(bad_xr, align_dim='trial')
        print(f"   ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError for missing dimension")
    
    # ========== Test 4: Test envelope directly ==========
    print("\n4. Testing envelope on single trial...")
    try:
        single_trial = ap_xr.isel({DIM_EPOCH: 0})
        env = envelope(single_trial, dim=DIM_TIME)
        
        assert env.shape == single_trial.shape, "Shape mismatch"
        assert np.all(env.values >= 0), "Envelope has negative values"
        
        print(f"   ✓ Envelope extracted: shape={env.shape}")
        print(f"   Envelope range: {env.values.min():.2f} to {env.values.max():.2f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 5: Visualization ==========
    print("\n5. Creating visualization...")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Single trial AP waveforms using wave_map
        ax = axes[0, 0]
        single_trial = ap_trials.isel({DIM_EPOCH: 0})
        wave_map(single_trial, x_scale=0.00005, y_scale=0.5, ax=ax, lw=0.3, color='b')
        ax.axvline(0, color='r', ls='--', alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Depth (µm)')
        ax.set_title('Single Trial AP (wave_map)')
        
        # Plot 2: Mean AP waveforms using wave_map
        ax = axes[0, 1]
        ap_mean = ap_trials.mean(dim=DIM_EPOCH)
        wave_map(ap_mean, x_scale=0.00005, y_scale=1, ax=ax, lw=0.5, color='k')
        ax.axvline(0, color='r', ls='--', alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Depth (µm)')
        ax.set_title('Mean Evoked AP')
        
        # Plot 3: Evoked AP envelope using wave_map
        ax = axes[1, 0]
        wave_map(ap_env, x_scale=0.05, y_scale=1, ax=ax, lw=1, color='darkred')
        ax.axvline(0, color='k', ls='--', alpha=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Depth (µm)')
        ax.set_title('Evoked AP Envelope')
        
        # Plot 4: Envelope heatmap
        ax = axes[1, 1]
        # Stack if needed for 2D plot
        env_2d = ap_env.stack(channel=('ch_x', 'ch_y')) if 'ch_x' in ap_env.dims else ap_env
        t_ms = env_2d[DIM_TIME].values
        y_coords = env_2d[COORD_Y].values if COORD_Y in env_2d.coords else np.arange(env_2d.shape[1])
        
        im = ax.pcolormesh(
            t_ms,
            y_coords,
            env_2d.values.T,
            shading='auto',
            cmap='hot'
        )
        ax.axvline(0, color='w', ls='--', lw=2)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Depth (µm)')
        ax.set_title('AP Envelope Heatmap')
        plt.colorbar(im, ax=ax, label='Amplitude')
        
        plt.tight_layout()
        plt.savefig('ap_analysis_tests.png', dpi=150)
        plt.show()
        print(f"   ✓ Visualization saved to ap_analysis_tests.png")
    except Exception as e:
        print(f"   ✗ Visualization failed: {e}")
    
    print("\n" + "=" * 60)
    print("AP ANALYSIS TESTS COMPLETE")
    print("=" * 60)
