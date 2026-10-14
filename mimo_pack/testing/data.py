# Testing data generators for mimo_pack library
# Author: Drew Headley
# Date: 2025-06-16

import numpy as np
import xarray as xr
import pynapple as nap

from mimo_pack.consts import (
    DIM_TIME, DIM_CHAN, DIM_EPOCH, COORD_X, COORD_Y, COORD_SHANK,
    ATTR_FS, ATTR_UNITS
)


def synthetic_lfp(duration, fs, n_chans, exponent=-2, oscillations=True, seed=None):
    """
    Generate synthetic LFP data with 1/f noise characteristics.

    Parameters
    ----------
    duration : float
        Duration of the signal in seconds.
    fs : float
        Sampling frequency in Hz.
    n_chans : int
        Number of channels.
    exponent : float, optional
        The 1/f exponent for the power spectrum. Default is -2 (brown noise).
        Common values: -1 (pink noise), -2 (brown noise).
    oscillations : bool, optional
        Whether to add oscillatory bursts (beta/gamma) and sustained theta.
        Default is True.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    xarray.DataArray
        Synthetic LFP data with proper coordinates and metadata.
        Dimensions: (time, channel)
        Coordinates: time, channel, ch_x, ch_y, ch_shank
        Attributes: fs, units
        
    Notes
    -----
    Lower frequency components are more correlated across channels than 
    high frequency components, mimicking the spatial properties of real LFP.
    Oscillatory bursts at 20 Hz and 50 Hz (50-200 ms duration) and sustained
    theta oscillations (4-8 Hz, several seconds) are optionally added.
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = int(duration * fs)
    time = np.arange(n_samples) / fs

    # Generate frequency array
    freqs = np.fft.rfftfreq(n_samples, d=1/fs)
    freqs[0] = 1e-10  # Avoid division by zero

    # Create 1/f^alpha spectrum (alpha = exponent/2 for amplitude spectrum)
    # Power spectrum ~ f^exponent, so amplitude spectrum ~ f^(exponent/2)
    amplitude_spectrum = np.power(freqs, exponent / 2)
    
    # Generate shared low-frequency component and independent high-frequency component
    # Low frequencies (< 30 Hz) are more correlated across channels
    low_freq_cutoff = 30  # Hz
    low_freq_mask = freqs < low_freq_cutoff
    high_freq_mask = ~low_freq_mask
    
    # Shared phase for low frequencies (correlated across channels)
    shared_phases_low = np.random.uniform(0, 2 * np.pi, len(freqs))
    
    data = np.zeros((n_samples, n_chans))
    
    for ch in range(n_chans):
        # Low frequency: use shared phases with small channel-specific jitter
        phase_jitter = np.random.uniform(-np.pi/8, np.pi/8, len(freqs))
        phases_low = shared_phases_low + phase_jitter * low_freq_mask
        
        # High frequency: independent random phases for each channel
        phases_high = np.random.uniform(0, 2 * np.pi, len(freqs))
        
        # Combine phases: shared for low freq, independent for high freq
        phases = np.where(low_freq_mask, phases_low, phases_high)
        
        # Create complex spectrum
        complex_spectrum = amplitude_spectrum * np.exp(1j * phases)
        
        # Inverse FFT to get time domain signal
        signal = np.fft.irfft(complex_spectrum, n=n_samples)
        
        # Normalize
        data[:, ch] = (signal - signal.mean()) / signal.std()
    
    # Add oscillatory components if requested
    if oscillations:
        data = _add_lfp_oscillations(data, time, fs, n_chans)

    # Create channel coordinates (simulating a linear probe)
    ch_ids = np.arange(n_chans)
    ch_x = np.zeros(n_chans)  # All channels on same x position
    ch_y = np.arange(n_chans) * 25.0  # 25 µm spacing
    ch_shank = np.zeros(n_chans, dtype=int)  # Single shank

    lfp = xr.DataArray(
        data,
        dims=[DIM_TIME, DIM_CHAN],
        coords={
            DIM_TIME: time,
            DIM_CHAN: ch_ids,
            COORD_X: (DIM_CHAN, ch_x),
            COORD_Y: (DIM_CHAN, ch_y),
            COORD_SHANK: (DIM_CHAN, ch_shank),
        },
        attrs={
            ATTR_FS: fs,
            ATTR_UNITS: 'µV',
        }
    )

    return lfp


def _add_lfp_oscillations(data, time, fs, n_chans):
    """
    Add oscillatory bursts and sustained theta to LFP data.
    
    Parameters
    ----------
    data : np.ndarray
        LFP data array (n_samples, n_chans).
    time : np.ndarray
        Time vector in seconds.
    fs : float
        Sampling frequency in Hz.
    n_chans : int
        Number of channels.
        
    Returns
    -------
    np.ndarray
        LFP data with added oscillations.
    """
    duration = time[-1] - time[0]
    n_samples = len(time)
    
    # Add beta bursts (20 Hz, 50-200 ms duration)
    n_beta_bursts = max(1, int(duration / 2))  # ~1 burst per 2 seconds
    for _ in range(n_beta_bursts):
        burst_duration = np.random.uniform(0.05, 0.2)  # 50-200 ms
        burst_start = np.random.uniform(0, max(0.01, duration - burst_duration))
        burst_mask = (time >= burst_start) & (time < burst_start + burst_duration)
        
        # Gaussian envelope
        burst_time = time[burst_mask] - burst_start
        envelope = np.exp(-((burst_time - burst_duration/2)**2) / (2 * (burst_duration/4)**2))
        
        # 20 Hz oscillation with shared phase across channels (small jitter)
        shared_phase = np.random.uniform(0, 2 * np.pi)
        burst_amplitude = np.random.uniform(0.5, 1.5)
        
        for ch in range(n_chans):
            phase_jitter = np.random.uniform(-np.pi/6, np.pi/6)
            amp_variation = np.random.uniform(0.8, 1.2)
            oscillation = burst_amplitude * amp_variation * envelope * np.sin(
                2 * np.pi * 20 * burst_time + shared_phase + phase_jitter
            )
            data[burst_mask, ch] += oscillation
    
    # Add gamma bursts (50 Hz, 50-200 ms duration)
    n_gamma_bursts = max(1, int(duration / 3))  # ~1 burst per 3 seconds
    for _ in range(n_gamma_bursts):
        burst_duration = np.random.uniform(0.05, 0.2)  # 50-200 ms
        burst_start = np.random.uniform(0, max(0.01, duration - burst_duration))
        burst_mask = (time >= burst_start) & (time < burst_start + burst_duration)
        
        burst_time = time[burst_mask] - burst_start
        envelope = np.exp(-((burst_time - burst_duration/2)**2) / (2 * (burst_duration/4)**2))
        
        shared_phase = np.random.uniform(0, 2 * np.pi)
        burst_amplitude = np.random.uniform(0.3, 0.8)
        
        for ch in range(n_chans):
            phase_jitter = np.random.uniform(-np.pi/4, np.pi/4)
            amp_variation = np.random.uniform(0.8, 1.2)
            oscillation = burst_amplitude * amp_variation * envelope * np.sin(
                2 * np.pi * 50 * burst_time + shared_phase + phase_jitter
            )
            data[burst_mask, ch] += oscillation
    
    # Add sustained theta (4-8 Hz, several seconds)
    if duration >= 2.0:
        n_theta_epochs = max(1, int(duration / 10))  # ~1 epoch per 10 seconds
        for _ in range(n_theta_epochs):
            theta_duration = np.random.uniform(2.0, min(5.0, duration * 0.4))
            theta_start = np.random.uniform(0, max(0.01, duration - theta_duration))
            theta_mask = (time >= theta_start) & (time < theta_start + theta_duration)
            
            theta_time = time[theta_mask] - theta_start
            theta_freq = np.random.uniform(4, 8)  # 4-8 Hz
            
            # Smooth onset/offset envelope
            ramp_time = 0.3  # 300ms ramp
            envelope = np.ones(len(theta_time))
            onset_mask = theta_time < ramp_time
            offset_mask = theta_time > (theta_duration - ramp_time)
            envelope[onset_mask] = theta_time[onset_mask] / ramp_time
            envelope[offset_mask] = (theta_duration - theta_time[offset_mask]) / ramp_time
            
            shared_phase = np.random.uniform(0, 2 * np.pi)
            theta_amplitude = np.random.uniform(0.8, 1.5)
            
            for ch in range(n_chans):
                phase_jitter = np.random.uniform(-np.pi/8, np.pi/8)
                amp_variation = np.random.uniform(0.9, 1.1)
                oscillation = theta_amplitude * amp_variation * envelope * np.sin(
                    2 * np.pi * theta_freq * theta_time + shared_phase + phase_jitter
                )
                data[theta_mask, ch] += oscillation
    
    return data


def synthetic_ap(duration, fs, n_chans, spike_rate=5.0, lfp_fs=1000, seed=None):
    """
    Generate synthetic action potential data with spike-like events.
    
    The background signal is generated using synthetic_lfp (with 1/f noise and
    oscillations), upsampled to the AP sampling rate, with spike waveforms
    inserted on top.

    Parameters
    ----------
    duration : float
        Duration of the signal in seconds.
    fs : float
        Sampling frequency in Hz for the AP data (typically 30000 Hz).
    n_chans : int
        Number of channels.
    spike_rate : float, optional
        Average spike rate in Hz. Default is 5.0.
    lfp_fs : float, optional
        Sampling frequency used to generate the LFP background before
        upsampling. Default is 1000 Hz.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    xarray.DataArray
        Synthetic AP data with proper coordinates and metadata.
        Dimensions: (time, channel)
        Coordinates: time, channel, ch_x, ch_y, ch_shank
        Attributes: fs, units
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = int(duration * fs)
    time = np.arange(n_samples) / fs

    # Generate LFP background at lower sampling rate
    lfp_background = synthetic_lfp(duration, lfp_fs, n_chans, exponent=-2, 
                                   oscillations=True, seed=None)
    
    # Upsample LFP to AP sampling rate using interpolation
    from scipy import interpolate
    lfp_time = lfp_background.time.values
    data = np.zeros((n_samples, n_chans))
    
    for ch in range(n_chans):
        # Create interpolation function for this channel
        interp_func = interpolate.interp1d(
            lfp_time, 
            lfp_background.isel(channel=ch).values,
            kind='cubic',
            fill_value='extrapolate'
        )
        # Upsample to AP time base
        data[:, ch] = interp_func(time) * 0.1  # Scale down for background

    # Create a spike waveform template (realistic biphasic shape)
    spike_duration_samples = int(0.002 * fs)  # 2ms spike
    t_spike = np.arange(spike_duration_samples) / fs
    # Realistic biphasic waveform: sharp negative deflection followed by positive
    spike_template = (
        -np.exp(-((t_spike - 0.0003) / 0.0001)**2) +  # Sharp negative peak at 0.3ms
        0.3 * np.exp(-((t_spike - 0.0008) / 0.0003)**2)  # Smaller positive at 0.8ms
    )
    spike_template = spike_template / np.abs(spike_template.min())  # Normalize

    # Add spikes at random times
    for ch in range(n_chans):
        n_spikes = np.random.poisson(spike_rate * duration)
        spike_times = np.random.uniform(0, duration - spike_duration_samples / fs, n_spikes)
        spike_indices = (spike_times * fs).astype(int)

        for idx in spike_indices:
            if idx + spike_duration_samples < n_samples:
                amplitude = np.random.uniform(0.5, 1.5)
                data[idx:idx + spike_duration_samples, ch] += spike_template * amplitude

    # Create channel coordinates (simulating a linear probe)
    ch_ids = np.arange(n_chans)
    ch_x = np.zeros(n_chans)
    ch_y = np.arange(n_chans) * 25.0  # 25 µm spacing
    ch_shank = np.zeros(n_chans, dtype=int)

    ap = xr.DataArray(
        data,
        dims=[DIM_TIME, DIM_CHAN],
        coords={
            DIM_TIME: time,
            DIM_CHAN: ch_ids,
            COORD_X: (DIM_CHAN, ch_x),
            COORD_Y: (DIM_CHAN, ch_y),
            COORD_SHANK: (DIM_CHAN, ch_shank),
        },
        attrs={
            ATTR_FS: fs,
            ATTR_UNITS: 'µV',
        }
    )

    return ap


def synthetic_spikes(n_units=10, duration=60.0, rate_range=(1.0, 20.0), seed=None):
    """
    Generate synthetic spike data as a pynapple TsGroup.

    Parameters
    ----------
    n_units : int, optional
        Number of units. Default is 10.
    duration : float, optional
        Duration of the recording in seconds. Default is 60.0.
    rate_range : tuple of float, optional
        Range of firing rates (min, max) in Hz. Default is (1.0, 20.0).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pynapple.TsGroup
        Synthetic spike data with unit metadata.
        Metadata includes: lfp_channel, firing_rate
    """
    if seed is not None:
        np.random.seed(seed)

    spike_times = {}
    rates = []
    lfp_channels = []

    for unit in range(n_units):
        # Random firing rate within range
        rate = np.random.uniform(*rate_range)
        rates.append(rate)

        # Generate spike times as Poisson process
        n_spikes = np.random.poisson(rate * duration)
        times = np.sort(np.random.uniform(0, duration, n_spikes))
        spike_times[unit] = nap.Ts(t=times)

        # Assign random LFP channel
        lfp_channels.append(np.random.randint(0, 10))

    # Create TsGroup with metadata
    tsgroup = nap.TsGroup(spike_times)
    tsgroup.set_info(
        lfp_channel=np.array(lfp_channels),
        firing_rate=np.array(rates)
    )

    return tsgroup


def synthetic_lfp_trials(n_trials, fs, n_chans, pre=0.2, post=0.5, 
                         evoked_duration=0.2, seed=None):
    """
    Generate synthetic LFP epoch data suitable for evoked analysis.
    
    The background signal uses synthetic_lfp, with an evoked response added
    that varies in amplitude (not latency) across channels.

    Parameters
    ----------
    n_trials : int
        Number of trials.
    fs : float
        Sampling frequency in Hz.
    n_chans : int
        Number of channels.
    pre : float, optional
        Duration of baseline period before event (seconds). Default is 0.2.
    post : float, optional
        Duration of post-event period (seconds). Default is 0.5.
    evoked_duration : float, optional
        Duration of the evoked response (seconds). Default is 0.2.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    xarray.DataArray
        Synthetic LFP epoch data with proper coordinates and metadata.
        Dimensions: (time, channel, epoch)
        Coordinates: time, channel, epoch, ch_x, ch_y, ch_shank
        Attributes: fs, units
        
    Notes
    -----
    The evoked response amplitude varies across channels (simulating depth
    profile or source proximity) rather than latency. The evoked component
    is a damped oscillation that begins at time=0 (event onset).
    """
    if seed is not None:
        np.random.seed(seed)

    trial_duration = pre + post
    n_samples = int(trial_duration * fs)
    time = np.linspace(-pre, post, n_samples)
    
    # Time indices for the evoked period (from 0 to evoked_duration)
    evoked_mask = (time >= 0) & (time < evoked_duration)
    evoked_time = time[evoked_mask]
    
    # Create channel-dependent amplitude profile (varies smoothly across depth)
    # Simulates a dipole source at a particular depth
    source_channel = n_chans // 3  # Source near upper third
    ch_distances = np.abs(np.arange(n_chans) - source_channel)
    # Amplitude falls off with distance, with polarity reversal
    amplitude_profile = np.exp(-ch_distances / (n_chans / 3))
    # Add polarity reversal across the source
    polarity = np.ones(n_chans)
    polarity[np.arange(n_chans) > source_channel] = -1
    amplitude_profile = amplitude_profile * polarity
    
    # Create the evoked waveform template (damped oscillation, ~15 Hz)
    evoked_freq = 15  # Hz
    decay_tau = evoked_duration / 3
    evoked_template = (
        np.sin(2 * np.pi * evoked_freq * evoked_time) * 
        np.exp(-evoked_time / decay_tau) *
        (1 - np.exp(-evoked_time / 0.01))  # Smooth onset
    )
    
    # Generate data: background LFP + evoked response
    # Shape: (time, channel, epoch) - epoch is last dimension
    data = np.zeros((n_samples, n_chans, n_trials))
    
    for trial in range(n_trials):
        # Generate background using synthetic_lfp (without oscillations to keep it cleaner)
        background = synthetic_lfp(
            trial_duration, fs, n_chans, 
            exponent=-2, oscillations=False, seed=None
        )
        
        # Scale background noise
        noise_scale = 0.5
        data[:, :, trial] = background.values * noise_scale
        
        # Add evoked response with channel-dependent amplitude
        # Add some epoch-to-epoch variability in overall amplitude
        trial_amp_variation = np.random.uniform(0.8, 1.2)
        
        for ch in range(n_chans):
            # Channel-specific amplitude with some random variation
            ch_amp = amplitude_profile[ch] * trial_amp_variation
            ch_amp *= np.random.uniform(0.9, 1.1)  # Small channel jitter per trial
            
            data[evoked_mask, ch, trial] += evoked_template * ch_amp

    # Create channel coordinates
    ch_ids = np.arange(n_chans)
    ch_x = np.zeros(n_chans)
    ch_y = np.arange(n_chans) * 25.0
    ch_shank = np.zeros(n_chans, dtype=int)

    lfp_trials = xr.DataArray(
        data,
        dims=[DIM_TIME, DIM_CHAN, DIM_EPOCH],
        coords={
            DIM_TIME: time,
            DIM_CHAN: ch_ids,
            DIM_EPOCH: np.arange(n_trials),
            COORD_X: (DIM_CHAN, ch_x),
            COORD_Y: (DIM_CHAN, ch_y),
            COORD_SHANK: (DIM_CHAN, ch_shank),
        },
        attrs={
            ATTR_FS: fs,
            ATTR_UNITS: 'µV',
        }
    )

    return lfp_trials


def plot_synthetic_examples(save_path=None):
    """
    Generate and plot examples of all synthetic data types.
    
    This function creates visual examples of synthetic LFP, AP, spike,
    and trial data to verify the data generators are working correctly.
    Also tests the wave_map plotting function from mimo_pack.plot.map.
    
    Parameters
    ----------
    save_path : str or Path, optional
        If provided, saves the figure to this path. Otherwise displays it.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing all example plots.
    """
    import matplotlib.pyplot as plt
    from mimo_pack.plot.map import wave_map
    
    # Generate synthetic data
    lfp = synthetic_lfp(duration=2.0, fs=1000, n_chans=16, exponent=-2, 
                        oscillations=True, seed=42)
    ap = synthetic_ap(duration=0.2, fs=30000, n_chans=8, spike_rate=10.0, seed=42)
    spikes = synthetic_spikes(n_units=5, duration=10.0, seed=42)
    lfp_trials = synthetic_lfp_trials(n_trials=20, fs=1000, n_chans=16, 
                                       pre=0.2, post=0.5, evoked_duration=0.2, seed=42)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Synthetic Data Examples', fontsize=14, fontweight='bold')
    
    # Plot 1: LFP wave_map (using refactored wave_map function)
    ax = axes[0, 0]
    # Show full 2 second duration
    wave_map(lfp, x_scale=0.05, y_scale=50, ax=ax)
    ax.set_xlabel('Time (scaled) + X position')
    ax.set_ylabel('Y position (µm)')
    ax.set_title('LFP wave_map (2s)')
    
    # Plot 2: LFP power spectrum
    ax = axes[0, 1]
    from scipy import signal
    for i in range(min(3, lfp.sizes['channel'])):
        freqs, psd = signal.welch(lfp.isel(channel=i).values, fs=1000, nperseg=256)
        ax.loglog(freqs[1:], psd[1:], lw=1, alpha=0.7, label=f'Ch {i}')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')
    ax.set_title('LFP Power Spectrum (1/f)')
    ax.legend(loc='upper right', fontsize=8)
    
    # Plot 3: AP wave_map (using refactored wave_map function) - full 200ms
    ax = axes[0, 2]
    wave_map(ap, x_scale=0.0005, y_scale=100, ax=ax)
    ax.set_xlabel('Time (scaled) + X position')
    ax.set_ylabel('Y position (µm)')
    ax.set_title('AP wave_map (200ms)')
    
    # Plot 4: Spike raster
    ax = axes[1, 0]
    for unit_id in spikes.index:
        spike_times = spikes[unit_id].times()
        ax.vlines(spike_times, unit_id - 0.4, unit_id + 0.4, lw=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Unit')
    ax.set_title(f'Spike Raster ({len(spikes)} units)')
    ax.set_ylim(-0.5, len(spikes) - 0.5)
    
    # Plot 5: Spike rate histogram
    ax = axes[1, 1]
    rates = spikes.get_info('firing_rate')
    ax.bar(range(len(rates)), rates, color='steelblue', edgecolor='black')
    ax.set_xlabel('Unit')
    ax.set_ylabel('Firing Rate (Hz)')
    ax.set_title('Unit Firing Rates')
    
    # Plot 6: Evoked LFP wave_map (epoch-averaged)
    ax = axes[1, 2]
    trial_mean = lfp_trials.mean(dim='epoch')
    wave_map(trial_mean, x_scale=500, y_scale=100, ax=ax)
    ax.axvline(0, color='r', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel('Time (scaled) + X position')
    ax.set_ylabel('Y position (µm)')
    ax.set_title(f'Evoked LFP wave_map ({lfp_trials.sizes["epoch"]} epochs avg)')
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    return fig


def synthetic_iv_data(i_steps=None, fs=10000, step_dur=0.5, pre_dur=0.1, post_dur=0.1, 
                      v_rest=-70, r_input=100e6, tau=0.02, add_aps=True, seed=None):
    """
    Generate synthetic IV protocol data with realistic membrane dynamics.
    
    Parameters
    ----------
    i_steps : array_like, optional
        Current steps in pA. Default: -150 to +100 pA in 25 pA steps.
    fs : float, optional
        Sampling frequency in Hz. Default: 10000.
    step_dur : float, optional  
        Current step duration in seconds. Default: 0.5.
    pre_dur : float, optional
        Pre-step duration in seconds. Default: 0.1.
    post_dur : float, optional
        Post-step duration in seconds. Default: 0.1.
    v_rest : float, optional
        Resting membrane potential in mV. Default: -70.
    r_input : float, optional
        Input resistance in Ohms. Default: 100e6 (100 MΩ).
    tau : float, optional
        Membrane time constant in seconds. Default: 0.02 (20 ms).
    add_aps : bool, optional
        Whether to add action potentials to suprathreshold steps. Default: True.
    seed : int, optional
        Random seed for reproducible noise. Default: None.
        
    Returns
    -------
    xarray.DataArray
        IV protocol data with dimensions ['time', 'current_step'] and coordinates:
        - 'time': time axis in seconds
        - 'current_step': current step values in pA
        Attributes include 'fs', 'step_dur', 'units', 'add_aps'.
    """
    if i_steps is None:
        i_steps = np.arange(-150, 125, 25)  # -150 to +100 pA in 25 pA steps
    
    if seed is not None:
        np.random.seed(seed)
    
    i_steps = np.asarray(i_steps)
    n_trials = len(i_steps)
    
    total_dur = pre_dur + step_dur + post_dur
    n_samples = int(total_dur * fs)
    t = np.arange(n_samples) / fs
    
    step_start = int(pre_dur * fs)
    step_end = int((pre_dur + step_dur) * fs)
    
    v_traces = np.zeros((n_samples, n_trials))
    
    for i, i_step in enumerate(i_steps):
        # Baseline
        v_traces[:, i] = v_rest
        
        # Calculate steady-state voltage change (V = IR)
        dv_ss = i_step * 1e-12 * r_input * 1e3  # Convert to mV
        
        # Step response with exponential rise/decay
        for j in range(step_start, step_end):
            t_from_start = (j - step_start) / fs
            v_traces[j, i] = v_rest + dv_ss * (1 - np.exp(-t_from_start / tau))
        
        # For hyperpolarizing steps, add Ih-mediated sag
        if i_step < 0:
            sag_amplitude = 0.15 * abs(dv_ss)  # 15% sag recovery
            sag_tau = 0.15  # 150 ms sag time constant
            for j in range(step_start, step_end):
                t_from_start = (j - step_start) / fs
                sag_recovery = sag_amplitude * (1 - np.exp(-t_from_start / sag_tau))
                v_traces[j, i] += sag_recovery
        
        # For suprathreshold steps, add action potentials
        if add_aps and i_step > 25:  # Threshold around 25-50 pA
            ap_threshold = -55  # mV
            if v_traces[step_start + int(0.01 * fs), i] > ap_threshold:
                # Calculate firing rate (linear with current above threshold)
                excess_current = max(0, i_step - 25)
                base_rate = 10  # Hz for threshold current
                rate_slope = 0.2  # Hz per pA
                firing_rate = base_rate + rate_slope * excess_current
                
                # Generate spike times with adaptation
                n_expected_spikes = int(firing_rate * step_dur)
                if n_expected_spikes > 0:
                    # Initial ISI
                    initial_isi = 1.0 / firing_rate
                    adaptation_factor = 1.2  # ISIs get 20% longer
                    
                    spike_times = []
                    current_time = pre_dur + 0.005  # Start 5ms into step
                    isi = initial_isi
                    
                    while current_time < (pre_dur + step_dur - 0.01):  # Stop 10ms before step end
                        spike_times.append(current_time)
                        current_time += isi
                        isi *= adaptation_factor  # Spike frequency adaptation
                        if isi > 0.2:  # Max ISI of 200ms
                            break
                    
                    # Add realistic triphasic spike waveforms
                    for spike_time in spike_times:
                        spike_idx = int(spike_time * fs)
                        if spike_idx < n_samples - 30:  # Ensure we don't go out of bounds
                            # Triphasic spike: depolarization -> repolarization -> afterhyperpolarization
                            spike_template = np.array([
                                0, 2, 8, 20, 40, 60, 80, 95, 100, 95, 80, 60, 40, 20, 0,
                                -10, -20, -25, -20, -15, -10, -8, -6, -4, -3, -2, -1, 0, 0, 0
                            ])
                            
                            # Scale and add to trace
                            spike_amplitude = 125 - v_rest  # Peak around +55 mV
                            scaled_spike = spike_template * (spike_amplitude / 100)
                            
                            end_idx = min(spike_idx + len(scaled_spike), n_samples)
                            template_end = end_idx - spike_idx
                            v_traces[spike_idx:end_idx, i] += scaled_spike[:template_end]
        
        # Post-step relaxation  
        v_at_step_end = v_traces[step_end - 1, i]
        for j in range(step_end, n_samples):
            t_from_end = (j - step_end) / fs
            v_traces[j, i] = v_rest + (v_at_step_end - v_rest) * np.exp(-t_from_end / tau)
    
    # Add realistic background noise
    noise_std = 0.8  # 0.8 mV RMS noise
    v_traces += np.random.randn(n_samples, n_trials) * noise_std
    
    # Create xarray DataArray
    iv_data = xr.DataArray(
        v_traces,
        dims=['time', 'current_step'],
        coords={
            'time': t,
            'current_step': i_steps,
        },
        attrs={
            'fs': fs,
            'step_dur': step_dur, 
            'pre_dur': pre_dur,
            'post_dur': post_dur,
            'units': 'mV',
            'add_aps': add_aps
        }
    )
    
    return iv_data


def synthetic_nidq_digital(duration, fs, channel_names, pulse_times=None,
                           pulse_duration=0.01, seed=None):
    """
    Generate synthetic NIDQ digital (TTL) channel data.

    Each channel contains a binary 0/1 signal with square pulses at specified
    (or randomly generated) onset times. Suitable for testing ``digital_to_events``,
    ``digital_to_intervals``, and ``pulse_stats``.

    Parameters
    ----------
    duration : float
        Duration of the signal in seconds.
    fs : float
        Sampling rate in Hz.
    channel_names : list of str
        Label for each digital channel (e.g. ``['laser', 'sync']``).  These
        become the values of the ``channel`` coordinate, matching the convention
        used by ``load_nidq`` when ``chan_names`` is supplied.
    pulse_times : list of array_like or None, optional
        Pulse onset times (seconds) for each channel.  Must have the same
        length as ``channel_names``.  Each element is an array of onset
        timestamps.  If ``None``, 5 random pulses per channel are generated.
    pulse_duration : float, optional
        ON duration of each pulse in seconds.  Default is 0.01 (10 ms).
    seed : int, optional
        Random seed for reproducibility when ``pulse_times`` is ``None``.

    Returns
    -------
    xarray.DataArray
        Dimensions: ``(time, channel)``
        Dtype: float32 (values are 0.0 or 1.0)
        Coordinates: ``time``, ``channel`` (string labels), ``chan_type``
        (all ``'digital'``).
        Attributes: ``fs``, ``units='V'``, ``description``.

    Examples
    --------
    >>> da = synthetic_nidq_digital(1.0, 1000, ['ttl_a'], [[0.1, 0.5, 0.9]])
    >>> da.sel(channel='ttl_a').values.max()
    1.0
    """
    if seed is not None:
        np.random.seed(seed)

    n_chans = len(channel_names)
    n_samples = int(duration * fs)
    time = np.arange(n_samples) / fs

    if pulse_times is None:
        pulse_times = [
            np.sort(np.random.uniform(0, duration - pulse_duration, 5))
            for _ in range(n_chans)
        ]

    if len(pulse_times) != n_chans:
        raise ValueError(
            f"pulse_times has {len(pulse_times)} entries but channel_names has {n_chans}."
        )

    data = np.zeros((n_samples, n_chans), dtype=np.float32)
    pulse_samples = int(pulse_duration * fs)

    for ch_idx, onsets in enumerate(pulse_times):
        for onset in np.asarray(onsets):
            start = int(onset * fs)
            end = min(start + pulse_samples, n_samples)
            data[start:end, ch_idx] = 1.0

    return xr.DataArray(
        data,
        dims=[DIM_TIME, DIM_CHAN],
        coords={
            DIM_TIME: time,
            DIM_CHAN: channel_names,
            'chan_type': (DIM_CHAN, ['digital'] * n_chans),
        },
        attrs={
            ATTR_FS: float(fs),
            ATTR_UNITS: 'V',
            'description': 'Synthetic NIDQ digital data',
        },
    )


def synthetic_nidq_analog(duration, fs, channel_names, pulse_times=None,
                          pulse_duration=0.01, pulse_amplitude=1.0,
                          baseline=0.0, noise_std=0.05, seed=None):
    """
    Generate synthetic NIDQ analog channel data.

    Each channel contains a continuous voltage signal with optional square
    pulses (above a flat baseline) plus Gaussian noise.  Suitable for testing
    ``analog_to_events``, ``analog_to_intervals``, ``binarize``, and
    ``quantize``.

    Parameters
    ----------
    duration : float
        Duration of the signal in seconds.
    fs : float
        Sampling rate in Hz.
    channel_names : list of str
        Label for each analog channel.
    pulse_times : list of array_like or None, optional
        Pulse onset times (seconds) for each channel.  Same length as
        ``channel_names``.  If ``None``, 5 random pulses per channel are
        generated.
    pulse_duration : float, optional
        ON duration of each pulse in seconds.  Default is 0.01 (10 ms).
    pulse_amplitude : float, optional
        Amplitude of each pulse above the baseline (volts).  Default is 1.0.
    baseline : float, optional
        Baseline voltage level (volts).  Default is 0.0.
    noise_std : float, optional
        Standard deviation of additive Gaussian noise (volts).  Default is
        0.05, which is well below ``pulse_amplitude`` so pulses are cleanly
        detectable with a threshold at ``pulse_amplitude / 2``.  Set to 0 for
        noiseless signals (useful for exact edge-count assertions).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    xarray.DataArray
        Dimensions: ``(time, channel)``
        Dtype: float32
        Coordinates: ``time``, ``channel`` (string labels), ``chan_type``
        (all ``'analog'``).
        Attributes: ``fs``, ``units='V'``, ``description``.

    Examples
    --------
    >>> da = synthetic_nidq_analog(1.0, 1000, ['photo'], [[0.2, 0.6]],
    ...                            noise_std=0.0)
    >>> float(da.sel(channel='photo', time=0.205, method='nearest'))
    1.0
    """
    if seed is not None:
        np.random.seed(seed)

    n_chans = len(channel_names)
    n_samples = int(duration * fs)
    time = np.arange(n_samples) / fs

    if pulse_times is None:
        pulse_times = [
            np.sort(np.random.uniform(0, duration - pulse_duration, 5))
            for _ in range(n_chans)
        ]

    if len(pulse_times) != n_chans:
        raise ValueError(
            f"pulse_times has {len(pulse_times)} entries but channel_names has {n_chans}."
        )

    data = np.full((n_samples, n_chans), baseline, dtype=np.float32)
    pulse_samples = int(pulse_duration * fs)

    for ch_idx, onsets in enumerate(pulse_times):
        for onset in np.asarray(onsets):
            start = int(onset * fs)
            end = min(start + pulse_samples, n_samples)
            data[start:end, ch_idx] += pulse_amplitude

    if noise_std > 0:
        data += np.random.randn(n_samples, n_chans).astype(np.float32) * noise_std

    return xr.DataArray(
        data,
        dims=[DIM_TIME, DIM_CHAN],
        coords={
            DIM_TIME: time,
            DIM_CHAN: channel_names,
            'chan_type': (DIM_CHAN, ['analog'] * n_chans),
        },
        attrs={
            ATTR_FS: float(fs),
            ATTR_UNITS: 'V',
            'description': 'Synthetic NIDQ analog data',
        },
    )


def synthetic_nidq(duration, fs, digital_channels=None, analog_channels=None,
                   pulse_duration=0.01, noise_std=0.05, seed=None):
    """
    Generate a combined synthetic NIDQ DataArray with both digital and analog
    channels.

    Concatenates digital (0/1) and analog (continuous) channels into a single
    ``xr.DataArray`` with a mixed ``chan_type`` coordinate, matching the
    structure produced by ``load_nidq(..., expand_digital=True)``.  Suitable
    for testing ``extract_nidq_events`` and other session-level functions.

    Parameters
    ----------
    duration : float
        Duration of the signal in seconds.
    fs : float
        Sampling rate in Hz.
    digital_channels : dict of {str: array_like} or None, optional
        Mapping from channel label to pulse onset times (seconds).  If
        ``None``, two digital channels named ``'ttl_a'`` and ``'ttl_b'`` are
        created with random pulses.
    analog_channels : dict of {str: array_like} or None, optional
        Mapping from channel label to pulse onset times (seconds).  If
        ``None``, one analog channel named ``'photodiode'`` is created with
        random pulses (``pulse_amplitude=1.0``, ``noise_std=noise_std``).
    pulse_duration : float, optional
        ON duration of each pulse in seconds.  Default is 0.01.
    noise_std : float, optional
        Noise added to analog channels only.  Set to 0 for noiseless analog
        signals.  Default is 0.05.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    xarray.DataArray
        Dimensions: ``(time, channel)``
        All digital channels are float32 0/1; all analog channels are float32
        continuous.  The ``channel`` coordinate contains string labels; the
        ``chan_type`` coordinate is ``'digital'`` or ``'analog'`` per channel.
        Attributes: ``fs``, ``units='V'``, ``description``.

    Examples
    --------
    >>> da = synthetic_nidq(2.0, 1000, seed=0)
    >>> list(da.coords['chan_type'].values)
    ['digital', 'digital', 'analog']
    """
    if seed is not None:
        np.random.seed(seed)

    if digital_channels is None:
        digital_channels = {
            'ttl_a': np.sort(np.random.uniform(0, duration - pulse_duration, 5)),
            'ttl_b': np.sort(np.random.uniform(0, duration - pulse_duration, 3)),
        }
    if analog_channels is None:
        analog_channels = {
            'photodiode': np.sort(np.random.uniform(0, duration - pulse_duration, 5)),
        }

    parts = []

    if digital_channels:
        d_names = list(digital_channels.keys())
        d_times = list(digital_channels.values())
        parts.append(
            synthetic_nidq_digital(
                duration, fs, d_names, d_times,
                pulse_duration=pulse_duration, seed=None,
            )
        )

    if analog_channels:
        a_names = list(analog_channels.keys())
        a_times = list(analog_channels.values())
        parts.append(
            synthetic_nidq_analog(
                duration, fs, a_names, a_times,
                pulse_duration=pulse_duration, noise_std=noise_std, seed=None,
            )
        )

    combined = xr.concat(parts, dim=DIM_CHAN)
    combined.attrs.update({
        ATTR_FS: float(fs),
        ATTR_UNITS: 'V',
        'description': 'Synthetic NIDQ data',
    })
    return combined


def synthetic_fiber_photometry(duration, fs, decay=True, seed=None):
    """
    Generate synthetic fiber photometry signal and isosbestic control data.

    Combines biexponential photobleaching, slow drift, an
    isosbestic-correlated motion artifact, injected transient events (signal
    channel only), and independent noise on each channel.

    Parameters
    ----------
    duration : float
        Duration of the recording in seconds.
    fs : float
        Sampling frequency in Hz.
    decay : bool, optional
        Whether to add biexponential photobleaching decay to the baseline.
        Default is True.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    signal : xr.DataArray
        Synthetic fiber photometry signal channel. Dimensions: (time,).
        Attributes: fs, units, description.
    isosbestic : xr.DataArray
        Synthetic isosbestic control channel, sharing the drift and motion
        artifact with `signal` but without the injected transient events.
        Dimensions: (time,). Attributes: fs, units, description.
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs

    if decay:
        bleach = 2.0 * np.exp(-t / (duration * 0.6)) + 0.5 * np.exp(-t / (duration * 3))
    else:
        bleach = np.zeros(n_samples)
    baseline = 5.0 + bleach

    drift = 0.2 * np.sin(2 * np.pi * 0.01 * t + np.random.uniform(0, 2 * np.pi))

    motion = 0.15 * np.sin(2 * np.pi * 0.3 * t) + 0.05 * np.random.randn(n_samples)
    smooth_kernel = np.ones(5) / 5
    motion = np.convolve(motion, smooth_kernel, mode='same')

    transients = np.zeros(n_samples)
    n_events = max(1, int(duration / 5))
    for _ in range(n_events):
        rise_tau = 0.05
        decay_tau = 0.5
        event_dur = min(2.0, duration)
        onset = np.random.uniform(0, max(0.01, duration - event_dur))
        onset_idx = int(onset * fs)
        te = np.arange(int(event_dur * fs)) / fs
        amp = np.random.uniform(0.5, 2.0)
        event = amp * (1 - np.exp(-te / rise_tau)) * np.exp(-te / decay_tau)
        end_idx = min(onset_idx + len(event), n_samples)
        transients[onset_idx:end_idx] += event[:end_idx - onset_idx]

    signal_vals = baseline + drift + motion + transients + 0.05 * np.random.randn(n_samples)
    isosbestic_vals = 0.9 * baseline + drift + 0.8 * motion + 0.05 * np.random.randn(n_samples)

    signal = xr.DataArray(
        signal_vals,
        dims=[DIM_TIME],
        coords={DIM_TIME: t},
        attrs={
            ATTR_FS: float(fs),
            ATTR_UNITS: 'a.u.',
            'description': 'Synthetic fiber photometry signal',
        }
    )
    isosbestic = xr.DataArray(
        isosbestic_vals,
        dims=[DIM_TIME],
        coords={DIM_TIME: t},
        attrs={
            ATTR_FS: float(fs),
            ATTR_UNITS: 'a.u.',
            'description': 'Synthetic fiber photometry isosbestic control',
        }
    )

    return signal, isosbestic


if __name__ == '__main__':
    # Run the test when executed directly
    import os
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'synthetic_data_examples.pdf')
    
    print("Generating synthetic data examples...")
    fig = plot_synthetic_examples(save_path=output_path)
    print("Done!")

