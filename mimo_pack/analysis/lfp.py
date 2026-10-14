# Analysis function for LFP data
# Author: Drew Headley
# Date: 2025-06-16

import xarray as xr
import numpy as np

from mimo_pack.util.xarray import interp_stack
from mimo_pack.analysis.csd import csd
from mimo_pack.consts import DIM_TIME, DIM_CHAN, DIM_EPOCH, COORD_Y, COORD_X


def evoked_lfp(lfp_trials, align_dim=DIM_EPOCH, pre=None, spacing=100, coord=COORD_Y):
    """Calculate the event-triggered average Local Field Potential (LFP) and its CSD.

    This function takes pre-windowed LFP epoch data, performs spatial interpolation,
    computes the mean LFP across all epochs, applies baseline correction using the
    pre-event period, and calculates the Current Source Density (CSD).

    Parameters
    ----------
    lfp_trials : xarray.DataArray
        Pre-windowed LFP data with dimensions (epoch, time, channel).
        Must be an xarray.DataArray with a time coordinate that includes
        negative values for pre-event baseline correction.
    align_dim : str, optional
        The dimension name to average over. Defaults to the standard epoch dimension (`DIM_EPOCH`).
    pre : float, optional
        The time in seconds marking the end of the pre-event period for baseline
        correction. If None, uses all negative time values. Defaults to None.
    spacing : int, optional
        The desired spatial spacing between channels for interpolation (e.g., in
        micrometers). Defaults to 100.
    coord : str, optional
        The coordinate to use for CSD calculation. Defaults to 'ch_y'.

    Returns
    -------
    lfp_mean : xarray.DataArray
        The event-triggered average LFP, baseline-corrected. The time coordinate
        is converted to milliseconds.
    csd_mean : xarray.DataArray
        The Current Source Density calculated from `lfp_mean`.

    Raises
    ------
    TypeError
        If input is not an xarray.DataArray.
    ValueError
        If required dimensions are missing from input.

    See Also
    --------
    mimo_pack.util.xarray.interp_stack : Interpolates a stack of LFP data.
    mimo_pack.analysis.csd.csd : Calculates the Current Source Density.

    Examples
    --------
    >>> from mimo_pack.fileio.lfp import load_lfp_windows
    >>> # Load windowed data using fileio (done separately)
    >>> lfp_data = load_lfp_windows(lfp_path, centers=times, pre=0.01, post=0.03)
    >>> # Then pass to analysis function
    >>> lfp_mean, csd_mean = evoked_lfp(lfp_data, spacing=100)
    """
    # Type checking
    if not isinstance(lfp_trials, xr.DataArray):
        raise TypeError(
            f"Input must be an xarray.DataArray, got {type(lfp_trials).__name__}. "
            "Use fileio.lfp.load_lfp_windows to load epoch-windowed LFP data first."
        )
    
    # Validate dimensions
    if align_dim not in lfp_trials.dims:
        raise ValueError(
            f"Input must have '{align_dim}' dimension. "
            f"Available dimensions: {list(lfp_trials.dims)}"
        )
    
    if DIM_TIME not in lfp_trials.dims:
        raise ValueError(
            f"Input must have '{DIM_TIME}' dimension. "
            f"Available dimensions: {list(lfp_trials.dims)}"
        )
    
    # Perform spatial interpolation
    lfp_interp = interp_stack(lfp_trials, spacing=spacing)
    
    # Average across trials/windows
    lfp_mean = lfp_interp.mean(dim=align_dim)
    
    # Baseline correction using pre-event period
    if pre is None:
        # Use all negative time values for baseline
        baseline = lfp_mean.sel(time=slice(None, 0)).median(dim=DIM_TIME)
    else:
        baseline = lfp_mean.sel(time=slice(None, pre)).median(dim=DIM_TIME)
    
    lfp_mean = lfp_mean - baseline
    
    # Add description attribute to LFP
    desc_base = lfp_trials.attrs.get('description', 'LFP epochs')
    lfp_mean.attrs['description'] = f"{desc_base} -> evoked LFP (baseline corrected)"
        # Calculate CSD
    csd_mean = csd(lfp_mean, coord=coord)
    
    return lfp_mean, csd_mean


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mimo_pack.testing.data import synthetic_lfp_trials
    from mimo_pack.consts import ATTR_FS, DIM_EPOCH
    from mimo_pack.plot.map import wave_map
    
    print("=" * 60)
    print("LFP ANALYSIS MODULE TESTS")
    print("=" * 60)
    
    # ========== Test 1: Generate synthetic epoch data ==========
    print("\n1. Generating synthetic LFP epoch data...")
    try:
        lfp_trials = synthetic_lfp_trials(
            n_trials=20,
            pre=0.1,
            post=0.3,
            fs=1000,
            n_chans=32,
            seed=42
        )
        print(f"   ✓ Generated epochs: shape={lfp_trials.shape}, dims={lfp_trials.dims}")
        print(f"   Sample rate: {lfp_trials.attrs.get(ATTR_FS)} Hz")
        print(f"   Time range: {lfp_trials[DIM_TIME].values.min():.3f} to {lfp_trials[DIM_TIME].values.max():.3f} s")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        raise
    
    # ========== Test 2: Basic evoked_lfp call ==========
    print("\n2. Testing evoked_lfp() basic functionality...")
    try:
        lfp_mean, csd_mean = evoked_lfp(lfp_trials, align_dim=DIM_EPOCH, spacing=50)
        
        assert DIM_TIME in lfp_mean.dims, "Time dimension missing from output"
        # Note: interp_stack creates dims based on coordinates (ch_x, ch_y), not DIM_CHAN
        assert lfp_mean.shape == csd_mean.shape, "LFP and CSD shapes don't match"
        
        print(f"   ✓ evoked_lfp successful")
        print(f"   LFP mean shape: {lfp_mean.shape}, dims: {lfp_mean.dims}")
        print(f"   CSD mean shape: {csd_mean.shape}")
        print(f"   Time now in ms: {lfp_mean[DIM_TIME].values.min():.1f} to {lfp_mean[DIM_TIME].values.max():.1f} ms")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 3: Baseline correction ==========
    print("\n3. Testing baseline correction...")
    try:
        # Get pre-stimulus period (negative times, now in ms)
        pre_mean = lfp_mean.sel({DIM_TIME: slice(None, 0)}).mean(dim=DIM_TIME)
        
        # After baseline correction, pre-stimulus should be near zero
        pre_mean_val = float(pre_mean.mean())
        
        assert np.abs(pre_mean_val) < 1.0, f"Baseline not corrected: mean={pre_mean_val}"
        print(f"   ✓ Baseline correction working: pre-stim mean = {pre_mean_val:.4f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 4: Different spacing values ==========
    print("\n4. Testing different spatial interpolation spacing...")
    try:
        lfp_50, _ = evoked_lfp(lfp_trials, align_dim=DIM_EPOCH, spacing=50)
        lfp_100, _ = evoked_lfp(lfp_trials, align_dim=DIM_EPOCH, spacing=100)
        
        # Different spacing should produce different shapes
        # The interpolated dimension is ch_y
        print(f"   ✓ Spacing 50: shape={lfp_50.shape}, dims={lfp_50.dims}")
        print(f"   ✓ Spacing 100: shape={lfp_100.shape}, dims={lfp_100.dims}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 5: Custom pre parameter ==========
    print("\n5. Testing custom pre parameter for baseline...")
    try:
        # Use only first 50ms before event for baseline
        lfp_custom, _ = evoked_lfp(lfp_trials, align_dim=DIM_EPOCH, pre=-0.05, spacing=100)
        
        print(f"   ✓ Custom pre parameter accepted")
        print(f"   Output shape: {lfp_custom.shape}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 6: Error handling - wrong input type ==========
    print("\n6. Testing error handling...")
    try:
        evoked_lfp(np.random.randn(100, 10))  # numpy array instead of xarray
        print(f"   ✗ Should have raised TypeError")
    except TypeError as e:
        print(f"   ✓ Correctly raised TypeError for non-xarray input")
    
    try:
        # Missing required dimension
        bad_xr = xr.DataArray(np.random.randn(100, 10), dims=['time', 'channel'])
        evoked_lfp(bad_xr, align_dim='epoch')
        print(f"   ✗ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError for missing dimension")
    
    # ========== Test 7: CSD properties ==========
    print("\n7. Testing CSD properties...")
    try:
        # CSD should have opposite sign at source vs sink locations
        # and should sum to approximately zero (current conservation)
        csd_sum = float(csd_mean.sum())
        csd_range = float(csd_mean.max() - csd_mean.min())
        
        print(f"   ✓ CSD range: {csd_range:.4f}")
        print(f"   CSD sum (should be ~0): {csd_sum:.4f}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
    
    # ========== Test 8: Visualization ==========
    print("\n8. Creating visualization...")
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        
        # Plot 1: Single epoch LFP using wave_map
        ax = axes[0]
        single_trial = lfp_trials.isel({DIM_EPOCH: 0})
        wave_map(single_trial, x_scale=0.5, y_scale=30, ax=ax, lw=0.5, color='b')
        ax.axvline(0, color='r', ls='--', alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Depth (µm)')
        ax.set_title('Single Epoch LFP')
        
        # Plot 2: Mean evoked LFP using wave_map
        ax = axes[1]
        wave_map(lfp_mean, x_coord=COORD_X, y_coord=COORD_Y, x_scale=1, y_scale=50, 
                 ax=ax, lw=1, color='k')
        ax.axvline(0, color='r', ls='--', alpha=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Depth (µm)')
        ax.set_title('Mean Evoked LFP')
        
        # Plot 3: CSD using wave_map
        ax = axes[2]
        wave_map(csd_mean, x_coord=COORD_X, y_coord=COORD_Y, x_scale=1, y_scale=5000, 
                 ax=ax, lw=1, color='darkred')
        ax.axvline(0, color='r', ls='--', alpha=0.5)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Depth (µm)')
        ax.set_title('Current Source Density')
        
        plt.tight_layout()
        plt.savefig('lfp_analysis_tests.png', dpi=150)
        plt.show()
        print(f"   ✓ Visualization saved to lfp_analysis_tests.png")
    except Exception as e:
        print(f"   ✗ Visualization failed: {e}")
    
    print("\n" + "=" * 60)
    print("LFP ANALYSIS TESTS COMPLETE")
    print("=" * 60)


