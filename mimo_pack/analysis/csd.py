# Current source density analyses
import numpy as np
import xarray as xr

from mimo_pack.consts import DIM_TIME, DIM_CHAN, COORD_Y, ATTR_FS, ATTR_UNITS


def csd(lfp, coord=COORD_Y, conductivity=1.0):
    """
    Calculate 1D current source density using the Vaknin et al. (1988) method.
    
    Uses finite differences with boundary extrapolation to compute the second
    spatial derivative of the local field potential, which approximates the
    current source density under assumptions of volume conduction.
    
    Parameters
    ----------
    lfp : xarray.DataArray
        LFP data with proper spatial coordinates. Expected dimensions include
        time and channel with spatial coordinate information.
    coord : str, optional
        Name of the spatial coordinate for CSD calculation. Default: 'ch_y'.
    conductivity : float, optional
        Tissue conductivity (sigma_z) in S/m. Default: 1.0.
        
    Returns
    -------
    xarray.DataArray
        CSD data with same dimensions and coordinates as input LFP.
        Units are typically µA/mm² if LFP is in µV and spacing is in µm.
        
    Raises
    ------
    TypeError
        If input is not an xarray.DataArray.
    ValueError
        If specified coordinate is not found in the input data.
        
    Notes
    -----
    The CSD is calculated as: CSD = -σ * ∂²φ/∂z²
    where φ is the LFP and σ is the tissue conductivity.
    
    Boundary conditions use edge padding (repeating first/last values)
    as recommended by Vaknin et al. (1988).
    """
    
    # Validate inputs
    if not isinstance(lfp, xr.DataArray):
        raise TypeError(
            f"Input must be an xarray.DataArray, got {type(lfp).__name__}. "
            "Use fileio functions to load data into xarray format."
        )
        
    if coord not in lfp.coords:
        raise ValueError(f"Coordinate '{coord}' not found in LFP data. "
                        f"Available coordinates: {list(lfp.coords.keys())}")
    
    # Identify the axis index for spatial coordinate
    coord_dim = None
    for dim in lfp.dims:
        if coord in lfp.coords and dim in lfp.coords[coord].dims:
            coord_dim = dim
            break
    
    if coord_dim is None:
        raise ValueError(f"Could not find dimension associated with coordinate '{coord}'")
    
    grad_dim = lfp.dims.index(coord_dim)
    
    # Pad the data array with edge values (Vaknin et al. 1988 method)
    pad_width = [(0, 0)] * lfp.values.ndim
    pad_width[grad_dim] = (1, 1)  # Pad one element on both sides
    
    phi_padded = np.pad(lfp.values, pad_width, mode='edge')

    # Calculate second spatial derivative using finite differences
    spacing = float(lfp.coords[coord].diff(coord_dim).mean().item())  # Average spacing
    first_deriv = np.gradient(phi_padded, spacing, axis=grad_dim)
    second_deriv = np.gradient(first_deriv, spacing, axis=grad_dim)
    
    # Remove padding to return to original size
    slices = [slice(None)] * lfp.values.ndim
    slices[grad_dim] = slice(1, -1)
    csd_values = second_deriv[tuple(slices)]

    # Apply CSD formula: CSD = -σ * ∂²φ/∂z²
    csd_values = -1 * conductivity * csd_values

    # Create output xarray with updated attributes
    attrs = lfp.attrs.copy()
    attrs['units'] = 'µA/mm²'  # Typical CSD units
    if 'description' not in attrs:
        attrs['description'] = f'Current source density calculated along {coord}'
    
    csd_result = xr.DataArray(
        csd_values,
        coords=lfp.coords,
        dims=lfp.dims,
        attrs=attrs
    )
    
    return csd_result


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mimo_pack.testing.data import synthetic_lfp_trials
    from mimo_pack.plot.map import wave_map
    
    print("=" * 60)
    print("CURRENT SOURCE DENSITY ANALYSIS TESTS")
    print("=" * 60)
    
    # ========== Test 1: Generate synthetic LFP data ==========
    print("\n1. Generating synthetic LFP epoch data...")
    try:
        # Generate synthetic evoked LFP with clear current sources/sinks
        lfp_data = synthetic_lfp_trials(
            n_trials=50,
            fs=1000,
            n_chans=32,
            pre=0.1,
            post=0.4,
            evoked_duration=0.15,
            seed=42
        )
        
        # Take epoch average for cleaner CSD
        lfp_avg = lfp_data.mean(dim='epoch')
        # Preserve attributes from original data
        lfp_avg.attrs = lfp_data.attrs.copy()
        
        print(f"   + Generated LFP data: shape={lfp_avg.shape}")
        print(f"   + Channels: {lfp_avg.sizes['channel']} (spacing: {float(lfp_avg.coords['ch_y'].diff('channel').mean()):.1f} µm)")
        print(f"   + Duration: {(lfp_avg.sizes['time']/lfp_avg.attrs['fs'])*1000:.0f} ms")
        print(f"   + Sampling rate: {lfp_avg.attrs['fs']} Hz")
    except Exception as e:
        print(f"   - Failed: {e}")
        raise
    
    # ========== Test 2: Calculate CSD ==========
    print("\n2. Computing current source density...")
    try:
        # Calculate CSD using default parameters
        csd_result = csd(lfp_avg, conductivity=0.4)  # Typical brain conductivity
        
        print(f"   + CSD computed: shape={csd_result.shape}")
        print(f"   + Units: {csd_result.attrs['units']}")
        print(f"   + Min CSD: {float(csd_result.min()):.3f} {csd_result.attrs['units']}")
        print(f"   + Max CSD: {float(csd_result.max()):.3f} {csd_result.attrs['units']}")
        
        # Check that CSD preserves coordinates and dimensions
        assert csd_result.dims == lfp_avg.dims, "CSD dimensions don't match LFP"
        assert COORD_Y in csd_result.coords, "Spatial coordinate missing from CSD"
    except Exception as e:
        print(f"   - Failed: {e}")
        raise
    
    # ========== Test 3: Test different conductivity values ==========
    print("\n3. Testing conductivity parameter...")
    try:
        conductivities = [0.2, 0.4, 0.8]  # Range of typical tissue conductivities
        csd_results = {}
        
        for sigma in conductivities:
            csd_results[sigma] = csd(lfp_avg, conductivity=sigma)
        
        # Check that CSD scales linearly with conductivity
        ratio_expected = conductivities[1] / conductivities[0]
        ratio_actual = float(csd_results[conductivities[1]].mean() / csd_results[conductivities[0]].mean())
        
        print(f"   + Tested {len(conductivities)} conductivity values")
        print(f"   + Conductivity scaling test: expected={ratio_expected:.1f}, actual={ratio_actual:.1f}")
        
        assert abs(ratio_actual - ratio_expected) < 0.01, "CSD doesn't scale properly with conductivity"
        print("   + Conductivity scaling verified")
    except Exception as e:
        print(f"   - Failed: {e}")
        raise
    
    # ========== Test 4: Error handling ==========
    print("\n4. Testing error handling...")
    try:
        # Test missing coordinate
        try:
            csd(lfp_avg, coord='nonexistent_coord')
            print("   - Error: Should have raised ValueError for missing coordinate")
        except ValueError as ve:
            print(f"   + Correctly caught missing coordinate error: {str(ve)[:60]}...")
        
        # Test with minimal data
        minimal_lfp = lfp_avg.isel(channel=slice(0, 3))  # Only 3 channels
        csd_minimal = csd(minimal_lfp)
        assert csd_minimal.sizes['channel'] == 3, "CSD size doesn't match input"
        print("   + Minimal data test passed")
        
    except Exception as e:
        print(f"   - Failed: {e}")
    
    # ========== Test 5: Visualization ==========
    print("\n5. Creating CSD visualization...")
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        
        # Plot 1: Original LFP
        ax = axes[0]
        wave_map(lfp_avg, x_scale=200, y_scale=50, ax=ax, color='blue')
        ax.axvline(0, color='r', ls='--', lw=1, alpha=0.7)
        ax.set_title('LFP (Epoch Average)')
        ax.set_xlabel('Time + X offset')
        ax.set_ylabel('Channel Y position (µm)')
        
        # Plot 2: CSD
        ax = axes[1] 
        wave_map(csd_result, x_scale=200, y_scale=50, ax=ax, color='black')
        ax.axvline(0, color='r', ls='--', lw=1, alpha=0.7)
        ax.set_title('Current Source Density')
        ax.set_xlabel('Time + X offset')
        ax.set_ylabel('Channel Y position (µm)')
        
        # Plot 3: Time course at peak channel
        ax = axes[2]
        t_ms = lfp_avg.time.values * 1000
        
        # Find channel with strongest evoked response
        evoked_mask = (lfp_avg.time >= 0) & (lfp_avg.time <= 0.1)
        peak_ch = int(lfp_avg.sel(time=evoked_mask).std(dim='time').argmax())
        
        # Plot LFP and CSD time courses
        ax.plot(t_ms, lfp_avg.isel(channel=peak_ch), 'b-', lw=1.5, label=f'LFP (ch {peak_ch})')
        ax2 = ax.twinx()
        ax2.plot(t_ms, csd_result.isel(channel=peak_ch), 'k-', lw=1.5, label=f'CSD (ch {peak_ch})')
        
        ax.axvline(0, color='r', ls='--', lw=1, alpha=0.7)
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('LFP (µV)', color='b')
        ax2.set_ylabel('CSD (µA/mm²)', color='k')
        ax.set_title(f'Time Course (Channel {peak_ch})')
        
        # Add legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        plt.savefig('csd_analysis_tests.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("   + Visualization saved to csd_analysis_tests.png")
        
    except Exception as e:
        print(f"   - Visualization failed: {e}")
    
    # ========== Summary ==========
    print("\n" + "=" * 60)
    print("CSD ANALYSIS TESTS COMPLETE")
    print(f"Peak CSD magnitude: {float(abs(csd_result).max()):.3f} {csd_result.attrs['units']}")
    print(f"CSD range: [{float(csd_result.min()):.3f}, {float(csd_result.max()):.3f}] {csd_result.attrs['units']}")
    print("=" * 60)
