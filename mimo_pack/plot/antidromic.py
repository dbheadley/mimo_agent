# Antidromic plots

from mimo_pack.plot.xarray import composite_subplots, pcolormesh_subplots
from mimo_pack.consts import DIM_TIME, COORD_X, COORD_Y, COORD_SHANK
import matplotlib.pyplot as plt
import numpy as np

def plot_antidromic_summary(summary_data, current_uA, region, fig=None):
    """Plots the antidromic response summary including AP envelope, LFP, and CSD.
    This function generates a figure with three main sections, one for each data
    type: Action Potential (AP) envelope, Local Field Potential (LFP), and
    Current Source Density (CSD). Each section is further divided into subplots
    based on the shank of the recording probe, as defined in the input DataArrays.
    Parameters
    ----------
    summary_data : dict
        A dictionary containing the summary data. It must have the keys 'ap',
        'lfp', and 'csd', where each value is an xarray DataArray containing
        the corresponding data.
    current_uA : int or float
        The stimulation current in microamperes (uA) to be displayed in the
        figure's main title.
    region : str
        The brain region stimulated (e.g., 'AC', 'M1'), which will be included
        in the figure's main title.
    fig : matplotlib.figure.Figure, optional
        An existing matplotlib Figure object to plot on. If None, a new
        figure is created. Defaults to None.
    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure object containing the generated summary plots.
    Notes
    -----
    - The function relies on the custom plotting functions `composite_xr` and
      `pcolormesh_xr`, which are expected to handle xarray DataArrays.
    - The input DataArrays in `summary_data` are expected to have coordinates
      like 'ch_shank', 'ch_x', 'ch_y', and 'time' for proper plotting.
    - The layout is managed using `fig.subfigures` and `layout='constrained'`
      to organize the plots for AP, LFP, and CSD vertically and prevent titles
      from overlapping.
    """
 
    ap_env = summary_data['ap']
    lfp_mean = summary_data['lfp']
    csd_mean = summary_data['csd']

    def _n_shanks(xr_obj):
        """Return the number of unique shanks in a DataArray.

        Parameters
        ----------
        xr_obj : xarray.DataArray
            DataArray that may contain a ``ch_shank`` coordinate.

        Returns
        -------
        int
            Number of unique shank values, or 1 if the coordinate
            is absent.
        """
        try:
            coord = xr_obj.coords.get(COORD_SHANK)
        except Exception:
            coord = None
        if coord is None:
            return 1
        vals = np.asarray(coord.values)
        # handle scalar or 1D
        try:
            unique = np.unique(vals)
            return int(unique.size)
        except Exception:
            return 1

    # Determine shank counts for each dataset and ensure they agree
    n_ap = _n_shanks(ap_env)
    n_lfp = _n_shanks(lfp_mean)
    n_csd = _n_shanks(csd_mean)
    if not (n_ap == n_lfp == n_csd):
        raise ValueError(
            f"Mismatch in shank counts: ap={n_ap}, lfp={n_lfp}, csd={n_csd}"
        )
    n_shanks = n_ap

    # 1. Initialize with constrained layout to prevent overlapping titles
    if fig is None:
        fig = plt.figure(figsize=(8, 10), layout='constrained')
    
    fig.suptitle(f'Antidromic Response - Stim: {current_uA} uA {region}', fontsize=16)
    
    subfigs = fig.subfigures(nrows=3, ncols=1)

    # --- AP Envelope plot ---
    subfigs[0].suptitle('AP Envelope', weight='bold')
    # Determine number of subplots from shanks in the data
    axs = subfigs[0].subplots(1, n_shanks)
    axs = np.array(axs).ravel()
    # Robust color limits: 5th/95th percentile of values after 0.005 s
    time_vals = ap_env.coords[DIM_TIME].values
    ap_post = ap_env.isel({DIM_TIME: time_vals >= 0.005})
    ap_vmin = float(np.nanpercentile(ap_post.values, 25))
    ap_vmax = float(np.nanpercentile(ap_post.values, 75))
    composite_subplots(ap_env, subplot_coord=COORD_SHANK, color_coord=COORD_X, axs=axs,
                       vmin=ap_vmin, vmax=ap_vmax)
       
    for i, ax in enumerate(axs):
        if i > 0:
            ax.set_ylabel('')

    # --- LFP plot ---
    subfigs[1].suptitle('LFP', weight='bold')
    axs = subfigs[1].subplots(1, n_shanks)
    axs = np.array(axs).ravel()
    pcolormesh_subplots(lfp_mean, col_coord=DIM_TIME, row_coord=COORD_Y,
                  subplot_coord=COORD_SHANK, axs=axs)
    for i, ax in enumerate(axs):
        if i > 0:
            ax.set_ylabel('')

    # --- CSD plot ---
    subfigs[2].suptitle('CSD', weight='bold')
    axs = subfigs[2].subplots(1, n_shanks)
    axs = np.array(axs).ravel()
    pcolormesh_subplots(csd_mean, col_coord=DIM_TIME, row_coord=COORD_Y,
                  subplot_coord=COORD_SHANK, axs=axs)
    for i, ax in enumerate(axs):
        if i > 0:
            ax.set_ylabel('')

    return fig


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    from mimo_pack.testing.data import synthetic_lfp, synthetic_ap
    from mimo_pack.analysis.csd import csd
    from mimo_pack.analysis.signal_proc import envelope
    
    print("Testing antidromic plotting functions...")
    
    print("1. Testing plot_antidromic_summary() with synthetic data...")
    
    try:
        # Generate synthetic multi-shank data
        np.random.seed(42)
        
        # Create synthetic AP data (short duration for antidromic response)
        ap_data = synthetic_ap(duration=0.02, fs=30000, n_chans=16, seed=42)  # 20ms
        
        # Add shank coordinates (2 shanks, 8 channels each)
        shank_coords = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        ap_data = ap_data.assign_coords({'ch_shank': ('channel', shank_coords)})
        
        # Create synthetic LFP data (same duration, lower sampling rate)
        lfp_data = synthetic_lfp(duration=0.02, fs=2000, n_chans=16, seed=42)
        lfp_data = lfp_data.assign_coords({'ch_shank': ('channel', shank_coords)})
        
        # Calculate AP envelope
        ap_env = envelope(ap_data, dim='time')
        
        # Calculate CSD from LFP
        csd_data = csd(lfp_data)
        
        # Create summary data dictionary
        summary_data = {
            'ap': ap_env,
            'lfp': lfp_data,
            'csd': csd_data
        }
        
        # Test the plotting function
        fig = plot_antidromic_summary(summary_data, current_uA=100, region='M1')
        
        print("   ✓ plot_antidromic_summary() test passed")
        
        plt.show()
        
    except Exception as e:
        print(f"   ✗ plot_antidromic_summary() test failed: {e}")
    
    # Test 2: Different stimulation parameters
    print("2. Testing plot_antidromic_summary() with different parameters...")
    
    try:
        # Test with different current and region
        fig = plot_antidromic_summary(summary_data, current_uA=250, region='AC')
        
        print("   ✓ Different parameters test passed")
        
        plt.show()
        
    except Exception as e:
        print(f"   ✗ Different parameters test failed: {e}")
    
    # Test 3: Single shank data
    print("3. Testing plot_antidromic_summary() with single shank...")
    
    try:
        # Create single-shank data
        single_shank_coords = [0] * 8
        ap_single = ap_data.isel(channel=slice(0, 8))
        ap_single = ap_single.assign_coords({'ch_shank': ('channel', single_shank_coords)})
        
        lfp_single = lfp_data.isel(channel=slice(0, 8))
        lfp_single = lfp_single.assign_coords({'ch_shank': ('channel', single_shank_coords)})
        
        csd_single = csd(lfp_single)
        
        ap_env_single = envelope(ap_single, dim='time')
        
        summary_single = {
            'ap': ap_env_single,
            'lfp': lfp_single,
            'csd': csd_single
        }
        
        fig = plot_antidromic_summary(summary_single, current_uA=150, region='CA1')
        
        print("   ✓ Single shank test passed")
        
        plt.show()
        
    except Exception as e:
        print(f"   ✗ Single shank test failed: {e}")
    
    print("All antidromic plotting tests completed!")
