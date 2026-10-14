# Plots specifically for in vitro experiments
# Author: Drew Headley
# Date: 2025-07-10

import matplotlib.pyplot as plt
import xarray as xr
from mimo_pack.plot.map import wave_map
import numpy as np
from mimo_pack.consts import DIM_TIME, DIM_CHAN


def stim_pattern_map(patterns: xr.DataArray, ax: plt.Axes = None,
                     cmap: str = 'viridis') -> plt.Axes:
    """Plot a composite image showing all stimulation pattern locations.

    Each pattern is assigned a unique index value and all patterns are
    summed into a single image for visualization.

    Parameters
    ----------
    patterns : xr.DataArray
        Image stack with an 'image' dimension, e.g. from load_images_xr().
    ax : plt.Axes, optional
        The axes to plot on. If None, a new figure and axes are created.
    cmap : str, optional
        Colormap to use. Default is 'viridis'.

    Returns
    -------
    ax : plt.Axes
        The axes with the pattern map plotted.

    Examples
    --------
    >>> from mimo_pack.fileio.image import load_images
    >>> patterns = load_images(imgs_path, insert_blank=True)
    >>> ax = stim_pattern_map(patterns)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # Give each pattern a unique index value
    patterns_uniq = (patterns > 0) * (np.arange(len(patterns))[:, None, None] + 1)

    # Collapse all patterns for visualization
    patterns_sum = patterns_uniq.sum(dim='image').to_numpy()

    im = ax.imshow(patterns_sum, aspect='equal', cmap=cmap)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Pattern number')
    ax.set_xlabel('X pixel')
    ax.set_ylabel('Y pixel')
    ax.set_title('All Stimulation Patterns')

    return ax


def focal_stim_map(rec: xr.DataArray, ax: plt.Axes = None,
                   window: list = [-0.05, 0.1], **kwargs) -> plt.Axes:
    """Plot the focal stimulation map for a recording.

    Parameters
    ----------
    rec : xr.DataArray
        The recording data with 'stim_x' and 'stim_y' coordinates.
    ax : plt.Axes, optional
        The axes to plot on. If None, a new figure and axes are created.
    window : list
        The time window to plot, specified as [start, end] in seconds.
    **kwargs : dict
        Additional keyword arguments to pass to the wave_map function.

    Returns
    -------
    ax : plt.Axes
        The axes with the focal stimulation map plotted.
    """
    
    if ax is None:
        fig, ax = plt.subplots()

    # set default values for kwargs
    kwargs.setdefault('x_scale', 0.003)  
    kwargs.setdefault('y_scale', 0.2)    
    kwargs.setdefault('alpha', 0.5)

    # Ensure the recording has the required coordinates
    rec_plot = rec.sel({DIM_CHAN: 0, DIM_TIME: slice(window[0], window[1])})
    rec_plot = rec_plot - rec_plot.mean(dim=DIM_TIME)  # remove DC offset
    wave_map(rec_plot, x_coord='stim_x', y_coord='stim_y',
             ax=ax, **kwargs)
    ax.set_title('Waveform map for channel 0')
    _units = '(µm)' if rec.attrs.get('px_per_um') is not None else '(pixels)'
    ax.set_xlabel(f'Stimulus X position {_units}')
    ax.set_ylabel(f'Stimulus Y position {_units}')
    
    return ax


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import xarray as xr
    from mimo_pack.plot.map import wave_map
    
    print("Testing in vitro plotting functions...")
    
    print("1. Testing focal_stim_map() with synthetic data...")
    
    try:
        # Create synthetic focal stimulation data
        np.random.seed(42)
        
        # Define stimulation grid
        n_stim_x, n_stim_y = 5, 4
        stim_x_pos = np.linspace(0, 200, n_stim_x)  # µm
        stim_y_pos = np.linspace(0, 150, n_stim_y)  # µm
        
        # Create meshgrid for all stimulation positions
        stim_x_grid, stim_y_grid = np.meshgrid(stim_x_pos, stim_y_pos)
        stim_x_flat = stim_x_grid.flatten()
        stim_y_flat = stim_y_grid.flatten()
        
        # Time parameters
        fs = 10000  # Hz
        duration = 0.2  # seconds
        n_timepoints = int(duration * fs)
        time = np.linspace(-0.05, 0.15, n_timepoints)
        
        # Number of channels and stimulation positions
        n_channels = 2
        n_positions = len(stim_x_flat)
        
        # Create synthetic recording data
        data = np.zeros((n_timepoints, n_channels, n_positions))
        
        for pos_idx in range(n_positions):
            for ch in range(n_channels):
                # Create synthetic evoked response
                # Stimulus artifact at t=0
                stim_time = np.abs(time) < 0.001
                data[stim_time, ch, pos_idx] = np.random.uniform(-50, 50, np.sum(stim_time))
                
                # Evoked response (depends on distance from stimulation point)
                response_time = (time > 0.002) & (time < 0.05)
                if np.sum(response_time) > 0:
                    # Distance-dependent response amplitude
                    distance = np.sqrt((stim_x_flat[pos_idx] - 100)**2 + 
                                     (stim_y_flat[pos_idx] - 75)**2)
                    amplitude = 20 * np.exp(-distance / 50)  # Decay with distance
                    
                    response_signal = amplitude * np.exp(-time[response_time] / 0.02) * np.sin(2*np.pi*200*time[response_time])
                    data[response_time, ch, pos_idx] = response_signal
                
                # Add some noise
                data[:, ch, pos_idx] += np.random.randn(n_timepoints) * 2
        
        # Create xarray DataArray
        rec_data = xr.DataArray(
            data,
            dims=[DIM_TIME, DIM_CHAN, 'position'],
            coords={
                DIM_TIME: time,
                DIM_CHAN: range(n_channels),
                'stim_x': ('position', stim_x_flat),
                'stim_y': ('position', stim_y_flat),
            }
        )
        
        # Test focal_stim_map function
        fig, ax = plt.subplots(figsize=(12, 8))
        focal_stim_map(rec_data, ax=ax, window=[-0.01, 0.08])
        
        print("   ✓ focal_stim_map() test passed")
        
    except Exception as e:
        print(f"   ✗ focal_stim_map() test failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Test 2: Different window sizes
    print("2. Testing focal_stim_map() with different windows...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        windows = [
            [-0.01, 0.02],   # Short window around stimulus
            [-0.02, 0.05],   # Medium window  
            [-0.05, 0.1],    # Full response window
            [0.01, 0.08]     # Post-stimulus only
        ]
        
        window_names = ['Short', 'Medium', 'Full', 'Post-stim']
        
        for i, (window, name) in enumerate(zip(windows, window_names)):
            focal_stim_map(rec_data, ax=axes[i], window=window)
            axes[i].set_title(f'{name} Window: {window[0]:.3f} to {window[1]:.3f}s')
        
        print("   ✓ Different window tests passed")
        
    except Exception as e:
        print(f"   ✗ Different window tests failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Test 3: Custom styling parameters
    print("3. Testing focal_stim_map() with custom styling...")
    
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Test different scaling parameters
        focal_stim_map(rec_data, ax=axes[0], x_scale=0.001, y_scale=0.1, alpha=0.8)
        axes[0].set_title('Fine Scale, Low Alpha')
        
        focal_stim_map(rec_data, ax=axes[1], x_scale=0.01, y_scale=0.5, alpha=0.3)
        axes[1].set_title('Coarse Scale, High Alpha')
        
        focal_stim_map(rec_data, ax=axes[2], x_scale=0.005, y_scale=0.3, color='red')
        axes[2].set_title('Custom Color')
        
        print("   ✓ Custom styling tests passed")
        
    except Exception as e:
        print(f"   ✗ Custom styling tests failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    print("All in vitro plotting tests completed!")
