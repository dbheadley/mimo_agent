# Time series plotting functions
# Author: Drew Headley
# Date: 2024-06-17

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr

from mimo_pack.consts import DIM_TIME, DIM_CHAN


def _stacked_lines_np(data_vals, x_vals=None, disp_color='black', 
                      sep_ratio=1, labels=None, 
                      abs_sep=None, ax=None, **kwargs):
    """
    Internal numpy implementation for plotting stacked line plots.
    
    Use stacked_lines() for the public API that accepts xarray.DataArray.

    Parameters
    ----------
    data_vals : ndarray
        An NxM array, where each column is a different line to be plotted.
    x_vals : array-like, optional
        An array specifying the x-coordinates for each column in `data_vals`.
        If None, the indices of the columns are used.
    disp_color : str, list, or ndarray, optional
        The color of each line. Can be a single color (applied to all lines),
        a list of color names, or an Nx3 array of RGB values.
    sep_ratio : float, optional
        The separation ratio between lines based on their maximum value. Default is 1.
    labels : list of str, optional
        Labels for each line, displayed only on the right side as ticks.
    abs_sep : float, optional
        Absolute separation between lines. Overrides `sep_ratio` if provided.
    **kwargs : dict
        Additional keyword arguments passed to the `plot` function.

    Returns
    -------
    ax : axes object
        The axes object containing the plot.
    """
    # Validate inputs
    if x_vals is None:
        x_vals = np.arange(data_vals.shape[0])
    if len(x_vals) != data_vals.shape[0]:
        raise ValueError("Length of x_vals must match the number of rows in data_vals.")
    
    # Handle disp_color
    n_lines = data_vals.shape[1]
    if isinstance(disp_color, str):
        disp_color = [disp_color] * n_lines
    if isinstance(disp_color, (list, tuple)):
        disp_color = np.array(
            [mcolors.to_rgb(c) if isinstance(c, str) else c for c in disp_color],
            dtype=float,
        )
    else:
        disp_color = np.atleast_2d(np.array(disp_color, dtype=float))
        if disp_color.shape[0] == 1 and n_lines > 1:
            disp_color = np.tile(disp_color, (n_lines, 1))

    if disp_color.shape[0] != n_lines:
        raise ValueError("Number of colors must match the number of lines in data_vals.")

    if ax is None:
        fig, ax = plt.subplots()

    # Create a second axis for right-side labels
    ax_right = ax.twinx()

    # Handle separation
    if abs_sep is not None:
        line_sep = abs_sep
    else:
        max_diff = np.nanmax(np.abs(np.nanmax(data_vals, axis=0) - np.nanmin(data_vals, axis=0)))
        line_sep = max_diff * sep_ratio

    # Plot lines
    line_handles = []
    offsets = []
    for i, row in enumerate(data_vals.T):
        offset = i * line_sep
        offsets.append(offset)
        line, = ax.plot(x_vals, row + offset, color=disp_color[i], **kwargs)
        line_handles.append(line)
        ax.axhline(y=offset, color='gray', linestyle=':', linewidth=0.5)

    # Set right axis ticks and labels
    ax_right.set_ylim(ax.get_ylim())
    ax_right.set_yticks(offsets)
    if labels is not None:
        ax_right.set_yticklabels(labels)
    else:
        ax_right.set_yticklabels([''] * len(offsets))

    # Hide right axis spines except right
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_right.spines['bottom'].set_visible(False)

    return ax


def stacked_lines(data, time_dim=DIM_TIME, chan_dim=DIM_CHAN, 
                  label_coord=None, **kwargs):
    """
    Plot a series of line plots stacked along the y-axis.
    
    This is the public API for stacked line plotting. It accepts an xarray.DataArray
    with proper dimensions and delegates to the internal numpy implementation.

    Parameters
    ----------
    data : xarray.DataArray
        A 2D DataArray with time and channel dimensions.
    time_dim : str, optional
        The name of the time dimension in the DataArray (default is 'time').
    chan_dim : str, optional
        The name of the channel dimension in the DataArray (default is 'channel').
    label_coord : str, optional
        The name of the coordinate to use for labeling each line. If None,
        uses the channel dimension values as labels.
    **kwargs : dict
        Additional keyword arguments passed to _stacked_lines_np.
        Includes: disp_color, sep_ratio, abs_sep, ax.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
        
    Raises
    ------
    TypeError
        If input is not an xarray.DataArray.
    ValueError
        If required dimensions are missing.
        
    Examples
    --------
    >>> from mimo_pack.testing import synthetic_lfp
    >>> lfp = synthetic_lfp(duration=1.0, fs=1000, n_chans=8)
    >>> ax = stacked_lines(lfp)
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError(
            f"Input must be an xarray.DataArray, got {type(data).__name__}. "
            "Use _stacked_lines_np() for raw numpy array input."
        )
    
    # Validate dimensions exist
    if time_dim not in data.dims:
        raise ValueError(
            f"Dimension '{time_dim}' not found in DataArray. "
            f"Available dimensions: {list(data.dims)}"
        )
    if chan_dim not in data.dims:
        raise ValueError(
            f"Dimension '{chan_dim}' not found in DataArray. "
            f"Available dimensions: {list(data.dims)}"
        )
    
    # Transpose to have time first, then channel
    data = data.transpose(time_dim, chan_dim)
    
    # Extract x values from time dimension
    x_vals = data[time_dim].values
    
    # Extract labels
    if label_coord is not None:
        labels = data.coords[label_coord].values
    else:
        labels = data[chan_dim].values
    
    # Call the numpy implementation
    ax = _stacked_lines_np(
        data.values, 
        x_vals=x_vals, 
        labels=labels,
        **kwargs
    )
    
    return ax


# Example usage
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("Testing timeseries plotting functions...")
    
    # Test 1: Basic stacked lines
    print("1. Testing stacked_lines() with synthetic data...")
    
    try:
        np.random.seed(42)
        num_trials = 10
        dur_trial = 200
        
        # Create synthetic neural-like data with different baselines
        fake_data = np.random.randn(dur_trial, num_trials)  # Shape: (timepoints, trials)
        for i in range(num_trials):
            fake_data[:, i] += np.sin(np.linspace(0, 4*np.pi, dur_trial)) * 0.3
            fake_data[:, i] += i * 0.1  # Small offset per trial
        
        labels = [f"Trial {i+1}" for i in range(num_trials)]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        stacked_lines(fake_data, disp_color='blue', sep_ratio=2, 
                     labels=labels, ax=ax)
        ax.set_title('Stacked Lines Test - Basic')
        ax.set_xlabel('Time Points')
        ax.set_ylabel('Trials (stacked)')
        
        print("   ✓ Basic stacked lines test passed")
        
    except Exception as e:
        print(f"   ✗ Basic stacked lines test failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Test 2: Different colors per line
    print("2. Testing stacked_lines() with different colors...")
    
    try:
        # Create colorful data
        colors = plt.cm.rainbow(np.linspace(0, 1, num_trials))
        
        fig, ax = plt.subplots(figsize=(12, 8))
        stacked_lines(fake_data, disp_color=colors, sep_ratio=1.5,
                     labels=labels, ax=ax)
        ax.set_title('Stacked Lines Test - Different Colors')
        ax.set_xlabel('Time Points')
        ax.set_ylabel('Trials (stacked)')
        
        print("   ✓ Different colors test passed")
        
    except Exception as e:
        print(f"   ✗ Different colors test failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Test 3: Custom x values
    print("3. Testing stacked_lines() with custom x values...")
    
    try:
        x_vals = np.linspace(-1, 1, dur_trial)  # Custom time axis
        
        fig, ax = plt.subplots(figsize=(12, 8))
        stacked_lines(fake_data, x_vals=x_vals, disp_color='green',
                     sep_ratio=3, labels=labels, ax=ax)
        ax.set_title('Stacked Lines Test - Custom X Values')
        ax.set_xlabel('Time (custom units)')
        ax.set_ylabel('Trials (stacked)')
        
        print("   ✓ Custom x values test passed")
        
    except Exception as e:
        print(f"   ✗ Custom x values test failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Test 4: Absolute separation
    print("4. Testing stacked_lines() with absolute separation...")
    
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        stacked_lines(fake_data, disp_color='red', abs_sep=2.0,
                     labels=labels, ax=ax)
        ax.set_title('Stacked Lines Test - Absolute Separation')
        ax.set_xlabel('Time Points')
        ax.set_ylabel('Trials (stacked)')
        
        print("   ✓ Absolute separation test passed")
        
    except Exception as e:
        print(f"   ✗ Absolute separation test failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Test 5: LFP-like data with multiple channels
    print("5. Testing stacked_lines() with LFP-like data...")
    
    try:
        # Generate LFP-like data (multiple channels over time)
        n_channels = 8
        n_timepoints = 500
        time_vec = np.linspace(0, 2, n_timepoints)  # 2 seconds
        
        lfp_data = np.zeros((n_timepoints, n_channels))  # Shape: (time, channels)
        for ch in range(n_channels):
            # Add different frequency components per channel
            lfp_data[:, ch] = (np.sin(2 * np.pi * 10 * time_vec) * np.exp(-time_vec/2) +
                              np.sin(2 * np.pi * 30 * time_vec) * 0.3 +
                              np.random.randn(n_timepoints) * 0.1)
        
        channel_labels = [f'Ch {i+1}' for i in range(n_channels)]
        
        fig, ax = plt.subplots(figsize=(14, 10))
        stacked_lines(lfp_data, x_vals=time_vec, disp_color='purple',
                     sep_ratio=2, labels=channel_labels, ax=ax)
        ax.set_title('Stacked Lines Test - LFP-like Data')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Channels (stacked)')
        
        print("   ✓ LFP-like data test passed")
        
    except Exception as e:
        print(f"   ✗ LFP-like data test failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    print("All timeseries plotting tests completed!")
