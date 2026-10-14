# Mapped data plotting functions
# Author: Drew Headley
# Date: 2024-06-17

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from mimo_pack.consts import DIM_TIME, DIM_CHAN, COORD_X, COORD_Y
from mimo_pack.plot.colors import MIMO_SEQ_CMAP

def _wave_map_numpy(waves, x_pos, y_pos, c=None, x_scale=1, y_scale=1, 
             cmap=MIMO_SEQ_CMAP, vmin=None, vmax=None, ax=None, 
             text=None, **kwargs):
    """
    Internal numpy implementation for plotting a map of waveforms.
    
    Use wave_map() for the public API that accepts xarray.DataArray.

    Parameters
    ----------
    waves : N-length list of np.ndarray or (T, N) np.ndarray
        Waveforms for each channel. If a single array is provided,
        each column is treated as a channel waveform.
    x_pos : np.ndarray, (N,)
        The x coordinates of the channels.
    y_pos : np.ndarray, (N,)
        The y coordinates of the channels.
    c : np.ndarray, (N,) optional
        Values to use when setting the color of each waveform. If None,
        the waveforms are plotted black with no color scaling.
    x_scale : float, optional
        Scaling factor for x-axis (default is 1).
    y_scale : float, optional
        Scaling factor for y-axis (default is 1).
    cmap : str, optional,
        Colormap to use (default is 'viridis').
    vmin : float, optional
        Minimum value for color scaling (default is None).
    vmax : float, optional
        Maximum value for color scaling (default is None).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (default is None, creates a new figure).
    text : list of str, optional
        List of text labels for each waveform (default is None).
    **kwargs : additional keyword arguments
        Additional parameters for `plt.plot`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes on which the waveforms were plotted.
    """
    
    if ax is None:
        fig, ax = plt.subplots()

    if isinstance(waves, np.ndarray):
        if waves.ndim == 1:
            waves = waves.reshape(-1, 1)
        elif waves.ndim == 2:
            waves = [waves[:, i] for i in range(waves.shape[1])]
    elif not isinstance(waves, list):
        raise ValueError("waves must be a list of np.ndarray or a 2D np.ndarray")

    if len(waves) != x_pos.size or len(waves) != y_pos.size:
        raise ValueError("Length of waves must match length of x_pos and y_pos")
    
    # Handle color argument
    if c is None:
        c = np.repeat('black', len(waves))
    elif isinstance(c, str):
        # Single color string - apply to all waveforms
        c = np.repeat(c, len(waves))
    elif not hasattr(c, '__len__'):
        # Single value - apply to all
        c = np.repeat(c, len(waves))
    elif len(c) != len(waves):
        raise ValueError("Length of c must match length of waves")
    
    if not isinstance(c, np.ndarray):
        c = np.array(c)

    if text is not None:
        if len(text) != len(waves):
            raise ValueError("Length of text must match length of waves")
    
    # set colors by colormap
    if isinstance(c, np.ndarray) and np.issubdtype(c.dtype, np.number):
        if vmin is None:
            vmin = np.min(c)
        if vmax is None:
            vmax = np.max(c)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap(cmap)
        c = cmap(norm(c))
        

    for i, wave in enumerate(waves):
        if wave.ndim == 1:
            wave = wave.reshape(-1,1)

        x_vals = x_pos[i] + np.arange(wave.shape[0]) * x_scale
        y_vals = y_pos[i] + wave * y_scale

        ax.plot(x_vals, y_vals, color=c[i], **kwargs)
        if text is not None:
            ax.text(x_vals[0], y_vals[0], text[i], fontsize=8,
                    verticalalignment='bottom', horizontalalignment='right')

    return ax


def _amp_map_numpy(amp, x_pos, y_pos, cmap=MIMO_SEQ_CMAP, ax=None, **kwargs):
        """
        Plot a map of amplitude values across channels.

        Parameters
        ----------
        amp : np.ndarray, (N,)
            The amplitude values for each channel.
        x_pos : np.ndarray, (N,)
            The x coordinates of the channels.
        y_pos : np.ndarray, (N,)
            The y coordinates of the channels.
        cmap : str, optional
            Colormap to use for the plot, by default 'viridis'.
        ax : matplotlib.axes.Axes, optional
            Axes to plot on, if None a new figure is created, by default None.
        **kwargs : dict, optional
            Additional keyword arguments for the scatter plot.

        Returns
        -------
        ax: matplotlib.axes.Axes
            The axes with the amplitude map plotted.
        scat: matplotlib.collections.PathCollection
            The scatter plot object.
        """

        if ax is None:
            fig, ax = plt.subplots()
        
        # set point size to minimum non-zero distance between positions
        median_dist = (np.sqrt((x_pos.reshape(-1,1)-x_pos)**2 + (y_pos.reshape(-1,1)-y_pos)**2))
        median_dist = np.nanmin(median_dist[median_dist > 0])  # ignore zero distances

        scat = ax.scatter(x_pos, y_pos, s=median_dist,
                          c=amp, cmap=cmap, **kwargs)
                
        return ax, scat

def amp_map(amp, x_coord=COORD_X, y_coord=COORD_Y, **kwargs):
    """
    Plot a map of amplitude values across channels using xarray coordinates.

    Parameters
    ----------
    amp : xarray.DataArray
        The amplitude values for each channel.
    x_coord : str, optional
        The name of the x coordinate in the DataArray (default is COORD_X).
    y_coord : str, optional
        The name of the y coordinate in the DataArray (default is COORD_Y).
    **kwargs : dict, optional
        Additional keyword arguments for the scatter plot.

    Returns
    -------
    ax: matplotlib.axes.Axes
        The axes with the amplitude map plotted.
    scat: matplotlib.collections.PathCollection
        The scatter plot object.
    """
    
    ax, scat = _amp_map_numpy(amp.values, amp[x_coord].values, amp[y_coord].values, **kwargs)
    
    return ax, scat

def wave_map(waves, x_coord=COORD_X, y_coord=COORD_Y, time_dim=DIM_TIME, 
             label_coord=None, color=None, **kwargs):
    """
    Plot a map of waveforms across channels using xarray coordinates.
    
    This is the public API for wave map plotting. It accepts an xarray.DataArray
    with proper coordinates and delegates to the internal numpy implementation.

    Parameters
    ----------
    waves : xarray.DataArray
        The waveforms for each channel, with spatial coordinates.
    x_coord : str, optional
        The name of the x coordinate in the DataArray (default is 'ch_x').
    y_coord : str, optional
        The name of the y coordinate in the DataArray (default is 'ch_y').
    time_dim : str, optional
        The name of the time dimension in the DataArray (default is 'time').
    label_coord : str, optional
        The name of the coordinate to use for labeling each waveform (default is None).
    color : str, array-like, or None, optional
        Color specification for the waveforms. Can be:
        - A single color string (e.g., 'blue', 'k', '#FF0000') applied to all waveforms
        - An array of values for colormap-based coloring (uses cmap, vmin, vmax kwargs)
        - None to use default black color
    **kwargs : dict, optional
        Additional keyword arguments passed to _wave_map_numpy.
        Includes: c, x_scale, y_scale, cmap, vmin, vmax, ax, text.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes with the waveform map plotted.
        
    Raises
    ------
    TypeError
        If input is not an xarray.DataArray.
    ValueError
        If required coordinates are missing.
        
    Examples
    --------
    >>> from mimo_pack.testing import synthetic_lfp
    >>> lfp = synthetic_lfp(duration=0.1, fs=1000, n_chans=32)
    >>> ax = wave_map(lfp, color='blue')
    """
    if not isinstance(waves, xr.DataArray):
        raise TypeError(
            f"Input must be an xarray.DataArray, got {type(waves).__name__}. "
            "Use _wave_map_numpy() for raw numpy array input."
        )
    
    # Validate coordinates exist
    if x_coord not in waves.coords:
        raise ValueError(
            f"Coordinate '{x_coord}' not found in DataArray. "
            f"Available coordinates: {list(waves.coords.keys())}"
        )
    if y_coord not in waves.coords:
        raise ValueError(
            f"Coordinate '{y_coord}' not found in DataArray. "
            f"Available coordinates: {list(waves.coords.keys())}"
        )
    if time_dim not in waves.dims:
        raise ValueError(
            f"Dimension '{time_dim}' not found in DataArray. "
            f"Available dimensions: {list(waves.dims)}"
        )
    
    # Get the dimensions for x_coord and y_coord
    x_dims = waves.coords[x_coord].dims
    y_dims = waves.coords[y_coord].dims
    
    # If x_coord and y_coord are on different dimensions, stack them
    if len(x_dims) == 1 and len(y_dims) == 1 and x_dims[0] != y_dims[0]:
        # Stack the two spatial dimensions into a single channel dimension
        waves = waves.stack(channel=(x_dims[0], y_dims[0]))
        ch_dim = 'channel'
    elif len(x_dims) == 1:
        ch_dim = x_dims[0]
    else:
        # x_coord is a scalar - use y_coord's dimension
        ch_dim = y_dims[0] if len(y_dims) == 1 else None
        if ch_dim is None:
            raise ValueError("Cannot determine channel dimension from coordinates")
    
    # Transpose to have time first, then channel
    waves = waves.transpose(time_dim, ch_dim)
    
    # Extract label text if specified
    if label_coord is not None:
        text = waves.coords[label_coord].values
    else:
        text = kwargs.pop('text', None)

    # Handle color argument - map to 'c' parameter for _wave_map_numpy
    if color is not None:
        kwargs['c'] = color

    ax = _wave_map_numpy(
        waves.data, 
        waves[x_coord].values, 
        waves[y_coord].values, 
        text=text, 
        **kwargs
    )
    
    return ax


# Legacy alias for backwards compatibility
def wave_map_xr(waves, x_coord=COORD_X, y_coord=COORD_Y, time_dim=DIM_TIME, label_coord=None, **kwargs):
    """
    Plot a map of waveforms across channels using xarray coordinates.
    
    .. deprecated::
        Use :func:`wave_map` instead. This function is provided for 
        backwards compatibility only.

    Parameters
    ----------
    waves : xarray.DataArray
        The waveforms for each channel.
    x_coord : str, optional
        The name of the x coordinate in the DataArray (default is COORD_X).
    y_coord : str, optional
        The name of the y coordinate in the DataArray (default is COORD_Y).
    time_dim : str, optional
        The name of the time dimension in the DataArray (default is DIM_TIME).
    label_coord : str, optional
        The name of the coordinate to use for labeling each waveform (default is None).
    **kwargs : dict, optional
        Additional keyword arguments for the plot.

    Returns
    -------
    ax: matplotlib.axes.Axes
        The axes with the waveform map plotted.
    """
    import warnings
    warnings.warn(
        "wave_map_xr is deprecated, use wave_map instead",
        DeprecationWarning,
        stacklevel=2
    )
    return wave_map(waves, x_coord=x_coord, y_coord=y_coord, 
                    time_dim=time_dim, label_coord=label_coord, **kwargs)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from mimo_pack.testing.data import synthetic_lfp, synthetic_ap
    
    print("Testing map plotting functions...")
    
    # Generate synthetic data
    lfp = synthetic_lfp(duration=0.2, fs=1000, n_chans=6, seed=42)
    ap = synthetic_ap(duration=0.01, fs=30000, n_chans=6, seed=42)
    
    # Test wave_map with LFP
    print("1. Testing wave_map() with LFP data...")
    fig, ax = plt.subplots(figsize=(12, 8))
    try:
        wave_map(lfp, ax=ax, x_scale=0.01, y_scale=50)
        ax.set_title('LFP Wave Map Test')
        ax.set_xlabel('Time (scaled) + X position')
        ax.set_ylabel('Y position (µm)')
        print("   ✓ wave_map() LFP test passed")
    except Exception as e:
        print(f"   ✗ wave_map() LFP test failed: {e}")
    plt.tight_layout()
    plt.show()
    
    # Test wave_map with AP data
    print("2. Testing wave_map() with AP data...")
    fig, ax = plt.subplots(figsize=(12, 8))
    try:
        wave_map(ap, ax=ax, x_scale=0.0005, y_scale=100)
        ax.set_title('AP Wave Map Test')
        ax.set_xlabel('Time (scaled) + X position')
        ax.set_ylabel('Y position (µm)')
        print("   ✓ wave_map() AP test passed")
    except Exception as e:
        print(f"   ✗ wave_map() AP test failed: {e}")
    plt.tight_layout()
    plt.show()
    
    # Test amp_map with numpy arrays
    print("3. Testing amp_map() with numpy arrays...")
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        # Create test amplitude data
        amp_vals = np.array([1.0, 2.5, 0.8, 3.2, 1.8, 2.1])
        x_pos = np.array([0, 50, 100, 0, 50, 100])
        y_pos = np.array([0, 0, 0, 50, 50, 50])
        
        result_ax, scat = amp_map(amp_vals, x_pos, y_pos, ax=ax, cmap='viridis')
        ax.set_title('Amplitude Map Test')
        ax.set_xlabel('X position (µm)')
        ax.set_ylabel('Y position (µm)')
        plt.colorbar(scat, ax=ax, label='Amplitude')
        print("   ✓ amp_map() numpy test passed")
    except Exception as e:
        print(f"   ✗ amp_map() numpy test failed: {e}")
    plt.tight_layout()
    plt.show()
    
    # Test amp_map with xarray
    print("4. Testing amp_map() with xarray...")
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        # Create amplitude DataArray
        amp_data = np.random.uniform(0.5, 3.0, lfp.sizes['channel'])
        amp_xr = lfp.isel(time=0).copy()  # Take first timepoint
        amp_xr.data = amp_data
        
        result_ax, scat = amp_map(amp_xr, ax=ax)
        ax.set_title('Amplitude Map (xarray) Test')
        ax.set_xlabel('X position (µm)')
        ax.set_ylabel('Y position (µm)')
        print("   ✓ amp_map() xarray test passed")
    except Exception as e:
        print(f"   ✗ amp_map() xarray test failed: {e}")
    plt.tight_layout()
    plt.show()
    
    # Test wave_map with labels
    print("5. Testing wave_map() with channel labels...")
    fig, ax = plt.subplots(figsize=(12, 8))
    try:
        # Add labels to LFP data
        lfp_labeled = lfp.assign_coords({
            'ch_label': ('channel', [f'Ch{i+1}' for i in range(lfp.sizes['channel'])])
        })
        
        wave_map(lfp_labeled, label_coord='ch_label', ax=ax, x_scale=0.01, y_scale=50)
        ax.set_title('LFP Wave Map with Labels Test')
        print("   ✓ wave_map() with labels test passed")
    except Exception as e:
        print(f"   ✗ wave_map() with labels test failed: {e}")
    plt.tight_layout()
    plt.show()
    
    print("All map plotting tests completed!")
