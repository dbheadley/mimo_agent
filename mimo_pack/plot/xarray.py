import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from mimo_pack.plot.image import composite_image
from mimo_pack.consts import DIM_TIME, DIM_CHAN, COORD_X, COORD_Y, COORD_SHANK
from mimo_pack.util.xarray import get_dim, group_and_stack
from mimo_pack.plot.colors import MIMO_SEQ_CMAP

def _apply_subplots(data, plot_func, subplot_coord, axs=None, **kwargs):
    """Apply a single-panel plotting function across groups defined by a coordinate.

    Groups *data* by *subplot_coord* and calls *plot_func* once per
    group on its own ``Axes``.

    Parameters
    ----------
    data : xarray.DataArray
        Data to split and plot.
    plot_func : callable
        Plotting function with signature ``plot_func(data, ax=..., **kwargs)``.
    subplot_coord : str or None
        Coordinate used to group data into subplots. If ``None``, the
        entire DataArray is plotted as a single panel.
    axs : array-like of matplotlib.axes.Axes or None, optional
        Pre-existing axes to draw on. Created automatically if ``None``.
    **kwargs
        Forwarded to *plot_func*.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : numpy.ndarray of matplotlib.axes.Axes
    last_return
        The return value of the last *plot_func* call.
    """
    # 1. Group Data
    if subplot_coord is None:
        # Treat as single group if None passed
        groups = [(None, data)]
    elif subplot_coord in data.coords:
        groups = list(data.groupby(subplot_coord))
    else:
        raise ValueError(f"Coordinate '{subplot_coord}' not found in DataArray.")
        
    n_plots = len(groups)

    # 2. Setup Axes
    if axs is None:
        fig, axs = plt.subplots(1, n_plots, figsize=(4 * n_plots, 4), squeeze=False)
        axs = axs.flatten()
    else:
        fig = plt.gcf()
        axs = np.atleast_1d(axs)
        if len(axs) < n_plots:
             raise ValueError(f"Provided {len(axs)} axes, but data requires {n_plots}.")

    last_return = None

    # 3. Iterate and Plot
    for i, (key, sub_da) in enumerate(groups):
        ax = axs[i]
        
        if sub_da.size == 0:
            ax.set_title(f'{subplot_coord} {key}\n(empty)')
            continue

        # Execute specific plotting function
        last_return = plot_func(sub_da, ax=ax, **kwargs)

        # Labels and Formatting
        ax.set_title(f'{subplot_coord} {key}')
            
        if i > 0:
            ax.set_ylabel('')

    return fig, axs, last_return

def composite(data, color_coord=COORD_X, row_coord=COORD_Y, col_coord=DIM_TIME,
                 background=[0, 0, 0], ax=None, vmin=None, vmax=None):
    """Create an alpha-blended composite image from an xarray DataArray.

    The DataArray values are grouped by *color_coord*, arranged by
    *row_coord* vs *col_coord*, and alpha-blended into a single RGB
    image.

    Parameters
    ----------
    data : xarray.DataArray
        2-D or 3-D DataArray to plot.
    color_coord : str, optional
        Coordinate whose unique values map to separate colour channels.
        Default is ``COORD_X``.
    row_coord : str, optional
        Coordinate or dimension for the y-axis. Default is ``COORD_Y``.
    col_coord : str, optional
        Coordinate or dimension for the x-axis. Default is ``DIM_TIME``.
    background : list of float, optional
        RGB background colour (0–1). Default is ``[0, 0, 0]``.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. A new figure is created if ``None``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Prepare Data
    # Get dimension names
    row_dim = get_dim(data, row_coord)
    col_dim = get_dim(data, col_coord)
    color_dim = get_dim(data, color_coord)
    
    # Handle 2D vs 3D data
    if data.ndim == 2:
        # 2D data - need to use group_and_stack to create 3rd dimension
        if color_coord not in data.coords:
            raise ValueError(f"For 2D data, color_coord '{color_coord}' must be a coordinate")
        plot_data = group_and_stack(data, color_coord)
        color_dim = color_coord  # After grouping, this becomes a dimension
    elif data.ndim == 3:
        # 3D data - use as is
        plot_data = data
    else:
        raise ValueError(f"Data must be 2D or 3D, got {data.ndim}D")
    
    # Validate that all coordinates map to different dimensions
    if row_dim == col_dim:
        raise ValueError(f"row_coord and col_coord map to same dimension '{row_dim}'")
    if row_dim == color_dim:
        raise ValueError(f"row_coord and color_coord map to same dimension '{row_dim}'")
    if col_dim == color_dim:
        raise ValueError(f"col_coord and color_coord map to same dimension '{col_dim}'")
    
    # Transpose to correct dimension order
    plot_data = plot_data.transpose(row_dim, col_dim, color_dim)

    img = plot_data.values
    # Flip y-axis (rows) so higher coordinates appear at the top
    img = np.flip(img, axis=0)
    
    # Extract extents
    col_vals = plot_data.coords[col_coord].values
    row_vals = plot_data.coords[row_coord].values
    extent = [col_vals[0], col_vals[-1], row_vals[0], row_vals[-1]]
    
    composite_image(
        img,
        extent=extent,
        ax=ax,
        colorbars=False,
        aspect='auto',
        background=background,
        vmin=vmin,
        vmax=vmax,
    )
    
    ax.set_xlabel(col_coord)
    ax.set_ylabel(row_coord)
    
    return fig, ax

def composite_subplots(data, subplot_coord=COORD_SHANK, color_coord=COORD_X,
                 row_coord=COORD_Y, col_coord=DIM_TIME,
                 background=[0, 0, 0], axs=None, vmin=None, vmax=None):
    """Plot composite images grouped by *subplot_coord*.

    Splits *data* by *subplot_coord* (e.g. shank) and draws a
    :func:`composite` panel for each group.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to plot.
    subplot_coord : str, optional
        Coordinate used to create subplots. Default is ``COORD_SHANK``.
    color_coord : str, optional
        Coordinate for colour channels. Default is ``COORD_X``.
    row_coord : str, optional
        Coordinate for the y-axis. Default is ``COORD_Y``.
    col_coord : str, optional
        Coordinate for the x-axis. Default is ``DIM_TIME``.
    background : list of float, optional
        RGB background colour. Default is ``[0, 0, 0]``.
    axs : array-like of matplotlib.axes.Axes or None, optional
        Pre-existing axes. Created automatically if ``None``.
    vmin : float or None, optional
        Minimum value for colour normalisation. If ``None``, per-channel
        minimum is used.
    vmax : float or None, optional
        Maximum value for colour normalisation. If ``None``, per-channel
        maximum is used.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : numpy.ndarray of matplotlib.axes.Axes
    last_return
        Return value of the last :func:`composite` call.
    """
    return _apply_subplots(
        data,
        composite,
        subplot_coord,
        axs=axs,
        color_coord=color_coord,
        row_coord=row_coord,
        col_coord=col_coord,
        background=background,
        vmin=vmin,
        vmax=vmax,
    )

def pcolormesh(data, row_coord=COORD_Y, col_coord=DIM_TIME, 
                  vmin=None, vmax=None, cmap=MIMO_SEQ_CMAP, ax=None):
    """Draw a pseudocolour mesh plot of a 2-D DataArray.

    Parameters
    ----------
    data : xarray.DataArray
        2-D (or squeezable-to-2-D) DataArray.
    row_coord : str, optional
        Coordinate for the y-axis. Default is ``COORD_Y``.
    col_coord : str, optional
        Coordinate for the x-axis. Default is ``DIM_TIME``.
    vmin : float or None, optional
        Minimum colour-scale value.
    vmax : float or None, optional
        Maximum colour-scale value.
    cmap : str, optional
        Matplotlib colourmap name. Default is ``'viridis'``.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. A new figure is created if ``None``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    mesh : matplotlib.collections.QuadMesh
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Prepare Data
    # Get dimension names
    row_dim = get_dim(data, row_coord)
    col_dim = get_dim(data, col_coord)
    
    plot_data = data

    # Remove singleton dimensions not named in row_coord or col_coord
    for dim in list(plot_data.dims):
        if plot_data.sizes[dim] == 1 and dim not in {row_dim, col_dim}:
            plot_data = plot_data.squeeze(dim)
    
    # Transpose to (row, col)
    plot_data = plot_data.transpose(row_dim, col_dim)

    mesh = ax.pcolormesh(
        plot_data.coords[col_coord].values,
        plot_data.coords[row_coord].values,
        plot_data.values,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        shading='auto'
    )
    
    ax.set_xlabel(col_coord)
    ax.set_ylabel(row_coord)

    return fig, ax, mesh

def pcolormesh_subplots(data, subplot_coord=COORD_SHANK, row_coord=COORD_Y, col_coord=DIM_TIME, 
                  vmin=None, vmax=None, cmap=MIMO_SEQ_CMAP, axs=None):
    """Plot pcolormesh heatmaps grouped by *subplot_coord*.

    Splits *data* by *subplot_coord* and draws a :func:`pcolormesh`
    panel for each group.

    Parameters
    ----------
    data : xarray.DataArray
        DataArray to plot.
    subplot_coord : str, optional
        Coordinate used to create subplots. Default is ``COORD_SHANK``.
    row_coord : str, optional
        Coordinate for the y-axis. Default is ``COORD_Y``.
    col_coord : str, optional
        Coordinate for the x-axis. Default is ``DIM_TIME``.
    vmin : float or None, optional
        Minimum colour-scale value.
    vmax : float or None, optional
        Maximum colour-scale value.
    cmap : str, optional
        Matplotlib colourmap name. Default is ``'viridis'``.
    axs : array-like of matplotlib.axes.Axes or None, optional
        Pre-existing axes. Created automatically if ``None``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : numpy.ndarray of matplotlib.axes.Axes
    last_return
        Return value of the last :func:`pcolormesh` call.
    """
    return _apply_subplots(
        data,
        pcolormesh,
        subplot_coord,
        axs=axs,
        row_coord=row_coord,
        col_coord=col_coord,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap
    )


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from mimo_pack.testing.data import synthetic_lfp
    
    print("Testing xarray plotting functions...")
    
    # Generate synthetic LFP data
    lfp = synthetic_lfp(duration=0.5, fs=1000, n_chans=8, seed=42)
    
    # Test composite plot
    print("1. Testing composite()...")
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        composite(lfp, ax=ax)
        ax.set_title('Composite Plot Test')
        print("   ✓ composite() test passed")
    except Exception as e:
        print(f"   ✗ composite() test failed: {e}")
    plt.tight_layout()
    plt.show()
    
    # Test pcolormesh plot  
    print("2. Testing pcolormesh()...")
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        pcolormesh(lfp, ax=ax)
        ax.set_title('Pcolormesh Plot Test')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Channel Y Position (µm)')
        print("   ✓ pcolormesh() test passed")
    except Exception as e:
        print(f"   ✗ pcolormesh() test failed: {e}")
    plt.tight_layout()
    plt.show()
    
    # Test with multi-shank data for subplots
    print("3. Testing composite_subplots()...")
    try:
        # Add shank coordinate
        lfp_shank = lfp.assign_coords({
            'ch_shank': ('channel', [0, 0, 1, 1, 0, 0, 1, 1])
        })
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        composite_subplots(lfp_shank, subplot_coord=COORD_SHANK, axs=axs)
        for i, ax in enumerate(axs):
            ax.set_title(f'Shank {i} Composite')
        print("   ✓ composite_subplots() test passed")
    except Exception as e:
        print(f"   ✗ composite_subplots() test failed: {e}")
    plt.tight_layout()
    plt.show()
    
    print("All xarray plotting tests completed!")
