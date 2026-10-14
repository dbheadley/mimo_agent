# Plotting functions for line plots
# Drew Headley, 2025

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

def plot_mean_shaded(x: xr.DataArray,
                     plot_dim: str,
                     ax=None,
                     err: str = 'sem',
                     ci: float = 95.0,
                     color: str = 'k',
                     alpha: float = 0.3,
                     label: str | None = None,
                     line_kwargs: dict | None = None,
                     fill_kwargs: dict | None = None):
    """
    Plot the mean of an xarray.DataArray along all dimensions except `plot_dim`,
    with shaded error bars.

    Parameters
    ----------
    x : xr.DataArray
        Input data.
    plot_dim : str
        Dimension to plot along (x-axis). Mean and error are computed over all
        other dimensions.
    ax : matplotlib.axes.Axes or None
        Axes to plot on. If None, a new axes is created.
    err : {'sem','std','ci'}
        Error to display: standard error of the mean ('sem'), standard deviation ('std'),
        or confidence interval ('ci' uses normal approx and `ci` level).
    ci : float
        Confidence level (%) used when err == 'ci' (default 95).
    color : str
        Line / fill color.
    alpha : float
        Fill transparency.
    label : str or None
        Line label.
    line_kwargs : dict or None
        Additional kwargs passed to ax.plot for the mean line.
    fill_kwargs : dict or None
        Additional kwargs passed to ax.fill_between for shaded region.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot.
    """
    if plot_dim not in x.dims:
        raise ValueError(f"plot_dim '{plot_dim}' not in x.dims: {x.dims}")

    reduce_dims = [d for d in x.dims if d != plot_dim]
    if len(reduce_dims) == 0:
        raise ValueError("No remaining dimensions to reduce over; nothing to average.")

    # compute mean
    mean = x.mean(dim=reduce_dims, skipna=True)

    # compute error band
    if err == 'std':
        err_arr = x.std(dim=reduce_dims, skipna=True)
        lower = mean - err_arr
        upper = mean + err_arr
    elif err == 'sem':
        std = x.std(dim=reduce_dims, skipna=True)
        # compute effective N along reduce_dims (product of counts)
        counts = x.count(dim=reduce_dims)
        # counts may be DataArray with same dims as mean (i.e. indexed by plot_dim)
        # Convert to float and avoid zeros
        counts = counts.where(counts > 0, other=np.nan).astype(float)
        sem = std / np.sqrt(counts)
        lower = mean - sem
        upper = mean + sem
    elif err == 'ci':
        std = x.std(dim=reduce_dims, skipna=True)
        counts = x.count(dim=reduce_dims)
        counts = counts.where(counts > 0, other=np.nan).astype(float)
        sem = std / np.sqrt(counts)
        z = norm.ppf(0.5 + ci/200.0)
        margin = z * sem
        lower = mean - margin
        upper = mean + margin
    else:
        raise ValueError("err must be one of {'sem','std','ci'}")

    # prepare plotting arrays
    x_vals = mean[plot_dim].values
    y_vals = mean.values
    y_lower = lower.values
    y_upper = upper.values

    if ax is None:
        fig, ax = plt.subplots()

    lkw = {} if line_kwargs is None else dict(line_kwargs)
    fkw = {} if fill_kwargs is None else dict(fill_kwargs)

    ax.plot(x_vals, y_vals, color=color, label=label, **lkw)
    ax.fill_between(x_vals, y_lower, y_upper, color=color, alpha=alpha, **fkw)

    if label is not None:
        ax.legend()

    return ax


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from mimo_pack.testing.data import synthetic_lfp_trials
    
    print("Testing line plotting functions...")
    
    # Generate synthetic trial data
    lfp_trials = synthetic_lfp_trials(n_trials=20, fs=1000, n_chans=4, 
                                      pre=0.2, post=0.5, seed=42)
    
    # Test plot_mean_shaded with different error types
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Select one channel for testing
    single_chan = lfp_trials.isel(channel=0)
    
    # Test 1: SEM
    print("1. Testing plot_mean_shaded() with SEM...")
    try:
        plot_mean_shaded(single_chan, plot_dim='time', err='sem', 
                        color='blue', label='SEM', ax=axes[0])
        axes[0].set_title('Mean ± SEM')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude (µV)')
        print("   ✓ SEM test passed")
    except Exception as e:
        print(f"   ✗ SEM test failed: {e}")
    
    # Test 2: Standard deviation
    print("2. Testing plot_mean_shaded() with STD...")
    try:
        plot_mean_shaded(single_chan, plot_dim='time', err='std', 
                        color='red', label='STD', ax=axes[1])
        axes[1].set_title('Mean ± STD')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Amplitude (µV)')
        print("   ✓ STD test passed")
    except Exception as e:
        print(f"   ✗ STD test failed: {e}")
    
    # Test 3: Confidence interval
    print("3. Testing plot_mean_shaded() with 95% CI...")
    try:
        plot_mean_shaded(single_chan, plot_dim='time', err='ci', ci=95.0,
                        color='green', label='95% CI', ax=axes[2])
        axes[2].set_title('Mean ± 95% CI')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Amplitude (µV)')
        print("   ✓ CI test passed")
    except Exception as e:
        print(f"   ✗ CI test failed: {e}")
    
    # Test 4: Custom styling
    print("4. Testing plot_mean_shaded() with custom styling...")
    try:
        line_kwargs = {'linewidth': 3, 'linestyle': '--'}
        fill_kwargs = {'alpha': 0.1}
        
        plot_mean_shaded(single_chan, plot_dim='time', err='sem', 
                        color='purple', alpha=0.2,
                        line_kwargs=line_kwargs, fill_kwargs=fill_kwargs,
                        label='Custom Style', ax=axes[3])
        axes[3].set_title('Custom Styling')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Amplitude (µV)')
        print("   ✓ Custom styling test passed")
    except Exception as e:
        print(f"   ✗ Custom styling test failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Test with multi-channel overlay
    print("5. Testing plot_mean_shaded() with multiple channels...")
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        colors = ['blue', 'red', 'green', 'orange']
        for i in range(min(4, lfp_trials.sizes['channel'])):
            chan_data = lfp_trials.isel(channel=i)
            plot_mean_shaded(chan_data, plot_dim='time', err='sem',
                           color=colors[i], alpha=0.3,
                           label=f'Channel {i+1}', ax=ax)
        
        ax.set_title('Multi-Channel Mean ± SEM')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (µV)')
        ax.legend()
        print("   ✓ Multi-channel test passed")
    except Exception as e:
        print(f"   ✗ Multi-channel test failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    print("All line plotting tests completed!")
