# Unit data plotting functions
# Author: Drew Headley
# Date: 2025-06-15

import matplotlib.pyplot as plt
import numpy as np
from mimo_pack.plot.colors import MIMO_SEQ_CMAP

def probe_units(spks_nap, jitter=10, labels=True, cmap=MIMO_SEQ_CMAP, 
                spread_scale=1, ax=None, **kwargs):
    """
    Plots the distribution of spikes on a probe.

    Parameters
    ----------
    spks_nap : TsGroup object
        The spike data containing the units and their metadata 
        calculated by the pynapple_spikes_qc function.
    jitter : numeric, optional
        Amount of jitter to apply to the x and y positions of the
          units (default is 10).
    labels : bool, optional
        Whether to add labels to the axes and colorbar (default is True).
    cmap : str, optional
        Colormap to use for the scatter plot of SU rates (default is 'viridis').
    spread_scale : numeric, optional
        Scale factor for the size of the SU units in the scatter plot (default is 1).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (default is None, creates a new figure).
    **kwargs : additional keyword arguments
        Additional parameters for `plt.scatter`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes on which the waveforms were plotted.
    """
    
    # create a new figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots()

    # check that metadata contains the required columns
    required_columns = ['x', 'y', 'class', 'rate', 'WaveformSpread']
    if not all(col in spks_nap.metadata.columns for col in required_columns):
        raise ValueError(f"Metadata must contain the following columns: {required_columns}")
    
    metadata = spks_nap.metadata
    mu_meta = metadata[metadata['class'] == 'MU']
    su_meta = metadata[metadata['class'] == 'SU']
    num_mu = mu_meta.shape[0]
    num_su = su_meta.shape[0]
    ax.scatter(mu_meta['x']+np.random.randn(num_mu)*jitter, 
               mu_meta['y']+np.random.randn(num_mu)*jitter,
               c='gray', s=20, marker='+', 
               label='MU units', **kwargs)
    ax.scatter(su_meta['x']+np.random.randn(num_su)*jitter, 
               su_meta['y']+np.random.randn(num_su)*jitter, 
               c=np.log10(su_meta['rate']), 
               s=su_meta['WaveformSpread']*spread_scale, 
               cmap=cmap, label='SU units', **kwargs)
    
    if np.max(mu_meta['x'])- np.min(mu_meta['x']) < 250:
        ax.set_xlim(np.min(mu_meta['x'])-250, np.max(mu_meta['x'])+250)
        
    ax.legend()
    cbar = plt.colorbar(ax.collections[1], ax=ax)
    
    if labels:
        cbar.set_label('log10(rate)')
        ax.set_title('Distribution of units on probe')
        ax.set_xlabel('x position (um)')
        ax.set_ylabel('y position (um)')
        
    return ax


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import pynapple as nap
    
    print("Testing unit plotting functions...")
    
    # Create synthetic spike data with metadata
    print("1. Testing probe_units() with synthetic data...")
    
    try:
        # Create synthetic unit metadata
        n_units = 20
        x_pos = np.random.uniform(0, 400, n_units)  # Probe x positions
        y_pos = np.random.uniform(0, 3000, n_units)  # Probe y positions  
        firing_rates = np.random.uniform(0.1, 50, n_units)  # Hz
        waveform_spreads = np.random.uniform(10, 100, n_units)  # Spatial spread
        unit_classes = np.random.choice(['SU', 'MU'], n_units, p=[0.7, 0.3])  # Single/Multi unit
        
        metadata_dict = {
            'x': x_pos,
            'y': y_pos, 
            'rate': firing_rates,
            'WaveformSpread': waveform_spreads,
            'class': unit_classes
        }
        metadata_df = pd.DataFrame(metadata_dict)
        
        # Create synthetic spike times
        spike_times = {}
        for i in range(n_units):
            n_spikes = int(firing_rates[i] * 10)  # 10 second recording
            spike_times[i] = np.sort(np.random.uniform(0, 10, n_spikes))
        
        # Create pynapple TsGroup
        spks_nap = nap.TsGroup(spike_times, metadata=metadata_df)
        
        # Test basic plotting
        fig, ax = plt.subplots(figsize=(10, 12))
        result_ax = probe_units(spks_nap, ax=ax, jitter=10)
        ax.set_title('Synthetic Unit Distribution Test')
        
        print("   ✓ Basic probe_units() test passed")
        
    except Exception as e:
        print(f"   ✗ probe_units() test failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Test with different parameters
    print("2. Testing probe_units() with custom parameters...")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Test with different jitter
        probe_units(spks_nap, ax=axes[0], jitter=30, cmap='plasma', spread_scale=2)
        axes[0].set_title('High Jitter, Custom Colormap')
        
        # Test without labels
        probe_units(spks_nap, ax=axes[1], labels=False, jitter=5, cmap='coolwarm')
        axes[1].set_title('No Labels, Low Jitter')
        
        print("   ✓ Custom parameter tests passed")
        
    except Exception as e:
        print(f"   ✗ Custom parameter tests failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Test edge cases
    print("3. Testing probe_units() edge cases...")
    try:
        # Create data with only SU units
        su_only_meta = metadata_df[metadata_df['class'] == 'SU'].copy()
        su_spike_times = {i: spike_times[i] for i in su_only_meta.index}
        spks_su_only = nap.TsGroup(su_spike_times, metadata=su_only_meta)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        probe_units(spks_su_only, ax=ax)
        ax.set_title('SU Units Only Test')
        
        print("   ✓ SU-only test passed")
        
    except Exception as e:
        print(f"   ✗ SU-only test failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    print("All unit plotting tests completed!")

