# Univariate plotting functions
# Author: Drew Headley
# Date: 2024-06-17

import matplotlib.pyplot as plt
import numpy as np

def violinplot_log(dataset, ax=None, **kwargs):
    """
    Create a violin plot of the dataset with a log scale on the y-axis.
    
    Parameters
    ----------
    dataset : array-like
        The dataset to plot.
    **kwargs : dict
        Additional keyword arguments to pass to the violinplot function.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object containing the plot.
    """

    # create a new figure if axes are not provided
    if ax is None:
        fig, ax = plt.subplots()

    # log transfom the data
    # get range of data values, set min and max to nearest power of 10
    if type(dataset) == list:
        dataset = [np.log10(x) for x in dataset]
        data_min = np.floor(np.min([np.min(x) for x in dataset]))
        data_max = np.ceil(np.max([np.max(x) for x in dataset]))
    else:
        dataset = np.log10(dataset)
        data_min = np.floor(np.min(dataset))
        data_max = np.ceil(np.max(dataset))
    
    # create a log scale for the y-axis
    pow10 = np.power(10, np.arange(data_min, data_max+1))
    yticks = (np.arange(1, 10) * pow10[:, np.newaxis]).ravel()

    # create the plot
    ax.violinplot(dataset, **kwargs)

    # set the y-axis scale and labels
    ax.set_yticks(np.log10(yticks))

    # label the y-ticks that are in pow10
    yticklabels = [f'{float(y)}' if y in pow10 else '' for y in yticks]
    ax.set_yticklabels(yticklabels)

    return ax


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("Testing univariate plotting functions...")
    
    # Test 1: Single dataset violin plot
    print("1. Testing violinplot_log() with single dataset...")
    
    try:
        np.random.seed(42)
        
        # Generate log-normal data (common in neuroscience, e.g., firing rates)
        data_single = np.random.lognormal(mean=1, sigma=1, size=1000)
        
        fig, ax = plt.subplots(figsize=(6, 8))
        violinplot_log(data_single, ax=ax)
        ax.set_title('Log Scale Violin Plot - Single Dataset')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Value (log scale)')
        
        print("   ✓ Single dataset test passed")
        
    except Exception as e:
        print(f"   ✗ Single dataset test failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Test 2: Multiple datasets
    print("2. Testing violinplot_log() with multiple datasets...")
    
    try:
        # Generate multiple datasets with different parameters
        datasets = [
            np.random.lognormal(mean=0.5, sigma=0.8, size=500),  # Lower values
            np.random.lognormal(mean=1.0, sigma=0.6, size=500),  # Medium values
            np.random.lognormal(mean=1.5, sigma=1.0, size=500),  # Higher values
            np.random.lognormal(mean=0.8, sigma=1.2, size=500),  # High variance
        ]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        violinplot_log(datasets, ax=ax)
        ax.set_title('Log Scale Violin Plot - Multiple Datasets')
        ax.set_xlabel('Dataset Index')
        ax.set_ylabel('Value (log scale)')
        
        # Add custom x-tick labels
        ax.set_xticks(range(1, len(datasets) + 1))
        ax.set_xticklabels(['Low', 'Medium', 'High', 'Variable'])
        
        print("   ✓ Multiple datasets test passed")
        
    except Exception as e:
        print(f"   ✗ Multiple datasets test failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Test 3: Neuroscience-like data (firing rates)
    print("3. Testing violinplot_log() with firing rate data...")
    
    try:
        # Simulate firing rates for different neuron populations
        # Population 1: Low firing interneurons
        low_firing = np.random.gamma(shape=1, scale=2, size=200)  # Mean ~2 Hz
        
        # Population 2: Medium firing pyramidal cells  
        med_firing = np.random.gamma(shape=2, scale=5, size=300)  # Mean ~10 Hz
        
        # Population 3: High firing fast-spiking interneurons
        high_firing = np.random.gamma(shape=3, scale=15, size=150)  # Mean ~45 Hz
        
        firing_data = [low_firing, med_firing, high_firing]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        violinplot_log(firing_data, ax=ax, showmeans=True, showmedians=True)
        ax.set_title('Firing Rate Distributions (Log Scale)')
        ax.set_xlabel('Neuron Population')
        ax.set_ylabel('Firing Rate (Hz, log scale)')
        
        # Custom labels
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Low Firing\nInterneurons', 'Pyramidal\nCells', 'Fast-Spiking\nInterneurons'])
        
        print("   ✓ Firing rate data test passed")
        
    except Exception as e:
        print(f"   ✗ Firing rate data test failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Test 4: Wide range data (spanning multiple orders of magnitude)
    print("4. Testing violinplot_log() with wide range data...")
    
    try:
        # Create data spanning several orders of magnitude
        wide_range_data = np.concatenate([
            np.random.uniform(0.001, 0.01, 200),    # 10^-3 to 10^-2
            np.random.uniform(0.1, 1, 200),         # 10^-1 to 10^0  
            np.random.uniform(10, 100, 200),        # 10^1 to 10^2
            np.random.uniform(1000, 10000, 200),    # 10^3 to 10^4
        ])
        
        fig, ax = plt.subplots(figsize=(8, 10))
        violinplot_log(wide_range_data, ax=ax, showmeans=True)
        ax.set_title('Wide Range Data (Log Scale)')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Value (log scale)')
        
        print("   ✓ Wide range data test passed")
        
    except Exception as e:
        print(f"   ✗ Wide range data test failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    print("All univariate plotting tests completed!")
