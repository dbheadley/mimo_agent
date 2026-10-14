# Histogram plotting functions
# Author: Drew Headley
# Date: 2024-06-17

import matplotlib.pyplot as plt
import numpy as np

def stairs_fl(values, edges, fill_color=None, edge_color=None,
              fill_alpha=0.5, edge_alpha=1.0, ax=None, **kwargs):
    """
    Plots a step function histogram with filled areas.

    Parameters
    ----------
    values : np.ndarray
        The heights of the steps.
    edges : np.ndarray
        The edges of the bins.
    fill_color : str or tuple, optional
        Color to fill the area under the steps (default is None).
    edge_color : str or tuple, optional
        Color of the step edges (default is None).
    fill_alpha : float, optional
        Alpha value for the filled area (default is 0.5).
    edge_alpha : float, optional
        Alpha value for the step edges (default is 1.0).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (default is None, creates a new figure).
    **kwargs : additional keyword arguments
        Additional parameters for `plt.plot`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes on which the histogram was plotted.
    """
    
    if ax is None:
        fig, ax = plt.subplots()

    if fill_color is not None:
        ax.stairs(values, edges, color=fill_color, alpha=fill_alpha, 
                  fill=True, **kwargs)

    if edge_color is not None:
        ax.stairs(values, edges, color=edge_color, alpha=edge_alpha,
                  fill=False, **kwargs)

    return ax


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("Testing histogram plotting functions...")
    
    # Generate synthetic histogram data
    np.random.seed(42)
    data = np.random.normal(0, 1, 1000)
    
    # Create histogram bins
    counts, bin_edges = np.histogram(data, bins=30)
    
    print("1. Testing stairs_fl() with different fill options...")
    
    # Test 1: Fill only
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    try:
        stairs_fl(counts, bin_edges, fill_color='blue', ax=axes[0,0])
        axes[0,0].set_title('Fill Only')
        axes[0,0].set_xlabel('Value')
        axes[0,0].set_ylabel('Count')
        print("   ✓ Fill only test passed")
    except Exception as e:
        print(f"   ✗ Fill only test failed: {e}")
    
    # Test 2: Edge only
    try:
        stairs_fl(counts, bin_edges, edge_color='red', ax=axes[0,1])
        axes[0,1].set_title('Edge Only')
        axes[0,1].set_xlabel('Value')
        axes[0,1].set_ylabel('Count')
        print("   ✓ Edge only test passed")
    except Exception as e:
        print(f"   ✗ Edge only test failed: {e}")
    
    # Test 3: Both fill and edge
    try:
        stairs_fl(counts, bin_edges, fill_color='lightgreen', edge_color='darkgreen',
                 fill_alpha=0.7, edge_alpha=1.0, ax=axes[1,0])
        axes[1,0].set_title('Fill + Edge')
        axes[1,0].set_xlabel('Value')
        axes[1,0].set_ylabel('Count')
        print("   ✓ Fill + edge test passed")
    except Exception as e:
        print(f"   ✗ Fill + edge test failed: {e}")
    
    # Test 4: Custom alpha values
    try:
        stairs_fl(counts, bin_edges, fill_color='purple', edge_color='black',
                 fill_alpha=0.3, edge_alpha=0.8, ax=axes[1,1])
        axes[1,1].set_title('Custom Alpha')
        axes[1,1].set_xlabel('Value')
        axes[1,1].set_ylabel('Count')
        print("   ✓ Custom alpha test passed")
    except Exception as e:
        print(f"   ✗ Custom alpha test failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Test with different data types
    print("2. Testing stairs_fl() with different data types...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Exponential distribution
    try:
        exp_data = np.random.exponential(2, 1000)
        exp_counts, exp_edges = np.histogram(exp_data, bins=25)
        
        stairs_fl(exp_counts, exp_edges, fill_color='orange', edge_color='red', ax=axes[0])
        axes[0].set_title('Exponential Distribution')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Count')
        print("   ✓ Exponential data test passed")
    except Exception as e:
        print(f"   ✗ Exponential data test failed: {e}")
    
    # Uniform distribution
    try:
        uniform_data = np.random.uniform(-3, 3, 1000)
        uniform_counts, uniform_edges = np.histogram(uniform_data, bins=20)
        
        stairs_fl(uniform_counts, uniform_edges, fill_color='cyan', edge_color='blue', ax=axes[1])
        axes[1].set_title('Uniform Distribution')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Count')
        print("   ✓ Uniform data test passed")
    except Exception as e:
        print(f"   ✗ Uniform data test failed: {e}")
    
    # Bimodal distribution
    try:
        bimodal_data = np.concatenate([
            np.random.normal(-2, 0.5, 500),
            np.random.normal(2, 0.5, 500)
        ])
        bimodal_counts, bimodal_edges = np.histogram(bimodal_data, bins=30)
        
        stairs_fl(bimodal_counts, bimodal_edges, fill_color='pink', edge_color='purple', ax=axes[2])
        axes[2].set_title('Bimodal Distribution')
        axes[2].set_xlabel('Value')
        axes[2].set_ylabel('Count')
        print("   ✓ Bimodal data test passed")
    except Exception as e:
        print(f"   ✗ Bimodal data test failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    print("All histogram plotting tests completed!")
