# Plotting functions for image data
# Author: Drew Headley

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def composite_image(data, colors=None, labels=None, background=None,
                                   colorbars=True, ax=None, vmin=None, vmax=None, **kwargs):
    """Create an alpha-blended RGB composite image from multi-channel data.

    Each channel along the third axis of *data* is mapped to a colour
    and the results are alpha-blended into a single RGB image.

    Parameters
    ----------
    data : numpy.ndarray
        3-D array of shape ``(H, W, N)`` where *N* is the number of
        overlay channels.
    colors : array-like of shape ``(N, 3)`` or None, optional
        RGB colour (values 0–1) for each channel. If ``None``, the
        first *N* colours from the ``tab10`` colourmap are used.
    labels : list of str or None, optional
        Labels for each channel (used in colourbar legends). If
        ``None``, generic names ``Data 0, Data 1, …`` are used.
    background : array-like of length 3 or None, optional
        RGB background colour (0–1). Defaults to white.
    colorbars : bool, optional
        Whether to draw a colourbar for each channel. Default is
        ``True``.
    ax : matplotlib.axes.Axes or None, optional
        Axes to draw on. A new figure is created if ``None``.
    **kwargs
        Additional keyword arguments forwarded to
        ``ax.imshow``.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes containing the composite image.
    composite : numpy.ndarray
        The blended RGB image as a float32 array of shape ``(H, W, 3)``.
    """

    n = data.shape[2]
    if colors is None:
        def_map = plt.get_cmap('tab10')
        colors = def_map(np.mod(np.arange(n), 10))[:,:3]  # Get first n colors from the colormap
    else:
        colors = np.array(colors)
        if colors.shape[0] < n:
            raise ValueError("Not enough colors specified for the number of data arrays.")
    
    if labels is None:
        labels = [f"Data {i}" for i in range(n)]

    if ax is None:
        fig, ax = plt.subplots()

    if vmin is not None and vmax is not None:
        data_mins = np.full((1, 1, n), vmin, dtype=float)
        data_maxs = np.full((1, 1, n), vmax, dtype=float)
    else:
        data_mins = np.nanmin(data, axis=(0, 1), keepdims=True)
        data_maxs = np.nanmax(data, axis=(0, 1), keepdims=True)

    # Normalize data to [0,1] for each channel
    data = np.clip((data - data_mins) / (data_maxs - data_mins + 1e-8), 0, 1)

    # Prepare background
    shape = data.shape[:2]
    if background is None:
        back_color = np.ones((1,1,3))
    else:
        try:
            back_color = np.array(background).reshape(1, 1, 3).astype(np.float32)
        except:
            raise ValueError("Background color must be 3 length vector")

    # Alpha blending
    alpha = np.max(data,axis=2,keepdims=True)

    # Create image using additive color mixing
    add_img = np.zeros((shape[0], shape[1], 3), dtype=np.float32)
    for i in range(n):
        add_img += data[:,:,i][:,:,np.newaxis] * colors[i].reshape(1, 1, 3)

    composite = (1 - alpha) * back_color + alpha * add_img
    composite /= np.max(composite)
    ax.imshow(composite, **kwargs)

    # add a colorbar for each data array
    if colorbars:
        for i in range(n):
            cmap = LinearSegmentedColormap.from_list("custom_cmap", [back_color.reshape(-1), colors[i]])
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=data_mins[0,0,i], vmax=data_maxs[0,0,i]))
            sm.set_array([])
            ax.figure.colorbar(sm, ax=ax, orientation='vertical', label=labels[i])

    return ax, composite


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("Testing image plotting functions...")
    
    print("1. Testing composite_image() with synthetic data...")
    
    try:
        np.random.seed(42)
        
        # Create synthetic 3D image data (like multi-channel fluorescence)
        height, width = 100, 120
        n_channels = 3
        
        # Channel 1: Circular pattern (like cell nucleus)
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        channel1 = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (20**2))
        
        # Channel 2: Horizontal gradient (like membrane staining)
        channel2 = np.linspace(0, 1, width)[np.newaxis, :] * np.ones((height, 1))
        channel2 = np.maximum(channel2 - 0.3, 0)  # Threshold
        
        # Channel 3: Random spots (like punctate markers)
        channel3 = np.zeros((height, width))
        for _ in range(15):
            spot_y = np.random.randint(10, height-10)
            spot_x = np.random.randint(10, width-10)
            spot_size = np.random.uniform(3, 8)
            y_spots, x_spots = np.ogrid[:height, :width]
            channel3 += np.exp(-((x_spots - spot_x)**2 + (y_spots - spot_y)**2) / (spot_size**2))
        
        # Stack into 3D array
        data = np.stack([channel1, channel2, channel3], axis=2)
        
        # Test basic composite
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # RGB colors
        labels = ['Nucleus', 'Membrane', 'Markers']
        
        result_ax, composite = composite_image(
            data, colors=colors, labels=labels, ax=ax, colorbars=True
        )
        
        ax.set_title('Composite Image Test - Basic')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        print("   ✓ Basic composite_image() test passed")
        
    except Exception as e:
        print(f"   ✗ Basic composite_image() test failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Test 2: Single channel
    print("2. Testing composite_image() with single channel...")
    
    try:
        # Test with single channel data
        single_channel_data = channel1[:, :, np.newaxis]  # Add third dimension
        
        fig, ax = plt.subplots(figsize=(10, 8))
        result_ax, composite = composite_image(
            single_channel_data, colors=[[1, 1, 0]], labels=['Single Channel'],
            background=[0, 0, 0], ax=ax
        )
        
        ax.set_title('Single Channel Composite')
        print("   ✓ Single channel test passed")
        
    except Exception as e:
        print(f"   ✗ Single channel test failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Test 3: No colorbars
    print("3. Testing composite_image() without colorbars...")
    
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        result_ax, composite = composite_image(
            data, colors=colors, labels=labels, colorbars=False, ax=ax
        )
        
        ax.set_title('Composite Image - No Colorbars')
        print("   ✓ No colorbars test passed")
        
    except Exception as e:
        print(f"   ✗ No colorbars test failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Test 4: Custom background
    print("4. Testing composite_image() with custom background...")
    
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        result_ax, composite = composite_image(
            data, colors=colors, labels=labels,
            background=[0.2, 0.2, 0.2],  # Dark gray background
            colorbars=False, ax=ax
        )
        
        ax.set_title('Composite Image - Dark Background')
        print("   ✓ Custom background test passed")
        
    except Exception as e:
        print(f"   ✗ Custom background test failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Test 5: Two-channel overlay (common in microscopy)
    print("5. Testing composite_image() with two channels...")
    
    try:
        # Create two-channel data
        two_channel_data = data[:, :, :2]  # Take first two channels
        
        fig, ax = plt.subplots(figsize=(10, 8))
        result_ax, composite = composite_image(
            two_channel_data, 
            colors=[[1, 0, 0], [0, 1, 0]],  # Red and green
            labels=['Channel 1', 'Channel 2'],
            ax=ax
        )
        
        ax.set_title('Two-Channel Overlay')
        print("   ✓ Two-channel test passed")
        
    except Exception as e:
        print(f"   ✗ Two-channel test failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    print("All image plotting tests completed!")
