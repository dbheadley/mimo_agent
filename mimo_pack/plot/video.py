import pandas as pd

# Plotting functions for video and related date

import matplotlib.pyplot as plt

def plot_video_data(tbl_path, ax=None, **kwargs):
    """Plot video segments as rectangles on a Matplotlib axis.
    This function reads video segment data from a specified CSV file. It
    identifies segments based on the 'epoch_num' column and plots their
    duration, determined by the 'time' column, as rectangles on a given or
    newly created Matplotlib axis.
    Parameters
    ----------
    tbl_path : str
        Path to the CSV file with video data. The file must contain
        'epoch_num' and 'time' columns.
    ax : matplotlib.axes.Axes, optional
        The axis to plot on. If None, a new figure and axis are created.
    **kwargs
        Arbitrary keyword arguments passed to `matplotlib.patches.Rectangle`
        to control the properties of the plotted segments (e.g., color, alpha).
    Returns
    -------
    matplotlib.axes.Axes
        The axis with the plotted video segments.
    """
    
    # confirm that highspeed camera segments are identified that were aligned with neural data
    vid_df = pd.read_csv(tbl_path)

    # display video data
    segs_df = vid_df.groupby('epoch_num')['time'].agg(['min', 'max']).reset_index()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 2))

    # Set default patch properties, which can be overridden by kwargs
    patch_defaults = {'color': 'red', 'alpha': 0.5}
    patch_defaults.update(kwargs)

    # Plot rectangles for each video segment
    for start, end in zip(segs_df['min'], segs_df['max']):
        ax.add_patch(plt.Rectangle((start, 0), end - start, 1, **patch_defaults))

    ax.set_xlim(0, vid_df['time'].max())
    ax.set_ylim(-1, 2)
    
    return ax


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import tempfile
    import os
    
    print("Testing video plotting functions...")
    
    print("1. Testing plot_video_data() with synthetic data...")
    
    try:
        # Create synthetic video data
        np.random.seed(42)
        
        # Simulate video recording with multiple epochs
        n_epochs = 5
        data_list = []
        
        for epoch in range(n_epochs):
            # Each epoch has different duration and timing
            start_time = epoch * 120 + np.random.uniform(0, 20)  # Start times with some jitter
            duration = np.random.uniform(60, 100)  # Random duration 1-1.7 minutes
            
            # Create time points within this epoch
            n_frames = int(duration * 30)  # 30 fps
            times = np.linspace(start_time, start_time + duration, n_frames)
            
            # Add some data for each frame
            for time in times:
                data_list.append({
                    'epoch_num': epoch,
                    'time': time,
                    'frame_num': len(data_list),  # Sequential frame number
                })
        
        # Create DataFrame
        vid_df = pd.DataFrame(data_list)
        
        # Save to temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            vid_df.to_csv(tmp_file.name, index=False)
            tmp_path = tmp_file.name
        
        try:
            # Test the plotting function
            fig, ax = plt.subplots(figsize=(14, 4))
            plot_video_data(tmp_path, ax=ax)
            ax.set_title('Video Segments Test - Basic')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Video Epochs')
            
            print("   ✓ Basic plot_video_data() test passed")
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
            
    except Exception as e:
        print(f"   ✗ Basic plot_video_data() test failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Test 2: Custom styling
    print("2. Testing plot_video_data() with custom styling...")
    
    try:
        # Create new synthetic data with different pattern
        data_list = []
        
        # Simulate shorter, more frequent video segments
        for epoch in range(8):
            start_time = epoch * 45 + np.random.uniform(0, 10)
            duration = np.random.uniform(20, 40)  # Shorter segments
            
            n_frames = int(duration * 30)
            times = np.linspace(start_time, start_time + duration, n_frames)
            
            for time in times:
                data_list.append({
                    'epoch_num': epoch,
                    'time': time,
                })
        
        vid_df = pd.DataFrame(data_list)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            vid_df.to_csv(tmp_file.name, index=False)
            tmp_path = tmp_file.name
        
        try:
            fig, axes = plt.subplots(2, 1, figsize=(14, 8))
            
            # Test with different colors and alpha
            plot_video_data(tmp_path, ax=axes[0], color='blue', alpha=0.7)
            axes[0].set_title('Blue Segments')
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylabel('Video Epochs')
            
            plot_video_data(tmp_path, ax=axes[1], color='green', alpha=0.3)
            axes[1].set_title('Green Segments (Low Alpha)')
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Video Epochs')
            
            print("   ✓ Custom styling test passed")
            
        finally:
            os.unlink(tmp_path)
            
    except Exception as e:
        print(f"   ✗ Custom styling test failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # Test 3: Overlapping video data
    print("3. Testing plot_video_data() with overlapping segments...")
    
    try:
        # Create overlapping video segments
        data_list = []
        
        base_times = np.linspace(0, 300, 9000)  # 5 minutes at 30 fps
        
        # Create overlapping epochs
        epoch_ranges = [
            (0, 100),    # First 100 seconds
            (80, 180),   # Overlapping: 80-180 seconds  
            (160, 260),  # Overlapping: 160-260 seconds
            (240, 300),  # Final 60 seconds
        ]
        
        for epoch_num, (start, end) in enumerate(epoch_ranges):
            epoch_times = base_times[(base_times >= start) & (base_times <= end)]
            
            for time in epoch_times:
                data_list.append({
                    'epoch_num': epoch_num,
                    'time': time,
                })
        
        vid_df = pd.DataFrame(data_list)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            vid_df.to_csv(tmp_file.name, index=False)
            tmp_path = tmp_file.name
        
        try:
            fig, ax = plt.subplots(figsize=(16, 6))
            plot_video_data(tmp_path, ax=ax, color='purple', alpha=0.6)
            ax.set_title('Overlapping Video Segments')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Video Epochs')
            
            print("   ✓ Overlapping segments test passed")
            
        finally:
            os.unlink(tmp_path)
            
    except Exception as e:
        print(f"   ✗ Overlapping segments test failed: {e}")
    
    plt.tight_layout()
    plt.show()
    
    print("All video plotting tests completed!")
