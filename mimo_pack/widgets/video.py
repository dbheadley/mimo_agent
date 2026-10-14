"""Video alignment widget for MIMO preprocessing."""

import ipywidgets as widgets
from IPython.display import display, clear_output
import json
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector


class VideoAlignmentWidget:
    """Interactive widget for video preprocessing with inline ROI selection using matplotlib"""
    
    REQUIRED_ROIS = 8
    
    def __init__(self, session):
        """Initialize the video alignment widget.

        Parameters
        ----------
        session : Session
            A loaded ``Session`` object whose directory contains
            behavioral video or TIF image sequences to align.
        """
        self.session = session
        self.vid_path = None
        self.tbl_path = None
        self.cap = None
        self.frame_count = 0
        self.current_frame = None
        self.rois = []  # List of (x, y, w, h) tuples
        self.fig = None
        self.ax = None
        self.rect_selector = None
        self.im_display = None  # Store imshow object for efficient updates
        self.roi_patches = []  # Store ROI patches
        self.roi_texts = []  # Store ROI text labels
        self.video_loaded = False
        self.create_widgets()
    
    def create_widgets(self):
        """Create all UI widgets"""
        # Title
        title = widgets.HTML("<h3>🎥 Interactive Video Alignment</h3>")
        
        # Session info display
        session_info = widgets.HTML(
            f"<div style='background:#f0f8ff; padding:10px; border:1px solid #ccc;'>"
            f"<b>Session Directory:</b> {self.session.dir}<br>"
            f"<b>Config File:</b> {self.session.config_path}"
            f"</div>"
        )
        
        # Get current parameters from config
        video_params = self.session.config['parameters'].get('video_processing', {})
        current_img_dir = video_params.get('image_directory', str(self.session.dir))
        current_sync_channel = video_params.get('sync_channel', {'channel': [384]})
        current_ref_probe = video_params.get('reference_probe_index', 0)
        
        # Parse sync channel
        if isinstance(current_sync_channel.get('channel', [384]), list):
            sync_chan_val = current_sync_channel['channel'][0] if current_sync_channel['channel'] else 384
        else:
            sync_chan_val = current_sync_channel.get('channel', 384)
        
        # Configuration Section
        config_label = widgets.HTML("<h4>🔧 Configuration</h4>")
        
        self.img_dir_input = widgets.Text(
            value=current_img_dir,
            description='Image Dir:',
            placeholder='Path to TIF images directory',
            layout=widgets.Layout(width='600px')
        )
        
        self.sync_channel_input = widgets.IntText(
            value=sync_chan_val,
            description='Sync Channel:',
            min=0, max=1000,
            layout=widgets.Layout(width='200px')
        )
        
        self.ref_probe_input = widgets.IntText(
            value=current_ref_probe,
            description='Ref Probe:',
            min=0, max=10,
            layout=widgets.Layout(width='200px')
        )
        
        config_help = widgets.HTML(
            "<small>💡 Image directory with TIF files, sync channel (typically 384), reference probe index</small>"
        )
        
        # Video path input for direct loading
        self.video_path_input = widgets.Text(
            value=video_params.get('video_path', ''),
            description='Video Path:',
            placeholder='Path to existing .mp4 video (optional)',
            layout=widgets.Layout(width='600px')
        )
        
        # Import from Session button
        self.import_btn = widgets.Button(
            description='📥 Import from Session',
            button_style='warning',
            layout=widgets.Layout(width='180px')
        )
        
        # Convert Images button
        self.convert_btn = widgets.Button(
            description='🎬 Convert Images to Video',
            button_style='info',
            layout=widgets.Layout(width='200px')
        )
        
        # Load Video button
        self.load_video_btn = widgets.Button(
            description='📹 Load Video',
            button_style='success',
            layout=widgets.Layout(width='150px')
        )
        
        self.retry_convert_btn = widgets.Button(
            description='🔄 Retry Conversion',
            button_style='warning',
            layout=widgets.Layout(width='150px', display='none')
        )
        
        # Frame Navigation Section (initially disabled)
        nav_label = widgets.HTML("<h4>🖼️ Frame Navigation</h4>")
        
        self.frame_slider = widgets.IntSlider(
            value=0, min=0, max=100, step=1,
            description='Frame:',
            continuous_update=False,
            layout=widgets.Layout(width='500px'),
            disabled=True
        )
        
        self.jump_frame_input = widgets.IntText(
            value=0,
            description='Jump to:',
            layout=widgets.Layout(width='150px'),
            disabled=True
        )
        
        self.brightness_slider = widgets.FloatSlider(
            value=1.0, min=0.1, max=10.0, step=0.1,
            description='Brightness:',
            continuous_update=False,
            layout=widgets.Layout(width='300px'),
            disabled=True
        )
        
        self.contrast_slider = widgets.FloatSlider(
            value=1.0, min=0.1, max=3.0, step=0.1,
            description='Contrast:',
            continuous_update=False,
            layout=widgets.Layout(width='300px'),
            disabled=True
        )
        
        self.save_video_btn = widgets.Button(
            description='💾 Save Adjusted Video',
            button_style='primary',
            layout=widgets.Layout(width='200px'),
            disabled=True
        )
        
        # Refresh display button
        self.refresh_btn = widgets.Button(
            description='🔄 Refresh Display',
            button_style='info',
            layout=widgets.Layout(width='150px'),
            disabled=True
        )
        
        nav_help = widgets.HTML(
            "<small>💡 Navigate frames to find a clear view of barcode LEDs, adjust brightness and contrast as needed</small>"
        )
        
        # ROI Selection Section
        roi_label = widgets.HTML("<h4>🎯 ROI Selection (Draw 8 barcode regions)</h4>")
        
        self.roi_status = widgets.HTML(
            "<span style='color:gray'>⏳ Load video to enable ROI selection</span>"
        )
        
        self.clear_rois_btn = widgets.Button(
            description='🗑️ Clear All ROIs',
            button_style='danger',
            layout=widgets.Layout(width='150px'),
            disabled=True
        )
        
        roi_instructions = widgets.HTML(
            "<small>💡 Click and drag on the image to draw rectangles around each barcode LED "
            "(start from 2nd LED on left, draw 8 total). Edit coordinates in table below.</small>"
        )
        
        # Figure output area for matplotlib
        self.figure_output = widgets.Output(layout=widgets.Layout(
            width='100%', min_height='450px', border='1px solid #ccc'
        ))
        
        # ROI Table container
        roi_table_label = widgets.HTML("<h5>📋 ROI Coordinates (editable):</h5>")
        self.roi_table_container = widgets.VBox(
            [], layout=widgets.Layout(max_height='250px', overflow_y='auto')
        )
        
        self.update_display_btn = widgets.Button(
            description='🔄 Update Display',
            button_style='info',
            layout=widgets.Layout(width='150px'),
            disabled=True
        )
        
        # Process & Align Section
        process_label = widgets.HTML("<h4>⚡ Process & Align Video</h4>")
        
        self.process_btn = widgets.Button(
            description='🎯 Process & Align Video',
            button_style='success',
            layout=widgets.Layout(width='200px'),
            disabled=True
        )
        
        self.save_config_btn = widgets.Button(
            description='💾 Save Config',
            button_style='',
            layout=widgets.Layout(width='150px')
        )
        
        self.clear_output_btn = widgets.Button(
            description='🧹 Clear Output',
            button_style='warning',
            layout=widgets.Layout(width='150px')
        )
        
        # Output area
        self.output = widgets.Output()
        
        # QC figure output area
        self.qc_output = widgets.Output()
        
        # Event handlers
        self.import_btn.on_click(self.on_import_from_session)
        self.convert_btn.on_click(self.on_convert_images)
        self.load_video_btn.on_click(self.on_load_video)
        self.retry_convert_btn.on_click(self.on_convert_images)
        self.frame_slider.observe(self.on_frame_change, names='value')
        self.jump_frame_input.observe(self.on_jump_frame, names='value')
        self.brightness_slider.observe(self.on_brightness_change, names='value')
        self.contrast_slider.observe(self.on_contrast_change, names='value')
        self.save_video_btn.on_click(self.on_save_video)
        self.refresh_btn.on_click(self.on_refresh_display)
        self.clear_rois_btn.on_click(self.on_clear_rois)
        self.update_display_btn.on_click(self.on_update_display)
        self.process_btn.on_click(self.on_process_align)
        self.save_config_btn.on_click(self.on_save_config)
        self.clear_output_btn.on_click(self.on_clear_output)
        
        # Layout
        self.container = widgets.VBox([
            title,
            session_info,
            config_label,
            self.img_dir_input,
            self.video_path_input,
            widgets.HBox([self.sync_channel_input, self.ref_probe_input]),
            config_help,
            widgets.HBox([self.import_btn, self.convert_btn, self.load_video_btn, self.retry_convert_btn]),
            widgets.HTML("<hr>"),
            nav_label,
            widgets.HBox([self.frame_slider, self.jump_frame_input, self.refresh_btn]),
            widgets.HBox([self.brightness_slider, self.contrast_slider]),
            self.save_video_btn,
            nav_help,
            widgets.HTML("<hr>"),
            roi_label,
            self.roi_status,
            roi_instructions,
            self.figure_output,
            widgets.HBox([self.clear_rois_btn, self.update_display_btn]),
            roi_table_label,
            self.roi_table_container,
            widgets.HTML("<hr>"),
            process_label,
            widgets.HBox([self.process_btn, self.save_config_btn, self.clear_output_btn]),
            self.output,
            self.qc_output
        ])
    
    def on_clear_output(self, btn):
        """Clear the output area"""
        with self.output:
            clear_output()
    
    def on_import_from_session(self, btn):
        """Import video settings from session configuration"""
        with self.output:
            print("\n" + "="*50)
            print("📥 Importing settings from session configuration...")
            
            video_params = self.session.config.get('parameters', {}).get('video_processing', {})
            
            if not video_params:
                print("⚠️ No saved video configuration found in session")
                return
            
            # Import configuration values
            if 'image_directory' in video_params:
                self.img_dir_input.value = video_params['image_directory']
                print(f"✅ Loaded image directory: {video_params['image_directory']}")
            
            sync_channel = video_params.get('sync_channel', {})
            if isinstance(sync_channel.get('channel', []), list):
                chan_val = sync_channel['channel'][0] if sync_channel['channel'] else 384
            else:
                chan_val = sync_channel.get('channel', 384)
            self.sync_channel_input.value = chan_val
            print(f"✅ Loaded sync channel: {chan_val}")
            
            if 'reference_probe_index' in video_params:
                self.ref_probe_input.value = video_params['reference_probe_index']
                print(f"✅ Loaded reference probe: {video_params['reference_probe_index']}")
            
            # Import ROIs if available
            if 'rois' in video_params and video_params['rois']:
                self.rois = [tuple(roi) for roi in video_params['rois']]
                print(f"✅ Loaded {len(self.rois)} ROIs from session")
            
            # Import video/table paths if available
            if 'video_path' in video_params:
                self.vid_path = video_params['video_path']
                self.video_path_input.value = self.vid_path
                print(f"✅ Loaded video path: {self.vid_path}")
            
            if 'timing_table_path' in video_params:
                self.tbl_path = video_params['timing_table_path']
                print(f"✅ Loaded timing table path: {self.tbl_path}")
            
            print("🎯 Settings imported successfully!")
            print("💡 Click 'Load Video' to load the video and display frames")
    
    def on_convert_images(self, btn):
        """Convert TIF images to video"""
        with self.output:
            print("\n" + "="*50)
            self.retry_convert_btn.layout.display = 'none'
            
            img_dir = self.img_dir_input.value.strip().strip('"').strip("'")
            self.img_dir_input.value = img_dir
            
            if not img_dir:
                print("❌ Please enter an image directory path")
                return
            
            if not Path(img_dir).exists():
                print(f"❌ Directory not found: {img_dir}")
                self.retry_convert_btn.layout.display = 'inline-block'
                return
            
            print(f"🎬 Converting images from: {img_dir}")
            print("⏳ This may take a few minutes for large image sets...")
            
            try:
                from mimo_pack.preprocess.video import tif_to_video
                
                self.vid_path, self.tbl_path = tif_to_video(img_dir)
                self.video_path_input.value = self.vid_path
                
                print(f"✅ Video created: {self.vid_path}")
                print(f"✅ Timing table: {self.tbl_path}")
                print("💡 Click 'Load Video' to display frames and select ROIs")
                
            except Exception as e:
                print(f"❌ Conversion failed: {e}")
                print("💡 Check that the directory contains valid TIF files")
                print("💡 Ensure sufficient disk space and write permissions")
                self.retry_convert_btn.layout.display = 'inline-block'
    
    def on_load_video(self, btn):
        """Load video from path"""
        with self.output:
            print("\n" + "="*50)
            
            # Get video path from input or stored path
            vid_path = self.video_path_input.value.strip()
            if not vid_path and self.vid_path:
                vid_path = self.vid_path
            
            if not vid_path:
                print("❌ No video path specified")
                print("💡 Enter a video path or convert images first")
                return
            
            if not Path(vid_path).exists():
                print(f"❌ Video file not found: {vid_path}")
                return
            
            self.vid_path = vid_path
            
            # Try to find the timing table (same base name with .csv extension)
            vid_path_obj = Path(vid_path)
            tbl_path = vid_path_obj.with_suffix('.csv')
            if tbl_path.exists():
                self.tbl_path = str(tbl_path)
                print(f"📋 Found timing table: {tbl_path.name}")
            elif not self.tbl_path:
                # Check if tbl_path was set from session import
                print("⚠️ No timing table (.csv) found alongside video")
                print("💡 If you converted images, the table should exist")
                print("💡 You may need to convert images first or specify table path")
            
            print(f"📹 Loading video: {vid_path}")
            self._load_video(vid_path)
    
    def _load_video(self, vid_path):
        """Load video and enable frame navigation"""
        try:
            if self.cap is not None:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(vid_path)
            
            if not self.cap.isOpened():
                raise IOError(f"Could not open video: {vid_path}")
            
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_loaded = True
            
            # Enable navigation controls
            self.frame_slider.max = self.frame_count - 1
            self.frame_slider.value = 0
            self.frame_slider.disabled = False
            self.jump_frame_input.disabled = False
            self.brightness_slider.disabled = False
            self.contrast_slider.disabled = False
            self.save_video_btn.disabled = False
            self.refresh_btn.disabled = False
            self.clear_rois_btn.disabled = False
            self.update_display_btn.disabled = False
            
            with self.output:
                print(f"✅ Video loaded: {self.frame_count} frames")
                print("🎯 Initializing frame display...")
            
            # Initialize matplotlib figure and display first frame
            self._init_matplotlib_figure()
            
            # Update ROI table and status
            self._update_roi_table()
            self._update_roi_status()
            self._update_process_button_state()
            
            with self.output:
                print("✅ Ready! Draw ROI rectangles on the image above.")
            
        except Exception as e:
            with self.output:
                print(f"❌ Failed to load video: {e}")
                import traceback
                traceback.print_exc()
    
    def _init_matplotlib_figure(self):
        """Initialize matplotlib figure with RectangleSelector"""
        # Clear existing figure output
        with self.figure_output:
            clear_output(wait=True)
            
            # Close any existing figure
            if self.fig is not None:
                plt.close(self.fig)
            
            # Create new figure
            self.fig, self.ax = plt.subplots(figsize=(12, 6))
            
            # Read and display first frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            
            if ret:
                frame = self._apply_adjustments(frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_frame = frame_rgb
                
                self.im_display = self.ax.imshow(frame_rgb)
                self._draw_rois()
                
                self._update_title()
                self.ax.axis('off')
            
            # Initialize RectangleSelector
            self.rect_selector = RectangleSelector(
                self.ax, self._on_roi_select,
                useblit=True,
                button=[1],  # Left mouse button
                minspanx=5, minspany=5,
                spancoords='pixels',
                interactive=True
            )
            
            plt.tight_layout()
            plt.show()
    
    def _draw_rois(self):
        """Draw all ROIs on the current axes"""
        # Remove existing ROI patches and texts
        for patch in self.roi_patches:
            patch.remove()
        for text in self.roi_texts:
            text.remove()
        self.roi_patches = []
        self.roi_texts = []
        
        # Draw new ROIs
        for i, (x, y, w, h) in enumerate(self.rois):
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            self.ax.add_patch(rect)
            self.roi_patches.append(rect)
            
            text = self.ax.text(
                x, y - 5, f'ROI {i}',
                color='red', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )
            self.roi_texts.append(text)
    
    def _on_roi_select(self, eclick, erelease):
        """Callback when ROI is drawn with RectangleSelector"""
        if len(self.rois) >= self.REQUIRED_ROIS:
            with self.output:
                print(f"\n⚠️ Maximum {self.REQUIRED_ROIS} ROIs reached. Clear ROIs to redraw.")
            return
        
        # Extract coordinates
        x1, y1 = int(min(eclick.xdata, erelease.xdata)), int(min(eclick.ydata, erelease.ydata))
        x2, y2 = int(max(eclick.xdata, erelease.xdata)), int(max(eclick.ydata, erelease.ydata))
        w, h = x2 - x1, y2 - y1
        
        if w < 5 or h < 5:
            return  # Ignore tiny selections
        
        # Add new ROI
        self.rois.append((x1, y1, w, h))
        
        with self.output:
            print(f"✅ Added ROI {len(self.rois)-1}: x={x1}, y={y1}, w={w}, h={h}")
        
        # Update displays
        self._update_roi_table()
        self._draw_rois()
        self._update_title()
        self._update_roi_status()
        self._update_process_button_state()
        
        self.fig.canvas.draw_idle()
    
    def _update_title(self):
        """Update the figure title"""
        if self.ax is not None:
            frame_num = self.frame_slider.value
            brightness = self.brightness_slider.value
            contrast = self.contrast_slider.value
            self.ax.set_title(f'Frame {frame_num}/{self.frame_count-1} | Brightness: {brightness:.1f}x | Contrast: {contrast:.1f}x | ROIs: {len(self.rois)}/{self.REQUIRED_ROIS}')
    
    def _apply_adjustments(self, frame):
        """Apply brightness and contrast adjustments to a frame.

        Parameters
        ----------
        frame : numpy.ndarray
            Frame array (BGR or RGB, any channel order); values 0-255.

        Returns
        -------
        numpy.ndarray
            Adjusted frame clipped to uint8 range.
        """
        brightness = self.brightness_slider.value
        contrast = self.contrast_slider.value
        f = frame.astype(np.float32) * brightness
        f = contrast * (f - 128.0) + 128.0
        return np.clip(f, 0, 255).astype(np.uint8)

    def _update_frame_display(self):
        """Update the frame display with current frame and ROIs"""
        if self.cap is None or self.ax is None or self.im_display is None:
            return
        
        frame_num = self.frame_slider.value
        
        # Read frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        
        if not ret:
            with self.output:
                print(f"⚠️ Could not read frame {frame_num}")
            return
        
        # Apply brightness and contrast
        frame = self._apply_adjustments(frame)
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_frame = frame_rgb
        
        # Update image data (more efficient than clearing and redrawing)
        self.im_display.set_data(frame_rgb)
        
        # Update title
        self._update_title()
        
        # Redraw ROIs
        self._draw_rois()
        
        # Refresh canvas
        self.fig.canvas.draw_idle()
    
    def _update_roi_status(self):
        """Update ROI status display"""
        n_rois = len(self.rois)
        
        if n_rois == self.REQUIRED_ROIS:
            self.roi_status.value = f"<span style='color:green; font-weight:bold'>✅ {n_rois} ROIs selected - Ready to process!</span>"
        elif n_rois > 0:
            needed = self.REQUIRED_ROIS - n_rois
            self.roi_status.value = f"<span style='color:orange'>⚠️ {n_rois} ROIs selected - Need {needed} more</span>"
        else:
            self.roi_status.value = "<span style='color:gray'>⏳ Draw rectangles on the image to select barcode regions</span>"
    
    def _update_roi_table(self):
        """Update the editable ROI table"""
        rows = []
        self.roi_widgets = {}
        
        for i, (x, y, w, h) in enumerate(self.rois):
            # Create coordinate input widgets
            x_input = widgets.IntText(value=x, layout=widgets.Layout(width='80px'))
            y_input = widgets.IntText(value=y, layout=widgets.Layout(width='80px'))
            w_input = widgets.IntText(value=w, layout=widgets.Layout(width='80px'))
            h_input = widgets.IntText(value=h, layout=widgets.Layout(width='80px'))
            
            # Store widget references
            self.roi_widgets[i] = {'x': x_input, 'y': y_input, 'w': w_input, 'h': h_input}
            
            # Add observers for coordinate changes
            x_input.observe(lambda change, idx=i: self._on_roi_coord_change(idx), names='value')
            y_input.observe(lambda change, idx=i: self._on_roi_coord_change(idx), names='value')
            w_input.observe(lambda change, idx=i: self._on_roi_coord_change(idx), names='value')
            h_input.observe(lambda change, idx=i: self._on_roi_coord_change(idx), names='value')
            
            # Delete button
            delete_btn = widgets.Button(
                description='🗑️',
                button_style='danger',
                layout=widgets.Layout(width='40px')
            )
            delete_btn.on_click(lambda btn, idx=i: self._on_delete_roi(idx))
            
            row = widgets.HBox([
                widgets.HTML(f"<b>ROI {i}:</b>", layout=widgets.Layout(width='60px')),
                widgets.HTML("x:", layout=widgets.Layout(width='20px')),
                x_input,
                widgets.HTML("y:", layout=widgets.Layout(width='20px')),
                y_input,
                widgets.HTML("w:", layout=widgets.Layout(width='20px')),
                w_input,
                widgets.HTML("h:", layout=widgets.Layout(width='20px')),
                h_input,
                delete_btn
            ])
            rows.append(row)
        
        if not rows:
            rows.append(widgets.HTML("<em>No ROIs defined. Draw rectangles on the image above.</em>"))
        
        self.roi_table_container.children = rows
    
    def _on_roi_coord_change(self, idx):
        """Handle ROI coordinate changes from table"""
        if idx in self.roi_widgets:
            widgets_dict = self.roi_widgets[idx]
            self.rois[idx] = (
                widgets_dict['x'].value,
                widgets_dict['y'].value,
                widgets_dict['w'].value,
                widgets_dict['h'].value
            )
    
    def _on_delete_roi(self, idx):
        """Delete ROI at given index"""
        if 0 <= idx < len(self.rois):
            del self.rois[idx]
            self._update_roi_table()
            if self.video_loaded:
                self._draw_rois()
                self._update_title()
                self.fig.canvas.draw_idle()
            self._update_roi_status()
            self._update_process_button_state()
    
    def _update_process_button_state(self):
        """Enable/disable process button based on ROI count"""
        self.process_btn.disabled = len(self.rois) != self.REQUIRED_ROIS
    
    def on_frame_change(self, change):
        """Handle frame slider change"""
        if self.video_loaded:
            self._update_frame_display()
    
    def on_jump_frame(self, change):
        """Handle jump to frame input"""
        if self.video_loaded:
            frame_num = max(0, min(change['new'], self.frame_count - 1))
            self.frame_slider.value = frame_num
    
    def on_brightness_change(self, change):
        """Handle brightness slider change"""
        if self.video_loaded:
            self._update_frame_display()
    
    def on_contrast_change(self, change):
        """Handle contrast slider change"""
        if self.video_loaded:
            self._update_frame_display()
    
    def on_save_video(self, btn):
        """Save the loaded video with brightness and contrast adjustments applied."""
        with self.output:
            print("\n" + "="*50)
            print("💾 Saving adjusted video...")
            
            if not self.vid_path or not Path(self.vid_path).exists():
                print("❌ No valid video loaded")
                return
            
            vid_path = Path(self.vid_path)
            out_path = vid_path.with_name(vid_path.stem + '_adjusted' + vid_path.suffix)
            
            cap = cv2.VideoCapture(str(vid_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            
            print(f"   Source: {vid_path.name}")
            print(f"   Output: {out_path.name}")
            print(f"   Brightness: {self.brightness_slider.value:.1f}x, Contrast: {self.contrast_slider.value:.1f}x")
            print(f"   Frames: {total}")
            
            for i in range(total):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(self._apply_adjustments(frame))
            
            cap.release()
            out.release()
            print(f"✅ Saved: {out_path}")
    
    def on_refresh_display(self, btn):
        """Manually refresh the frame display"""
        if self.video_loaded:
            self._update_frame_display()
            with self.output:
                print("🔄 Display refreshed")
    
    def on_clear_rois(self, btn):
        """Clear all ROIs"""
        self.rois = []
        self._update_roi_table()
        if self.video_loaded:
            self._draw_rois()
            self._update_title()
            self.fig.canvas.draw_idle()
        self._update_roi_status()
        self._update_process_button_state()
        
        with self.output:
            print("🗑️ All ROIs cleared")
    
    def on_update_display(self, btn):
        """Manually update frame display with current ROI coordinates"""
        # Read coordinates from table widgets
        for idx in self.roi_widgets:
            widgets_dict = self.roi_widgets[idx]
            self.rois[idx] = (
                widgets_dict['x'].value,
                widgets_dict['y'].value,
                widgets_dict['w'].value,
                widgets_dict['h'].value
            )
        
        if self.video_loaded:
            self._draw_rois()
            self.fig.canvas.draw_idle()
        
        with self.output:
            print("🔄 Display updated with edited coordinates")
    
    def on_save_config(self, btn):
        """Save configuration to session"""
        with self.output:
            print("\n" + "="*50)
            print("💾 Saving configuration to session...")
            
            # Build video processing config
            video_config = {
                'image_directory': self.img_dir_input.value,
                'sync_channel': {'channel': [self.sync_channel_input.value]},
                'reference_probe_index': self.ref_probe_input.value,
                'rois': [list(roi) for roi in self.rois],  # Convert tuples to lists for JSON
            }
            
            if self.vid_path:
                video_config['video_path'] = self.vid_path
            if self.tbl_path:
                video_config['timing_table_path'] = self.tbl_path
            
            # Update session config
            if 'parameters' not in self.session.config:
                self.session.config['parameters'] = {}
            
            self.session.config['parameters']['video_processing'] = video_config
            
            # Save to file
            with open(self.session.config_path, 'w') as f:
                json.dump(self.session.config, f, indent=2)
            
            print(f"✅ Configuration saved to: {self.session.config_path}")
            print(f"   Image directory: {video_config['image_directory']}")
            print(f"   Sync channel: {video_config['sync_channel']}")
            print(f"   Reference probe: {video_config['reference_probe_index']}")
            print(f"   ROIs: {len(self.rois)} defined")
    
    def on_process_align(self, btn):
        """Process and align video"""
        with self.output:
            if len(self.rois) != self.REQUIRED_ROIS:
                print(f"❌ Need exactly {self.REQUIRED_ROIS} ROIs, currently have {len(self.rois)}")
                return
            
            if not self.vid_path:
                print("❌ No video path set. Load a video first.")
                return
            
            if not self.tbl_path:
                print("❌ No timing table path set.")
                print(f"💡 Expected CSV file alongside video: {Path(self.vid_path).with_suffix('.csv')}")
                print("💡 If missing, convert images first to generate the timing table.")
                return
            
            if not Path(self.tbl_path).exists():
                print(f"❌ Timing table not found: {self.tbl_path}")
                print("💡 Convert images first to generate the timing table.")
                return
            
            print("\n" + "="*50)
            print("🎯 Processing and aligning video...")
            print("=" * 50)
            
            try:
                # Get dclut path for reference probe using Session API
                ref_probe = self.ref_probe_input.value
                
                # Find AP files and get dclut for the specified probe
                ap_files = sorted(self.session.find_files('ap_pattern'))
                
                if not ap_files:
                    print("❌ No AP files found in session")
                    return
                
                if ref_probe >= len(ap_files):
                    print(f"❌ Reference probe {ref_probe} not found. Only {len(ap_files)} probe(s) available.")
                    return
                
                ap_file = ap_files[ref_probe]
                dcl_path = str(self.session.get_dclut_path(str(ap_file)))
                
                sync_channel = {'channel': [self.sync_channel_input.value]}
                
                print(f"📹 Video: {Path(self.vid_path).name}")
                print(f"📋 Table: {Path(self.tbl_path).name}")
                print(f"🔗 Reference dclut: {Path(dcl_path).name}")
                print(f"📡 Sync channel: {sync_channel}")
                print(f"🎯 ROIs: {len(self.rois)}")
                
                from mimo_pack.preprocess.video import process_and_align_video
                
                aligned_tbl_path = process_and_align_video(
                    self.vid_path, self.tbl_path, self.rois, dcl_path, sync_channel
                )
                
                print("✅ Video alignment complete!")
                print(f"📋 Aligned table: {aligned_tbl_path}")
                
                # Save config after successful processing
                self.on_save_config(None)
                
                # Display QC figure
                self._display_qc_figure()
                
            except Exception as e:
                print(f"❌ Processing failed: {e}")
                import traceback
                traceback.print_exc()
                print("💡 Check your session files and ROI selection")
    
    def _display_qc_figure(self):
        """Display QC figure showing final frame with numbered ROIs"""
        with self.qc_output:
            clear_output(wait=True)
            
            if self.current_frame is None or not self.rois:
                return
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(self.current_frame)
            
            # Draw numbered ROIs matching QC pattern
            for i, (x, y, w, h) in enumerate(self.rois):
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)
                ax.text(
                    x, y - 10, f'ROI {i}',
                    color='red', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2)
                )
            
            ax.set_title(f'Video Alignment QC - {self.REQUIRED_ROIS} Barcode ROIs Selected')
            ax.axis('off')
            
            # Save to QC directory
            qc_dir = self.session.dir / 'qc_reports'
            qc_dir.mkdir(exist_ok=True)
            qc_path = qc_dir / 'video_alignment_rois.pdf'
            fig.savefig(qc_path, bbox_inches='tight')
            
            plt.tight_layout()
            plt.show()
            
            print(f"📊 QC figure saved: {qc_path}")
    
    def display(self):
        """Display the widget"""
        display(self.container)
    
    def __del__(self):
        """Cleanup on widget destruction"""
        if self.cap is not None:
            self.cap.release()
        if self.fig is not None:
            plt.close(self.fig)
