"""Behavioral state analysis widget for MIMO preprocessing."""

import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

from mimo_pack.fileio.lfp import load_lfp_full
from mimo_pack.analysis.beh_state import states_lfp_only, plot_band_power_states
from mimo_pack.plot.map import wave_map


class BehavioralStateWidget:
    """Interactive widget for LFP-based behavioral state analysis."""
    
    def __init__(self, session):
        """Initialize the behavioral-state analysis widget.

        Parameters
        ----------
        session : Session
            A loaded ``Session`` object whose directory contains LFP
            recordings for state classification.
        """
        self.session = session
        self.lfp_sig = None
        self.lfp_snip = None
        self.states = None
        self.spec_z = None
        
        # Load existing config
        self.config = session.config.get('parameters', {}).get('behavioral_states', {})
        
        self._build_ui()
        self._load_config()
        
    def _build_ui(self):
        """Build the widget UI."""
        # --- Configuration Section ---
        self.probe_input = widgets.IntText(
            value=0, description='Probe Index:', style={'description_width': '100px'}
        )
        self.channel_input = widgets.IntText(
            value=140, description='LFP Channel:', style={'description_width': '100px'}
        )
        
        # --- Visualization Time Range ---
        self.start_time_input = widgets.FloatText(
            value=10.0, description='Start Time (s):', style={'description_width': '100px'}
        )
        self.duration_input = widgets.FloatText(
            value=0.5, description='Duration (s):', style={'description_width': '100px'}
        )
        
        # --- State Analysis Parameters ---
        # Each row is a frequency band: [Label, Low Hz, High Hz]
        param_header = widgets.HTML("<b>State Analysis Parameters:</b>")
        col_labels = widgets.HBox([
            widgets.HTML("<span style='width:80px; display:inline-block'></span>"),
            widgets.HTML("<span style='width:150px; display:inline-block; text-align:center'><b>Low (Hz)</b></span>"),
            widgets.HTML("<span style='width:150px; display:inline-block; text-align:center'><b>High (Hz)</b></span>")
        ])
        
        # Delta band row
        self.delta_low = widgets.FloatText(value=1.0, layout=widgets.Layout(width='120px'))
        self.delta_high = widgets.FloatText(value=4.0, layout=widgets.Layout(width='120px'))
        delta_row = widgets.HBox([
            widgets.HTML("<span style='width:80px; display:inline-block'><b>Delta:</b></span>"),
            self.delta_low, self.delta_high
        ])
        
        # Theta band row
        self.theta_low = widgets.FloatText(value=6.0, layout=widgets.Layout(width='120px'))
        self.theta_high = widgets.FloatText(value=9.0, layout=widgets.Layout(width='120px'))
        theta_row = widgets.HBox([
            widgets.HTML("<span style='width:80px; display:inline-block'><b>Theta:</b></span>"),
            self.theta_low, self.theta_high
        ])
        
        # High band row
        self.high_low = widgets.FloatText(value=20.0, layout=widgets.Layout(width='120px'))
        self.high_high = widgets.FloatText(value=80.0, layout=widgets.Layout(width='120px'))
        high_row = widgets.HBox([
            widgets.HTML("<span style='width:80px; display:inline-block'><b>High:</b></span>"),
            self.high_low, self.high_high
        ])
        
        # Noise band row
        self.noise_low = widgets.FloatText(value=150.0, layout=widgets.Layout(width='120px'))
        self.noise_high = widgets.FloatText(value=200.0, layout=widgets.Layout(width='120px'))
        noise_row = widgets.HBox([
            widgets.HTML("<span style='width:80px; display:inline-block'><b>Noise:</b></span>"),
            self.noise_low, self.noise_high
        ])
        
        self.exclude_rem_check = widgets.Checkbox(value=True, description='Exclude REM', 
                                                   style={'description_width': '80px'})
        
        # --- Buttons ---
        self.load_btn = widgets.Button(description='Load LFP', button_style='primary')
        self.visualize_btn = widgets.Button(description='Visualize Channels', button_style='info')
        self.analyze_btn = widgets.Button(description='Run Analysis', button_style='success')
        self.save_btn = widgets.Button(description='Save Config', button_style='warning')
        self.clear_btn = widgets.Button(description='🧹 Clear Output', button_style='')
        
        self.load_btn.on_click(self._on_load_lfp)
        self.visualize_btn.on_click(self._on_visualize)
        self.analyze_btn.on_click(self._on_analyze)
        self.save_btn.on_click(self._on_save_config)
        self.clear_btn.on_click(self._on_clear_output)
        
        # --- Output Area ---
        self.output = widgets.Output()
        
        # --- Layout ---
        config_box = widgets.VBox([
            widgets.HTML("<h4>📊 Behavioral State Analysis</h4>"),
            widgets.HBox([self.probe_input, self.channel_input]),
            widgets.HBox([self.start_time_input, self.duration_input]),
            param_header,
            col_labels,
            delta_row,
            theta_row,
            high_row,
            noise_row,
            self.exclude_rem_check,
            widgets.HBox([self.load_btn, self.visualize_btn, self.analyze_btn, self.save_btn, self.clear_btn]),
        ])
        
        self.widget = widgets.VBox([config_box, self.output])
        
    def _load_config(self):
        """Load configuration from session."""
        if self.config:
            self.probe_input.value = self.config.get('probe_index', 0)
            self.channel_input.value = self.config.get('lfp_channel', 140)
            
            vis_params = self.config.get('visualization', {})
            self.start_time_input.value = vis_params.get('start_time', 10.0)
            self.duration_input.value = vis_params.get('duration', 0.5)
            
            state_params = self.config.get('state_analysis', {})
            delta = state_params.get('delta', [1, 4])
            self.delta_low.value = delta[0]
            self.delta_high.value = delta[1]
            theta = state_params.get('theta', [6, 9])
            self.theta_low.value = theta[0]
            self.theta_high.value = theta[1]
            high = state_params.get('high', [20, 80])
            self.high_low.value = high[0]
            self.high_high.value = high[1]
            noise = state_params.get('noise', [150, 200])
            self.noise_low.value = noise[0]
            self.noise_high.value = noise[1]
            self.exclude_rem_check.value = state_params.get('exclude_rem', True)
            
            with self.output:
                print("✅ Loaded configuration from session")
    
    def _on_clear_output(self, btn):
        """Clear the output area."""
        with self.output:
            clear_output()
    
    def _on_load_lfp(self, btn):
        """Load LFP data."""
        with self.output:
            print("\n" + "="*50)
            probe_ind = self.probe_input.value
            print(f"🔄 Loading LFP for probe {probe_ind}...")
            
            # Find LFP files using config pattern and filter by probe index
            lfp_files = self.session.find_files('lfp_pattern')
            if not lfp_files:
                print("❌ No LFP files found!")
                return
            
            # Filter to the requested probe
            probe_files = [f for f in lfp_files if f'imec{probe_ind}' in str(f)]
            if not probe_files:
                print(f"❌ No LFP file found for probe {probe_ind}")
                print(f"   Available files: {[f.name for f in lfp_files]}")
                return
            
            lfp_bin_path = probe_files[0]
            lfp_path = self.session.get_dclut_path(lfp_bin_path)
            print(f"   Loading from: {lfp_path}")
            
            self.lfp_sig = load_lfp_full(lfp_path)
            print(f"✅ Loaded LFP: {self.lfp_sig.shape}, fs={self.lfp_sig.attrs.get('fs', 'N/A')} Hz")
            print(f"   Channels: {self.lfp_sig.channel.values[:5]}...{self.lfp_sig.channel.values[-5:]}")
            
    def _on_visualize(self, btn):
        """Visualize LFP channels for selection."""
        with self.output:
            print("\n" + "="*50)
            if self.lfp_sig is None:
                print("❌ Load LFP first!")
                return
            
            start_time = self.start_time_input.value
            duration = self.duration_input.value
            end_time = start_time + duration
            
            print(f"📊 Visualizing LFP snippet: {start_time}-{end_time}s")
            
            # Extract and median-subtract snippet
            self.lfp_snip = self.lfp_sig.sel(time=slice(start_time, end_time))
            self.lfp_snip = self.lfp_snip - self.lfp_snip.median(dim='time')
            
            # Plot using wave_map
            fig = wave_map(self.lfp_snip, label_coord='channel', x_scale=0.4)
            plt.title(f'LFP Channels (current selection: {self.channel_input.value})')
            plt.show()
            
    def _on_analyze(self, btn):
        """Run behavioral state analysis."""
        with self.output:
            print("\n" + "="*50)
            if self.lfp_sig is None:
                print("❌ Load LFP first!")
                return
            
            channel = self.channel_input.value
            probe_ind = self.probe_input.value
            
            # Get state analysis parameters
            delta = [self.delta_low.value, self.delta_high.value]
            theta = (self.theta_low.value, self.theta_high.value)
            high = (self.high_low.value, self.high_high.value)
            noise = (self.noise_low.value, self.noise_high.value)
            exclude_rem = self.exclude_rem_check.value
            
            print(f"🎯 Running behavioral state analysis on channel {channel}")
            print(f"   Delta: {delta}, Theta: {theta}, High: {high}, Noise: {noise}")
            print(f"   Exclude REM: {exclude_rem}")
            
            # Run state analysis
            self.states, self.spec_z = states_lfp_only(
                self.lfp_sig.sel(channel=channel),
                delta=delta, theta=theta, high=high, noise=noise,
                exclude_rem=exclude_rem
            )
            
            # Create QC directory
            qc_dir = self.session.dir / 'qc_reports'
            qc_dir.mkdir(exist_ok=True)
            
            # Generate QC plots
            fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
            plot_band_power_states(self.spec_z, self.states, axs=axs)
            axs[0].set_title(f'LFP Band Power and Behavioral States: probe {probe_ind}, channel {channel}')
            axs[1].set_xlabel('Time (s)')
            plt.tight_layout()
            
            # Save QC plot
            qc_plot_path = qc_dir / 'behavioral_states_lfp.pdf'
            fig.savefig(qc_plot_path)
            print(f"📊 Saved QC plot: {qc_plot_path}")
            plt.show()
            
            # Save states data
            states_path = self.session.dir / 'states.nc'
            self.states.to_netcdf(states_path)
            print(f"💾 Saved behavioral states: {states_path}")
            
            print("✅ Behavioral state analysis complete!")
            
    def _on_save_config(self, btn):
        """Save configuration to session."""
        with self.output:
            print("\n" + "="*50)
            
            # Build config
            config = {
                'probe_index': self.probe_input.value,
                'lfp_channel': self.channel_input.value,
                'visualization': {
                    'start_time': self.start_time_input.value,
                    'duration': self.duration_input.value
                },
                'state_analysis': {
                    'delta': [self.delta_low.value, self.delta_high.value],
                    'theta': [self.theta_low.value, self.theta_high.value],
                    'high': [self.high_low.value, self.high_high.value],
                    'noise': [self.noise_low.value, self.noise_high.value],
                    'exclude_rem': self.exclude_rem_check.value
                }
            }
            
            # Update session config
            if 'parameters' not in self.session.config:
                self.session.config['parameters'] = {}
            self.session.config['parameters']['behavioral_states'] = config
            
            # Save to disk
            import json
            with open(self.session.config_path, 'w') as f:
                json.dump(self.session.config, f, indent=4)
            
            print(f"✅ Saved configuration to {self.session.config_path}")
            print(f"   Probe: {config['probe_index']}, Channel: {config['lfp_channel']}")
            print(f"   State params: delta={config['state_analysis']['delta']}, theta={config['state_analysis']['theta']}")
    
    def display(self):
        """Display the widget."""
        display(self.widget)
