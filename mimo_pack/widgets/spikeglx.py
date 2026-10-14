"""SpikeGLX preprocessing widget for MIMO preprocessing."""

import ipywidgets as widgets
from IPython.display import display, clear_output
import json
from pathlib import Path


class SpikeGLXPreprocessor:
    """Interactive widget interface for SpikeGLX preprocessing"""
    
    def __init__(self, session):
        """Initialize the SpikeGLX preprocessing widget.

        Parameters
        ----------
        session : Session
            A loaded ``Session`` object whose directory contains raw
            SpikeGLX ``.bin`` files to preprocess.
        """
        self.session = session
        self.create_widgets()
    
    def create_widgets(self):
        """Build all ipywidget controls and lay out the UI."""
        # Title
        title = widgets.HTML("<h3>⚡ Interactive SpikeGLX Preprocessing</h3>")
        
        # Session info display
        session_info = widgets.HTML(
            f"<div style='background:#f0f8ff; padding:10px; border:1px solid #ccc;'>"
            f"<b>Session Directory:</b> {self.session.dir}<br>"
            f"<b>Config File:</b> {self.session.config_path}"
            f"</div>"
        )
        
        # Step status panel
        status_label = widgets.HTML("<h4>📋 Step Status:</h4>")
        self.status_area = widgets.HTML()
        self.refresh_status_btn = widgets.Button(
            description='🔄 Refresh Status',
            button_style='info',
            layout=widgets.Layout(width='160px')
        )
        self.refresh_status_btn.on_click(self.on_refresh_status)
        
        # Get current parameters from config
        current_sync_nidq = self.session.config['parameters'].get('sync_nidq', {'channel': [13]})
        current_sync_ap = self.session.config['parameters'].get('sync_ap', {'channel': [384]})
        
        # Sync AP channel configuration
        sync_ap_label = widgets.HTML("<h4>🔗 Sync AP Channel Configuration:</h4>")
        
        # Parse current sync AP channel (handle both list and single value)
        if isinstance(current_sync_ap.get('channel', [384]), list):
            current_ap_channel = current_sync_ap['channel'][0] if current_sync_ap['channel'] else 384
        else:
            current_ap_channel = current_sync_ap.get('channel', 384)
        
        self.sync_ap_input = widgets.IntText(
            value=current_ap_channel,
            description='AP Channel:',
            min=0,
            max=1000,
            layout=widgets.Layout(width='200px')
        )
        
        sync_ap_help = widgets.HTML(
            "<small>💡 AP channel used for synchronization (typically 384)</small>"
        )
        
        # Sync NIDQ channel configuration
        sync_nidq_label = widgets.HTML("<h4>🔗 Sync NIDQ Channel Configuration:</h4>")
        
        # Parse current sync NIDQ channel (handle both list and single value)
        if isinstance(current_sync_nidq.get('channel', [13]), list):
            current_nidq_channel = current_sync_nidq['channel'][0] if current_sync_nidq['channel'] else 5
        else:
            current_nidq_channel = current_sync_nidq.get('channel', 13)
        
        self.sync_nidq_input = widgets.IntText(
            value=current_nidq_channel,
            description='NIDQ Channel:',
            min=0,
            max=1000,
            layout=widgets.Layout(width='200px')
        )
        
        sync_nidq_help = widgets.HTML(
            "<small>💡 NIDQ channel used for synchronization (typically 13)</small>"
        )
        
        # Rerun options
        rerun_label = widgets.HTML("<h4>🔁 Rerun Options:</h4>")
        self.force_dclut_cb = widgets.Checkbox(
            value=False,
            description='Force rerun dclut + alignment',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        self.force_lfp_cb = widgets.Checkbox(
            value=False,
            description='Force rerun LFP generation',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='300px')
        )
        rerun_help = widgets.HTML(
            "<small>💡 When unchecked, existing output files are skipped automatically</small>"
        )
        
        # Preview configuration
        preview_label = widgets.HTML("<h4>👀 Configuration Preview:</h4>")
        self.preview_area = widgets.HTML()
        
        # Control buttons
        self.update_preview_btn = widgets.Button(
            description='🔍 Preview Config',
            button_style='info',
            layout=widgets.Layout(width='150px')
        )
        
        self.save_config_btn = widgets.Button(
            description='💾 Save Config',
            button_style='',
            layout=widgets.Layout(width='150px')
        )
        
        self.run_preprocessing_btn = widgets.Button(
            description='⚡ Run Preprocessing',
            button_style='success',
            layout=widgets.Layout(width='180px')
        )
        
        self.clear_btn = widgets.Button(
            description='🧹 Clear Output',
            button_style='warning',
            layout=widgets.Layout(width='150px')
        )
        
        # Output area
        self.output = widgets.Output()
        
        # Event handlers
        self.update_preview_btn.on_click(self.on_update_preview)
        self.save_config_btn.on_click(self.on_save_config)
        self.run_preprocessing_btn.on_click(self.on_run_preprocessing)
        self.clear_btn.on_click(self.on_clear_output)
        
        # Auto-update preview when parameters change
        self.sync_ap_input.observe(self.on_parameter_change, names='value')
        self.sync_nidq_input.observe(self.on_parameter_change, names='value')
        
        # Initial preview and status
        self.update_preview()
        self._render_status()
        
        # Layout
        self.container = widgets.VBox([
            title,
            session_info,
            status_label,
            self.status_area,
            self.refresh_status_btn,
            sync_ap_label,
            widgets.HBox([self.sync_ap_input, sync_ap_help]),
            sync_nidq_label,
            widgets.HBox([self.sync_nidq_input, sync_nidq_help]),
            rerun_label,
            self.force_dclut_cb,
            self.force_lfp_cb,
            rerun_help,
            preview_label,
            self.preview_area,
            widgets.HTML("<h4>🎮 Controls:</h4>"),
            widgets.HBox([self.update_preview_btn, self.save_config_btn, self.run_preprocessing_btn, self.clear_btn]),
            self.output
        ])
    
    def on_parameter_change(self, change):
        """Auto-update preview when parameters change"""
        self.update_preview()
    
    def update_preview(self):
        """Update the configuration preview"""
        sync_ap = {'channel': [self.sync_ap_input.value]}
        sync_nidq = {'channel': [self.sync_nidq_input.value]}
        
        preview_html = (
            "<div style='background:#f9f9f9; padding:10px; border:1px solid #ddd; font-family:monospace;'>"
            f"<b>sync_ap:</b> {sync_ap}<br>"
            f"<b>sync_nidq:</b> {sync_nidq}<br>"
            "</div>"
        )
        self.preview_area.value = preview_html
    
    def on_update_preview(self, btn):
        """Regenerate the configuration preview when the button is clicked."""
        self.update_preview()
    
    def on_save_config(self, btn):
        """Save configuration to session"""
        with self.output:
            print("\n" + "="*50)
            print("💾 Saving configuration...")
            
            # Update session config
            if 'parameters' not in self.session.config:
                self.session.config['parameters'] = {}
            
            self.session.config['parameters']['sync_ap'] = {'channel': [self.sync_ap_input.value]}
            self.session.config['parameters']['sync_nidq'] = {'channel': [self.sync_nidq_input.value]}
            
            # Save to file
            with open(self.session.config_path, 'w') as f:
                json.dump(self.session.config, f, indent=2)
            
            print(f"✅ Configuration saved to: {self.session.config_path}")
            print(f"   sync_ap: {self.session.config['parameters']['sync_ap']}")
            print(f"   sync_nidq: {self.session.config['parameters']['sync_nidq']}")
    
    def on_run_preprocessing(self, btn):
        """Run SpikeGLX preprocessing"""
        with self.output:
            print("\n" + "="*50)
            print("⚡ Starting SpikeGLX preprocessing...")
            print("="*50)
            
            # Get current parameters
            sync_ap = {'channel': [self.sync_ap_input.value]}
            sync_nidq = {'channel': [self.sync_nidq_input.value]}
            force_dclut = self.force_dclut_cb.value
            force_lfp = self.force_lfp_cb.value
            
            print("🔧 Preprocessing with parameters:")
            print(f"   sync_ap: {sync_ap}")
            print(f"   sync_nidq: {sync_nidq}")
            print(f"   force_dclut: {force_dclut}  |  force_lfp: {force_lfp}")
            print(f"🔧 Using session directory: {self.session.dir}")
            print("🚀 Running preprocessing (this may take a while)...")
            
            try:
                # Import here to avoid issues if module isn't available
                from mimo_pack.preprocess.session import preprocess_session
                
                # Run preprocessing with session parameters (modern approach)
                preprocess_session(
                    self.session.dir,
                    sync_ap=sync_ap,
                    sync_nidq=sync_nidq,
                    force_dclut=force_dclut,
                    force_lfp=force_lfp,
                )
                
                print("✅ SpikeGLX preprocessing complete!")
                print("🎯 Ready for next step (antidromic stimulation analysis)")
                
            except Exception as e:
                print(f"❌ Preprocessing failed: {e}")
                print("💡 Check your session directory and ensure all required files are present")
        
        # Refresh status panel after run (outside output context so the widget updates)
        self._render_status()
    
    def on_clear_output(self, btn):
        """Clear the output area"""
        with self.output:
            clear_output()
    
    def _detect_steps(self):
        """Detect which preprocessing output files already exist.

        Returns
        -------
        dict
            Keys: ``'ap'`` (list of per-probe status dicts), ``'nidq'``
            (list of per-NIDQ-file status dicts), ``'error'`` (str or None).
        """
        try:
            ap_files = self.session.find_files('ap_pattern')
            nidq_files = self.session.find_files('nidq_pattern')
        except Exception as e:
            return {'ap': [], 'nidq': [], 'error': str(e)}

        ap_status = []
        for f in ap_files:
            f_str = str(f)
            dclut = f_str.replace('.bin', '_dclut.json')
            lfp_bin = dclut.replace('ap_dclut.json', 'lfp.bin')
            lfp_dclut = lfp_bin.replace('.bin', '_dclut.json')
            ap_status.append({
                'bin': f_str,
                'dclut': dclut,
                'dclut_exists': Path(dclut).exists(),
                'lfp_bin': lfp_bin,
                'lfp_bin_exists': Path(lfp_bin).exists(),
                'lfp_dclut': lfp_dclut,
                'lfp_dclut_exists': Path(lfp_dclut).exists(),
            })

        nidq_status = []
        for f in nidq_files:
            f_str = str(f)
            dclut = f_str.replace('.bin', '_dclut.json')
            nidq_status.append({
                'bin': f_str,
                'dclut': dclut,
                'dclut_exists': Path(dclut).exists(),
            })

        return {'ap': ap_status, 'nidq': nidq_status, 'error': None}

    def _render_status(self):
        """Detect output files and update the status panel HTML."""
        status = self._detect_steps()

        if status['error']:
            self.status_area.value = (
                "<div style='background:#fff3cd; padding:8px; border:1px solid #ffc107; border-radius:4px;'>"
                f"\u26a0\ufe0f Could not detect files: {status['error']}</div>"
            )
            return

        if not status['ap'] and not status['nidq']:
            self.status_area.value = (
                "<div style='background:#fff3cd; padding:8px; border:1px solid #ffc107; border-radius:4px;'>"
                "\u26a0\ufe0f No AP or NIDQ files found in session directory.</div>"
            )
            return

        ok = "\u2705"
        missing = "\u274c"
        auto = "\u26a1 auto"  # binary exists, dclut will be regenerated without reprocessing

        rows = []
        for i, ap in enumerate(status['ap']):
            probe_name = Path(ap['bin']).name
            if len(probe_name) > 45:
                probe_name = '...' + probe_name[-42:]
            dclut_cell = f"{ok} ap_dclut" if ap['dclut_exists'] else f"{missing} ap_dclut"
            lfp_cell = f"{ok} lfp.bin" if ap['lfp_bin_exists'] else f"{missing} lfp.bin"
            if ap['lfp_dclut_exists']:
                lfp_dclut_cell = f"{ok} lfp_dclut"
            elif ap['lfp_bin_exists']:
                lfp_dclut_cell = f"{auto} lfp_dclut"
            else:
                lfp_dclut_cell = f"{missing} lfp_dclut"
            rows.append(
                f"<tr>"
                f"<td style='padding:3px 8px;'><b>Probe {i}</b><br>"
                f"<small style='color:#666;'>{probe_name}</small></td>"
                f"<td style='padding:3px 8px; text-align:center;'>{dclut_cell}</td>"
                f"<td style='padding:3px 8px; text-align:center;'>{lfp_cell}</td>"
                f"<td style='padding:3px 8px; text-align:center;'>{lfp_dclut_cell}</td>"
                f"</tr>"
            )

        for nidq in status['nidq']:
            nidq_name = Path(nidq['bin']).name
            if len(nidq_name) > 45:
                nidq_name = '...' + nidq_name[-42:]
            dclut_cell = f"{ok} nidq_dclut" if nidq['dclut_exists'] else f"{missing} nidq_dclut"
            rows.append(
                f"<tr>"
                f"<td style='padding:3px 8px;'><b>NIDQ</b><br>"
                f"<small style='color:#666;'>{nidq_name}</small></td>"
                f"<td style='padding:3px 8px; text-align:center;'>{dclut_cell}</td>"
                f"<td style='padding:3px 8px; text-align:center;' colspan='2'>"
                f"<i style='color:#aaa;'>n/a</i></td>"
                f"</tr>"
            )

        table_html = (
            "<div style='background:#f8f8f8; padding:10px; border:1px solid #ddd; border-radius:4px;'>"
            "<table style='border-collapse:collapse; width:100%;'>"
            "<thead><tr style='background:#e8e8e8;'>"
            "<th style='padding:4px 8px; text-align:left;'>File</th>"
            "<th style='padding:4px 8px;'>AP Dclut</th>"
            "<th style='padding:4px 8px;'>LFP Bin</th>"
            "<th style='padding:4px 8px;'>LFP Dclut</th>"
            "</tr></thead><tbody>"
            + "".join(rows)
            + "</tbody></table></div>"
        )
        self.status_area.value = table_html

    def on_refresh_status(self, btn):
        """Refresh the step status panel."""
        self._render_status()

    def display(self):
        """Render the widget in the current notebook cell."""
        display(self.container)
