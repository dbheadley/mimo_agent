"""Antidromic stimulation analysis widget for MIMO preprocessing."""

import ipywidgets as widgets
from IPython.display import display, clear_output
import json
import pandas as pd
import numpy as np
from pathlib import Path


class AntidromicAnalysisWidget:
    """Comprehensive widget for complete antidromic stimulation analysis workflow"""
    
    def __init__(self, session):
        """Initialize the antidromic stimulation analysis widget.

        Parameters
        ----------
        session : Session
            A loaded ``Session`` object whose directory contains NIDQ
            files with stimulation pulse data.
        """
        self.session = session
        self.nidq_dclut = None
        self.stim_chan = 4
        self.detection_start_time_s = 0.0
        self.voltage_threshold = None
        self.stim_channel_max_voltage = None
        self.stim_df_raw = None
        self.summary_df = None
        self.stim_df_final = None
        self.bad_groups = []
        self.stimulation_annotations = {}
        self.create_widgets()
    
    def create_widgets(self):
        """Build all ipywidget controls and lay out the UI."""
        # Title
        title = widgets.HTML("<h3>⚡ Comprehensive Antidromic Stimulation Analysis</h3>")
        
        # Session info display
        session_info = widgets.HTML(
            f"<div style='background:#f0f8ff; padding:10px; border:1px solid #ccc;'>"
            f"<b>Session Directory:</b> {self.session.dir}<br>"
            f"<b>Config File:</b> {self.session.config_path}"
            f"</div>"
        )
        
        # Step 0: Import from Session Section
        import_label = widgets.HTML("<h4>📥 Step 0: Import Settings</h4>")
        import_help = widgets.HTML(
            "<small>💡 Load previously saved stimulation settings from the session configuration</small>"
        )
        
        self.import_btn = widgets.Button(
            description='📥 Import from Session',
            button_style='warning',
            layout=widgets.Layout(width='180px')
        )
        
        # Step 1: Configuration Section
        config_label = widgets.HTML("<h4>🔧 Step 1: Configuration</h4>")
        
        # Stimulation channel input
        stim_params = self.session.config['parameters'].get('antidromic_stimulation', {})
        current_stim_chan = stim_params.get('stim_channel', 4)
        current_detection_start_time_s = stim_params.get('detection_start_time_s', 0.0)
        current_voltage_threshold = stim_params.get('voltage_threshold')
        
        self.stim_channel_input = widgets.IntText(
            value=current_stim_chan,
            description='Stim Channel:',
            min=0,
            max=1000,
            layout=widgets.Layout(width='200px')
        )
        
        stim_help = widgets.HTML(
            "<small>💡 NIDQ channel used for stimulation pulses (typically 4)</small>"
        )

        self.detection_start_time_input = widgets.FloatText(
            value=current_detection_start_time_s,
            description='Detection Start (s):',
            layout=widgets.Layout(width='260px')
        )

        detection_start_help = widgets.HTML(
            "<small>💡 Ignore stimulation pulses before this recording time</small>"
        )

        self.voltage_threshold_input = widgets.Text(
            value='' if current_voltage_threshold is None else str(current_voltage_threshold),
            description='Voltage Threshold:',
            placeholder='Auto: max / 2',
            layout=widgets.Layout(width='320px')
        )

        voltage_threshold_help = widgets.HTML(
            "<small>💡 Leave blank to use half the maximum stim-channel voltage</small>"
        )
        
        # Step 1 buttons
        self.extract_btn = widgets.Button(
            description='🔍 Extract Data',
            button_style='info',
            layout=widgets.Layout(width='150px')
        )
        
        # Step 2: Statistics Section
        stats_label = widgets.HTML("<h4>📊 Step 2: Statistics Review</h4>")
        self.stats_display = widgets.HTML()
        
        # Step 3: Configuration Section  
        selection_label = widgets.HTML("<h4>🎯 Step 3: Group Selection & Annotation</h4>")
        
        # Bad groups selection (initially empty)
        self.bad_groups_widget = widgets.SelectMultiple(
            options=[],
            value=(),
            description='Bad Groups:',
            layout=widgets.Layout(height='120px', width='600px')
        )
        
        bad_groups_help = widgets.HTML(
            "<small>💡 Select groups to exclude from analysis</small>"
        )
        
        # Annotation section
        annotation_label = widgets.HTML("<h5>📝 Annotate Valid Groups:</h5>")
        
        # Annotation container (will be populated after data extraction)
        self.annotation_container = widgets.VBox([])
        
        # Step 4: Action buttons
        action_label = widgets.HTML("<h4>🎮 Step 4: Actions</h4>")
        
        self.apply_btn = widgets.Button(
            description='🔧 Apply Configuration',
            button_style='primary',
            layout=widgets.Layout(width='180px')
        )
        
        self.save_btn = widgets.Button(
            description='💾 Save to Session',
            button_style='success',
            layout=widgets.Layout(width='150px')
        )
        
        self.generate_reports_btn = widgets.Button(
            description='📊 Generate QC Reports',
            button_style='',
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
        self.import_btn.on_click(self.on_import_from_session)
        self.extract_btn.on_click(self.on_extract_data)
        self.apply_btn.on_click(self.on_apply_configuration)
        self.save_btn.on_click(self.on_save_to_session)
        self.generate_reports_btn.on_click(self.on_generate_reports)
        self.clear_btn.on_click(self.on_clear_output)
        self.bad_groups_widget.observe(self.on_bad_groups_change, names='value')
        
        # Layout
        self.container = widgets.VBox([
            title,
            session_info,
            import_label,
            import_help,
            widgets.HBox([self.import_btn]),
            widgets.HTML("<hr>"),
            config_label,
            widgets.HBox([self.stim_channel_input, stim_help]),
            widgets.HBox([self.detection_start_time_input, detection_start_help]),
            widgets.HBox([self.voltage_threshold_input, voltage_threshold_help]),
            widgets.HBox([self.extract_btn]),
            stats_label,
            self.stats_display,
            selection_label,
            widgets.HTML("<h5>❌ Select Bad Groups to Exclude:</h5>"),
            self.bad_groups_widget,
            bad_groups_help,
            annotation_label,
            self.annotation_container,
            action_label,
            widgets.HBox([self.apply_btn, self.save_btn, self.generate_reports_btn, self.clear_btn]),
            self.output
        ])
    
    def on_clear_output(self, btn):
        """Clear the output area"""
        with self.output:
            clear_output()
    
    def on_import_from_session(self, btn):
        """Import stimulation settings from session configuration"""
        with self.output:
            print("\n" + "="*50)
            print("📥 Importing settings from session configuration...")
            print("=" * 50)
            
            try:
                # Get antidromic config from session
                antidromic_config = self.session.config.get('parameters', {}).get('antidromic_stimulation', {})
                
                if not antidromic_config:
                    print("⚠️ No saved stimulation configuration found in session")
                    print("💡 Run extraction and configuration first to save settings")
                    return
                
                # Import stimulation channel
                stim_chan = antidromic_config.get('stim_channel')
                if stim_chan is not None:
                    self.stim_channel_input.value = stim_chan
                    print(f"✅ Loaded stimulation channel: {stim_chan}")

                detection_start_time_s = antidromic_config.get('detection_start_time_s')
                if detection_start_time_s is not None:
                    self.detection_start_time_input.value = detection_start_time_s
                    print(f"✅ Loaded detection start time: {detection_start_time_s} s")

                if 'voltage_threshold' in antidromic_config:
                    voltage_threshold = antidromic_config.get('voltage_threshold')
                    self.voltage_threshold_input.value = '' if voltage_threshold is None else str(voltage_threshold)
                    threshold_label = 'auto (max / 2)' if voltage_threshold is None else f'{voltage_threshold} V'
                    print(f"✅ Loaded voltage threshold: {threshold_label}")
                
                # Import bad groups
                bad_groups_list = antidromic_config.get('bad_groups', [])
                print(f"✅ Loaded bad groups: {bad_groups_list if bad_groups_list else 'None'}")
                
                # Import annotations
                stimulation_notes = antidromic_config.get('stimulation_notes', [])
                annotations_dict = {}
                for note in stimulation_notes:
                    stim_group = note.get('stim_group')
                    region = note.get('region')
                    current = note.get('current_uA')
                    if stim_group is not None:
                        annotations_dict[stim_group] = {
                            'region': region,
                            'current_uA': current
                        }
                
                print(f"✅ Loaded annotations for {len(annotations_dict)} groups")
                
                # Store for later use after data extraction
                self.imported_bad_groups = bad_groups_list
                self.imported_annotations = annotations_dict
                
                print("\n🎯 Settings imported successfully!")
                print("💡 Next step: Click 'Extract Data' to load the NIDQ file")
                print("💡 Bad groups and annotations will be automatically applied after extraction")
                
            except Exception as e:
                print(f"❌ Failed to import settings: {e}")
                print("💡 Check that the session configuration is properly formatted")

    def _parse_voltage_threshold(self):
        threshold_text = str(self.voltage_threshold_input.value).strip()
        if threshold_text == '':
            return None
        threshold = float(threshold_text)
        if not np.isfinite(threshold):
            raise ValueError("Voltage threshold must be a finite number or blank for auto detection")
        return threshold
    
    def on_extract_data(self, btn):
        """Extract stimulation data and create statistics"""
        with self.output:
            print("\n" + "="*50)
            print("🔍 Step 1: Extracting stimulation data...")
            print("=" * 50)
            
            # Get NIDQ dclut file using session method
            nidq_files = self.session.find_files('nidq_pattern')
            
            if not nidq_files:
                print("❌ No NIDQ files found. Please check your session directory.")
                return
            
            # Use session method to get dclut path
            nidq_bin_path = str(nidq_files[0])
            try:
                self.nidq_dclut = self.session.get_dclut_path(nidq_bin_path)
                print(f"📁 Found NIDQ dclut file: {Path(self.nidq_dclut).name}")
            except FileNotFoundError as e:
                print(f"❌ NIDQ dclut file not found: {e}")
                print("💡 Make sure SpikeGLX preprocessing has been completed")
                return
            except Exception as e:
                print(f"❌ Error getting NIDQ dclut file: {e}")
                return
            
            # Get stimulation channel
            self.stim_chan = self.stim_channel_input.value
            print(f"🔧 Using stimulation channel: {self.stim_chan}")

            self.detection_start_time_s = float(self.detection_start_time_input.value)
            print(f"🔧 Detecting pulses from: {self.detection_start_time_s} s")

            try:
                self.voltage_threshold = self._parse_voltage_threshold()
            except ValueError as e:
                print(f"❌ {e}")
                return
            threshold_label = 'auto (max / 2)' if self.voltage_threshold is None else f'{self.voltage_threshold} V'
            print(f"🔧 Detection voltage threshold: {threshold_label}")
            
            try:
                # Extract stimulation data
                print("⚡ Extracting stimulation groups...")
                from mimo_pack.preprocess.antidromic import get_antistim_times, create_stimulation_summary
                print(self.nidq_dclut, self.stim_chan)
                self.stim_df_raw, stim_metadata = get_antistim_times(
                    self.nidq_dclut,
                    self.stim_chan,
                    start_time_s=self.detection_start_time_s,
                    voltage_threshold=self.voltage_threshold,
                    return_metadata=True,
                )
                self.stim_channel_max_voltage = stim_metadata.get('max_voltage')
                if self.stim_channel_max_voltage is not None and np.isfinite(self.stim_channel_max_voltage):
                    print(f"📈 Maximum voltage on stim channel: {self.stim_channel_max_voltage:.6g} V")
                else:
                    print("📈 Maximum voltage on stim channel: unavailable")
                
                print(f"📊 Found {len(self.stim_df_raw)} total stimulation groups")
                
                # Create statistics summary
                print("📊 Creating statistics summary...")
                self.summary_df = create_stimulation_summary(self.stim_df_raw)
                
                # Update UI with extracted data
                self.update_statistics_display()
                self.update_selection_widgets()
                
                # Apply imported settings if available
                if hasattr(self, 'imported_bad_groups') and hasattr(self, 'imported_annotations'):
                    print("\n📥 Applying imported settings...")
                    self.apply_imported_settings()
                
                print("✅ Data extraction complete!")
                print("🎯 Review the statistics table and configure groups below.")
                
            except Exception as e:
                print(f"❌ Data extraction failed: {e}")
                print("💡 Check your session directory and ensure all required files are present")
    
    def apply_imported_settings(self):
        """Apply previously imported settings to the UI"""
        try:
            # Apply bad groups
            if self.imported_bad_groups:
                self.bad_groups_widget.value = tuple(self.imported_bad_groups)
                print(f"  ✅ Applied bad groups: {self.imported_bad_groups}")
            
            # Apply annotations
            if self.imported_annotations:
                for group_idx, annotation in self.imported_annotations.items():
                    if hasattr(self, 'region_widgets') and group_idx in self.region_widgets:
                        self.region_widgets[group_idx].value = annotation.get('region', '')
                        self.current_widgets[group_idx].value = annotation.get('current_uA', 500)
                print(f"  ✅ Applied annotations for {len(self.imported_annotations)} groups")
            
            print("📥 Imported settings applied successfully!")
            
        except Exception as e:
            print(f"⚠️ Could not fully apply imported settings: {e}")
    
    def update_statistics_display(self):
        """Update the statistics display with extracted data"""
        if self.summary_df is not None:
            stats_html = self.summary_df.round(3).to_html(index=False, table_id="stats_table")
            if self.stim_channel_max_voltage is not None and np.isfinite(self.stim_channel_max_voltage):
                max_voltage_html = f"<p><b>Stim Channel Max Voltage:</b> {self.stim_channel_max_voltage:.6g} V</p>"
            else:
                max_voltage_html = "<p><b>Stim Channel Max Voltage:</b> unavailable</p>"
            display_html = (
                f"<div style='height: 200px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px;'>"
                f"{stats_html}</div>"
                f"{max_voltage_html}"
                f"<p><b>Total Groups:</b> {len(self.summary_df)}</p>"
            )
            self.stats_display.value = display_html
    
    def update_selection_widgets(self):
        """Update selection widgets with extracted data"""
        if self.summary_df is not None:
            # Update bad groups selection
            options = [(f"Group {i}: {row['n_pulses']} pulses, {row['frequency_hz']:.1f} Hz", i) 
                      for i, row in self.summary_df.iterrows()]
            self.bad_groups_widget.options = options
            
            # Create annotation widgets
            self.region_widgets = {}
            self.current_widgets = {}
            annotation_items = []
            
            n_groups = len(self.summary_df)
            for i in range(n_groups):  # Create annotation widgets for all groups
                label = widgets.HTML(f"<b>Group {i}:</b>", layout=widgets.Layout(width='80px'))
                
                region_widget = widgets.Text(
                    placeholder='e.g., thalamus, cortex',
                    layout=widgets.Layout(width='200px')
                )
                
                current_widget = widgets.IntText(
                    value=500,
                    layout=widgets.Layout(width='100px')
                )
                
                self.region_widgets[i] = region_widget
                self.current_widgets[i] = current_widget
                
                row_box = widgets.HBox([
                    label,
                    widgets.HTML("Region:", layout=widgets.Layout(width='50px')),
                    region_widget,
                    widgets.HTML("µA:", layout=widgets.Layout(width='30px')),
                    current_widget
                ])
                annotation_items.append(row_box)
            
            self.annotation_container.children = annotation_items
    
    def on_bad_groups_change(self, change):
        """Update annotation widget states when bad groups change"""
        if hasattr(self, 'region_widgets'):
            bad_list = list(change['new'])
            for i in self.region_widgets:
                disabled = i in bad_list
                self.region_widgets[i].disabled = disabled
                self.current_widgets[i].disabled = disabled
                if disabled:
                    self.region_widgets[i].placeholder = 'Excluded'
                else:
                    self.region_widgets[i].placeholder = 'e.g., thalamus, cortex'
    
    def _apply_from_session_config(self):
        """Build stim_df_final directly from saved stimulation_notes in the session config.

        Returns
        -------
        bool
            ``True`` if the configuration was loaded successfully, ``False`` if no
            saved notes were found.
        """
        antidromic_config = self.session.config.get('parameters', {}).get('antidromic_stimulation', {})
        stimulation_notes = antidromic_config.get('stimulation_notes', [])

        if not stimulation_notes:
            return False

        print("\n" + "="*50)
        print("🔧 Step 3: Applying configuration from saved session data...")
        print("=" * 50)

        self.bad_groups = antidromic_config.get('bad_groups', [])
        self.stimulation_annotations = {}
        final_data = []

        for note in stimulation_notes:
            stim_group = note.get('stim_group')
            region = note.get('region')
            current = note.get('current_uA')
            stim_times = note.get('stim_times')
            if stim_times is not None:
                stim_times = np.array(stim_times)

            if stim_group is not None and region is not None:
                self.stimulation_annotations[stim_group] = {
                    'region': region,
                    'current_uA': current,
                }

            final_data.append({
                'stim_group': stim_group,
                'original_group': stim_group,
                'region': region,
                'current_uA': current,
                'stim_times': stim_times,
            })

        self.stim_df_final = pd.DataFrame(final_data)

        print(f"✅ Configuration applied from saved session data!")
        print(f"   Bad groups excluded: {self.bad_groups}")
        print(f"   Valid annotated groups: {len(self.stimulation_annotations)}")
        print(f"   Final groups for analysis: {len(self.stim_df_final)}")

        if len(self.stim_df_final) > 0:
            print(f"\n📊 Final configuration table:")
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                display(self.stim_df_final[['stim_group', 'original_group', 'region', 'current_uA']])

        print("\n🎯 Ready to generate QC reports!")
        return True

    def on_apply_configuration(self, btn):
        """Apply configuration and create final dataframe"""
        with self.output:
            if self.stim_df_raw is None:
                if self._apply_from_session_config():
                    return
                print("❌ No extracted data and no saved stimulation notes found.")
                print("💡 Click '🔍 Extract Data' first, or save a configuration via '💾 Save to Session'.")
                return

            print("\n" + "="*50)
            print("🔧 Step 3: Applying configuration...")
            print("=" * 50)
            
            # Get selections
            self.bad_groups = list(self.bad_groups_widget.value)
            self.stimulation_annotations = {}
            
            # Collect annotations for valid groups
            if hasattr(self, 'region_widgets'):
                for i in self.region_widgets:
                    if i not in self.bad_groups:
                        region = self.region_widgets[i].value.strip()
                        current = self.current_widgets[i].value
                        if region:
                            self.stimulation_annotations[i] = {
                                'region': region,
                                'current_uA': current
                            }
            
            # Create final dataframe - preserve original group numbers
            # Add original index as a column before filtering
            stim_df_with_orig = self.stim_df_raw.copy()
            stim_df_with_orig['original_group'] = stim_df_with_orig.index
            
            # Filter out bad groups
            stim_df_filtered = stim_df_with_orig.drop(self.bad_groups).reset_index(drop=True)
            
            final_data = []
            for new_idx, row in stim_df_filtered.iterrows():
                orig_group = row['original_group']
                entry = {
                    'stim_group': new_idx,
                    'original_group': orig_group,
                    'stim_times': row['stim_times']
                }
                
                if orig_group in self.stimulation_annotations:
                    entry.update(self.stimulation_annotations[orig_group])
                else:
                    entry.update({'region': None, 'current_uA': None})
                
                final_data.append(entry)
            
            self.stim_df_final = pd.DataFrame(final_data)
            
            print(f"✅ Configuration applied!")
            print(f"   Bad groups excluded: {len(self.bad_groups)}")
            print(f"   Valid annotated groups: {len(self.stimulation_annotations)}")
            print(f"   Final groups for analysis: {len(self.stim_df_final)}")
            
            # Show detailed breakdown of all groups
            if len(self.stim_df_raw) > 0:
                print(f"\n📋 Detailed group breakdown:")
                for i in range(len(self.stim_df_raw)):
                    if i in self.bad_groups:
                        print(f"   Group {i}: ❌ EXCLUDED")
                    elif i in self.stimulation_annotations:
                        ann = self.stimulation_annotations[i]
                        print(f"   Group {i}: ✅ {ann['region']} ({ann['current_uA']}µA)")
                    else:
                        print(f"   Group {i}: ⚠️  No annotation provided")
            
            if len(self.stim_df_final) > 0:
                print(f"\n📊 Final configuration table:")
                # Ensure all rows are displayed
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    display(self.stim_df_final[['stim_group', 'original_group', 'region', 'current_uA']])
            
            print("\n🎯 Ready to save to session or generate QC reports!")
    
    def on_save_to_session(self, btn):
        """Save configuration to session"""
        with self.output:
            if not self.stimulation_annotations and self.stim_df_raw is not None:
                print("⚠️ No annotations found. Applying current configuration first...")
                self.on_apply_configuration(btn)
                return
            elif self.stim_df_raw is None:
                print("❌ Please extract data first!")
                return
            
            print("\n" + "="*50)
            print("💾 Step 4: Saving to session...")
            print("=" * 50)
            
            # Update session config
            antidromic_config = {
                'stim_channel': self.stim_chan,
                'detection_start_time_s': self.detection_start_time_s,
                'voltage_threshold': self.voltage_threshold,
                'bad_groups': self.bad_groups,
                'stimulation_notes': []
            }
            
            for orig_group, annotation in self.stimulation_annotations.items():
                # Get stim_times from the original dataframe
                stim_times = self.stim_df_raw.loc[orig_group, 'stim_times']
                # Convert numpy array to list for JSON serialization
                stim_times_list = stim_times.tolist() if hasattr(stim_times, 'tolist') else list(stim_times)
                
                antidromic_config['stimulation_notes'].append({
                    'stim_group': orig_group,
                    'region': annotation['region'],
                    'current_uA': annotation['current_uA'],
                    'stim_times': stim_times_list
                })
            
            # Save to session config
            if 'parameters' not in self.session.config:
                self.session.config['parameters'] = {}
            
            self.session.config['parameters']['antidromic_stimulation'] = antidromic_config
            
            # Save to file
            with open(self.session.config_path, 'w') as f:
                json.dump(self.session.config, f, indent=2)
            
            print(f"✅ Configuration saved to: {self.session.config_path}")
            print(f"✅ Configuration contains {len(self.stimulation_annotations)} annotated groups")
            print("🎯 Configuration is now persistent across sessions!")
    
    def on_generate_reports(self, btn):
        """Generate QC reports for valid stimulation groups"""
        with self.output:
            # Check if stim_df_final exists and has data
            if self.stim_df_final is None or len(self.stim_df_final) == 0:
                print("❌ Please apply configuration first to generate reports!")
                print("💡 Steps: 1) Extract Data → 2) Select Groups → 3) Apply Configuration → 4) Generate Reports")
                return
            
            print("\n" + "="*50)
            print("📊 Step 5: Generating QC Reports...")
            print("=" * 50)
            
            try:
                # Import required modules
                from mimo_pack.preprocess.antidromic import get_antidromic_evoked_summary
                from mimo_pack.plot.antidromic import plot_antidromic_summary
                import matplotlib.pyplot as plt
                
                # Get session files (need to implement fallback)
                probes = []
                try:
                    sess_files = self.session.get_session_files()
                    imec_data = sess_files.get('imec', [])
                    
                    if isinstance(imec_data, dict):
                        # If it's a dict, convert to list of probe data
                        for probe_idx, probe_info in imec_data.items():
                            probes.append(probe_info)
                    elif isinstance(imec_data, list):
                        # If already a list, use as is
                        probes = imec_data
                    else:
                        print(f"⚠️ Unexpected session_files format: {type(imec_data)}")
                        
                except Exception as e:
                    print(f"⚠️ Could not get session files: {e}")
                    print("📍 Using alternative file discovery method...")
                
                # Fallback method if session files didn't work
                if not probes:
                    ap_files = list(self.session.find_files('ap_pattern'))
                    if not ap_files:
                        print("❌ No AP files found for QC report generation")
                        print("💡 Ensure SpikeGLX preprocessing has been completed")
                        return
                    
                    print(f"📍 Found {len(ap_files)} AP file(s)")
                    
                    # Create probe structure using session methods
                    for ap_file in ap_files:
                        probe_dir = str(Path(ap_file).parent)
                        try:
                            ap_dclut = self.session.get_dclut_path(str(ap_file))
                            lfp_file = str(ap_file).replace('ap.bin', 'lfp.bin')
                            lfp_dclut = self.session.get_dclut_path(lfp_file)
                            
                            probes.append({
                                'dir': probe_dir,
                                'ap_dclut': str(ap_dclut),
                                'lfp_dclut': str(lfp_dclut)
                            })
                            print(f"  ✅ Probe: {Path(probe_dir).name}")
                        except Exception as e:
                            print(f"  ⚠️ Skipping probe {Path(ap_file).name}: {e}")
                            continue
                
                if not probes:
                    print("❌ Could not find any probes with valid dclut files")
                    return
                
                print(f"📊 Processing {len(probes)} probe(s) and {len(self.stim_df_final)} valid group(s)...")
                
                total_reports = 0
                for i, probe in enumerate(probes):
                    # Handle different probe data structures
                    if isinstance(probe, dict):
                        probe_dir = probe.get('dir', '')
                        ap_dclut = probe.get('ap_dclut', '')
                        lfp_dclut = probe.get('lfp_dclut', '')
                    else:
                        print(f"  ⚠️ Unexpected probe structure, skipping")
                        continue
                    
                    if not probe_dir:
                        print(f"  ⚠️ Probe {i}: No directory found, skipping")
                        continue
                    
                    qc_dir = Path(probe_dir) / 'qc_reports'
                    qc_dir.mkdir(parents=True, exist_ok=True)
                    
                    print(f"\n📊 Probe {i}: Generating reports...")
                    probe_reports = 0
                    
                    for idx, row in self.stim_df_final.iterrows():
                        # Check for required annotation fields
                        try:
                            region = row['region'] if 'region' in row.index else None
                            current = row['current_uA'] if 'current_uA' in row.index else None
                            stim_times = row['stim_times'] if 'stim_times' in row.index else None
                        except KeyError:
                            print(f"  ⚠️ Group {idx}: Error accessing row data, skipping")
                            continue
                        
                        # Check for None and NaN values
                        if (region is None or (isinstance(region, float) and pd.isna(region)) or 
                            current is None or (isinstance(current, float) and pd.isna(current)) or 
                            stim_times is None):
                            print(f"  ⚠️ Group {idx}: Missing annotation (region/current/times), skipping")
                            continue
                        
                        try:
                            # Verify dclut files exist
                            if not ap_dclut or not Path(ap_dclut).exists():
                                print(f"  ⚠️ Group {idx}: AP dclut file missing or invalid")
                                continue
                            if not lfp_dclut or not Path(lfp_dclut).exists():
                                print(f"  ⚠️ Group {idx}: LFP dclut file missing or invalid")
                                continue
                            
                            print(f"  🔄 Group {idx} ({region}, {current}µA)...", end="", flush=True)
                            
                            try:
                                # Generate summary using the new function
                                print(" [loading]", end="", flush=True)
                                summary = get_antidromic_evoked_summary(ap_dclut, lfp_dclut, stim_times)
                                
                                print(" [plotting]", end="", flush=True)
                                fig = plot_antidromic_summary(summary, current, region)
                                
                                print(" [saving]", end="", flush=True)
                                output_path = qc_dir / f'antidromic_summary_stimgroup{idx}.pdf'
                                fig.savefig(output_path)
                                plt.close(fig)  # Close figure to free memory
                                
                                print(" ✅")
                                probe_reports += 1
                                total_reports += 1
                            
                            except Exception as analysis_error:
                                print(f"\n       ❌ Analysis Error: {type(analysis_error).__name__}")
                                print(f"          Message: {str(analysis_error)[:200]}")
                                raise  # Re-raise to outer exception handler
                            
                        except Exception as e:
                            error_msg = f"{type(e).__name__}: {str(e)}"
                            print(f"\n       Full Error: {error_msg[:200]}")
                            
                            # Detailed error logging
                            import traceback
                            import sys
                            exc_type, exc_value, exc_traceback = sys.exc_info()
                            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
                            print("       Traceback:")
                            for line in tb_lines[-5:]:  # Last 5 lines of traceback
                                print(f"       {line.rstrip()}")
                    
                    if probe_reports > 0:
                        print(f"  📁 {probe_reports} reports saved to: {qc_dir.name}/")
                
                print("\n" + "=" * 50)
                print(f"🎯 Generated {total_reports} total QC reports!")
                if total_reports == 0:
                    print("⚠️ No reports were generated. Check that all groups have region and current annotations.")
                else:
                    print("✅ QC reports ready for review!")
                
            except Exception as e:
                print(f"❌ QC report generation failed: {e}")
                print("💡 Ensure all required analysis modules are available")
                print("\nDebugging info:")
                print(f"  - stim_df_final exists: {self.stim_df_final is not None}")
                if self.stim_df_final is not None:
                    print(f"  - stim_df_final length: {len(self.stim_df_final)}")
                    print(f"  - stim_df_final columns: {list(self.stim_df_final.columns)}")
                import traceback
                traceback.print_exc()
    
    def display(self):
        """Render the widget in the current notebook cell."""
        display(self.container)
