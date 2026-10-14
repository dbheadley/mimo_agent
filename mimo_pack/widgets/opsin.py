"""Opsin classification widget for in vitro LED-response analysis.

Provides an interactive ipywidgets interface that wraps the analysis
pipeline from ``format_blue_tonic_led`` and ``fit_opsin_response``.
Users select a cells DataFrame, configure LED detection parameters,
run the fitting loop, inspect per-cell fits, and classify opsin-positive
neurons using adjustable thresholds — all without writing loop code.

Requires ``ipydatagrid`` for the interactive parameter table.
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mimo_pack.fileio.abf import load_abf
from mimo_pack.preprocess.invitro import format_blue_tonic_led
from mimo_pack.analysis.invitro_ephys_props import fit_opsin_response
from mimo_pack.consts import DIM_CHAN, DIM_EPOCH

# Fit-parameter columns used for classification and scatter plots
_PARAM_COLS = ['peak_amp', 'A_sust', 'inactivation_ratio', 'tau_inact_ms',
               'tau_on_ms', 'tau_off_ms', 'peak_time_ms']

_PARAM_LABELS = {
    'peak_amp':           'Peak amplitude',
    'A_sust':             'Sustained response',
    'inactivation_ratio': 'Inactivation ratio',
    'tau_inact_ms':       'Inactivation τ (ms)',
    'tau_on_ms':          'Activation τ (ms)',
    'tau_off_ms':         'Deactivation τ (ms)',
    'peak_time_ms':       'Peak time (ms)',
}

# Default threshold directions (True = >=, False = <=)
_DEFAULT_DIRECTIONS = {
    'peak_amp':           True,   # >=
    'A_sust':             True,   # >=
    'inactivation_ratio': False,  # <=
    'tau_inact_ms':       True,   # >=
    'tau_on_ms':          True,   # >=
    'tau_off_ms':         True,   # >=
    'peak_time_ms':       True,   # >=
}

# Default threshold values (empty string = not used)
_DEFAULT_THRESHOLDS = {
    'peak_amp':           '5.0',
    'A_sust':             '20.0',
    'inactivation_ratio': '0.95',
    'tau_inact_ms':       '',
    'tau_on_ms':          '',
    'tau_off_ms':         '',
    'peak_time_ms':       '',
}


class OpsinClassificationWidget:
    """Interactive widget for opsin-positive cell classification.

    Wraps the LED response fitting and classification pipeline into a
    single ipywidgets interface with:

    - Configurable LED detection parameters
    - Protocol-name dropdown populated from the cells DataFrame
    - ``ipydatagrid`` parameter table with row selection
    - Automatic reclassification when thresholds change
    - Scatter plot of two selectable parameters with decision boundaries
    - Per-cell fit overlay plot (selected via the table)

    Parameters
    ----------
    None

    Examples
    --------
    >>> from mimo_pack.widgets.opsin import OpsinClassificationWidget
    >>> w = OpsinClassificationWidget()
    >>> w.set_dataframes({'cells': cells})
    >>> w.display()
    """

    def __init__(self):
        """Initialize the opsin classification widget.

        Creates an empty widget ready to receive a cells DataFrame
        via ``set_dataframes``.  No data is loaded until the user
        clicks *Run Analysis*.
        """
        self.cells_df = None
        self.led_params_df = None
        self.led_fits = {}
        self.rec_fmts = {}
        self._df_dict = {}
        self._selected_cell_idx = None
        self._suppress_classify = False

        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_dataframes(self, df_dict: dict):
        """Populate the DataFrame dropdown.

        Parameters
        ----------
        df_dict : dict
            Mapping of ``{'label': pd.DataFrame, ...}``.  Each key becomes
            an option in the dropdown; the first entry is selected by
            default.
        """
        self._df_dict = dict(df_dict)
        self._df_dropdown.options = list(df_dict.keys())
        if df_dict:
            self._df_dropdown.value = list(df_dict.keys())[0]
            self._on_df_selected(None)

    def display(self):
        """Show the widget."""
        display(self.widget)

    def get_results(self):
        """Return cell metadata merged with opsin classification results.

        Returns
        -------
        pd.DataFrame or None
            Copy of the source DataFrame with two added columns:

            * ``'opsin_fit'`` — dict of all LED-response fit parameters per
              cell (``peak_amp``, ``A_sust``, ``inactivation_ratio``,
              ``tau_inact_ms``)
            * ``'opsin_class'`` — classification result string
              (``'opsin+'``, ``'opsin-'``, or ``'no_recording'``)

            Returns ``None`` if analysis has not been run.
        """
        if self.cells_df is None or self.led_params_df is None:
            return None
        skip_cols = {'opsin_positive', 'opsin_class'}
        param_cols = [c for c in self.led_params_df.columns if c not in skip_cols]
        result = self.cells_df.copy()
        result['opsin_fit'] = [
            self.led_params_df.loc[idx, param_cols].to_dict()
            if idx in self.led_params_df.index else None
            for idx in result.index
        ]
        opsin_class = self.led_params_df['opsin_class'] \
            if 'opsin_class' in self.led_params_df.columns \
            else self.led_params_df['opsin_positive'].map({True: 'opsin+', False: 'opsin-'})
        result['opsin_class'] = opsin_class.reindex(result.index)
        result['opsin_class'] = result['opsin_class'].fillna('no_recording')
        return result

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Construct all ipywidget controls and assemble the layout."""
        # ---- Step 1: DataFrame & protocol selection --------------------
        step1_label = widgets.HTML("<h4>📥 Step 1: Select Data</h4>")

        self._df_dropdown = widgets.Dropdown(
            options=[],
            description='DataFrame:',
            style={'description_width': '90px'},
            layout=widgets.Layout(width='300px'),
        )
        self._df_dropdown.observe(self._on_df_selected, names='value')

        self._protocol_dropdown = widgets.Dropdown(
            options=[],
            description='Protocol:',
            style={'description_width': '90px'},
            layout=widgets.Layout(width='400px'),
        )
        self._rec_index_input = widgets.IntText(
            value=0,
            description='Rec #:',
            style={'description_width': '50px'},
            layout=widgets.Layout(width='120px'),
        )

        step1_box = widgets.VBox([
            step1_label,
            widgets.HBox([self._df_dropdown, self._protocol_dropdown,
                          self._rec_index_input]),
        ])

        # ---- Step 2: LED detection parameters --------------------------
        step2_label = widgets.HTML("<h4>🔧 Step 2: LED Detection Parameters</h4>")

        self._led_channel_input = widgets.Text(
            value='LEDPow, BlueLED, IN 5',
            description='LED channels:',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='300px'),
        )
        self._threshold_input = widgets.FloatText(
            value=0.1, description='Threshold (V):',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='200px'),
        )
        self._min_dur_input = widgets.FloatText(
            value=0.01, description='Min dur (s):',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='200px'),
        )
        self._duration_input = widgets.FloatText(
            value=0.5, description='Fallback dur (s):',
            style={'description_width': '110px'},
            layout=widgets.Layout(width='210px'),
        )
        self._onset_input = widgets.FloatText(
            value=0.235, description='Fallback onset (s):',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='220px'),
        )
        self._a_sust_fixed_check = widgets.Checkbox(
            value=True, description='A_sust fixed',
            style={'description_width': '80px'},
        )

        self._run_btn = widgets.Button(
            description='▶ Run Analysis',
            button_style='primary',
            layout=widgets.Layout(width='160px'),
        )
        self._run_btn.on_click(self._on_run_analysis)

        step2_row1 = widgets.HBox([
            self._led_channel_input,
            self._threshold_input,
            self._min_dur_input,
        ])
        step2_row2 = widgets.HBox([
            self._duration_input,
            self._onset_input,
            self._a_sust_fixed_check,
        ])

        step2_box = widgets.VBox([
            step2_label, step2_row1, step2_row2,
            self._run_btn,
        ])

        # ---- Step 3: Classification thresholds -------------------------
        step3_label = widgets.HTML("<h4>🎯 Step 3: Classification Thresholds</h4>")
        step3_help = widgets.HTML(
            "<small>Leave a field blank to ignore that parameter. "
            "Check the box for ≥ (greater-than-or-equal); uncheck for ≤ (less-than-or-equal).</small>"
        )

        self._thresh_widgets = {}  # param -> (Text, Checkbox)
        thresh_rows = []
        for col in _PARAM_COLS:
            txt = widgets.Text(
                value=_DEFAULT_THRESHOLDS[col],
                layout=widgets.Layout(width='80px'),
            )
            chk = widgets.Checkbox(
                value=_DEFAULT_DIRECTIONS[col],
                description='≥',
                indent=False,
                layout=widgets.Layout(width='40px'),
            )
            lbl = widgets.HTML(
                f"<span style='width:170px; display:inline-block'>"
                f"{_PARAM_LABELS[col]}</span>"
            )
            # Auto-classify on change
            txt.observe(self._on_threshold_changed, names='value')
            chk.observe(self._on_threshold_changed, names='value')
            self._thresh_widgets[col] = (txt, chk)
            thresh_rows.append(widgets.HBox([lbl, txt, chk]))

        step3_box = widgets.VBox([step3_label, step3_help] + thresh_rows)

        # ---- Results panels --------------------------------------------
        # Table spans full width
        self._table_placeholder = widgets.HTML(
            "<i>Run analysis to populate the table.</i>"
        )
        self._table_container = widgets.VBox(
            [self._table_placeholder],
            layout=widgets.Layout(
                width='100%', min_height='300px',
                overflow_y='auto',
            ),
        )

        # Plots in a row below the table
        self._scatter_x_dropdown = widgets.Dropdown(
            options=_PARAM_COLS, value='peak_amp',
            description='X:', style={'description_width': '20px'},
            layout=widgets.Layout(width='200px'),
        )
        self._scatter_y_dropdown = widgets.Dropdown(
            options=_PARAM_COLS, value='A_sust',
            description='Y:', style={'description_width': '20px'},
            layout=widgets.Layout(width='200px'),
        )
        self._scatter_x_dropdown.observe(self._on_scatter_param_changed, names='value')
        self._scatter_y_dropdown.observe(self._on_scatter_param_changed, names='value')

        self._scatter_output = widgets.Output(
            layout=widgets.Layout(height='320px'),
        )
        self._fit_output = widgets.Output(
            layout=widgets.Layout(height='320px'),
        )

        scatter_controls = widgets.HBox([
            self._scatter_x_dropdown,
            self._scatter_y_dropdown,
        ])

        scatter_panel = widgets.VBox([
            widgets.HTML("<b>Scatter Plot</b>"),
            scatter_controls,
            self._scatter_output,
        ], layout=widgets.Layout(width='50%'))

        fit_panel = widgets.VBox([
            widgets.HTML("<b>Cell Fit</b>"),
            self._fit_output,
        ], layout=widgets.Layout(width='50%'))

        plots_row = widgets.HBox([
            scatter_panel,
            fit_panel,
        ])

        results_panel = widgets.VBox([
            self._table_container,
            plots_row,
        ])

        # ---- Status output & clear ------------------------------------
        self._clear_btn = widgets.Button(
            description='🧹 Clear Output',
            button_style='',
            layout=widgets.Layout(width='140px'),
        )
        self._clear_btn.on_click(self._on_clear_output)
        self._done_btn = widgets.Button(
            description='✅ Done',
            button_style='success',
            layout=widgets.Layout(width='120px'),
        )
        self._done_btn.on_click(self._on_done)
        self.output = widgets.Output()

        # ---- Assemble --------------------------------------------------
        self.widget = widgets.VBox([
            widgets.HTML("<h3>🔬 Opsin Classification</h3>"),
            step1_box,
            step2_box,
            step3_box,
            results_panel,
            widgets.HBox([self._clear_btn, self._done_btn]),
            self.output,
        ])

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_df_selected(self, change):
        """When a DataFrame is chosen, populate the protocol dropdown."""
        name = self._df_dropdown.value
        if name is None or name not in self._df_dict:
            return
        df = self._df_dict[name]
        # Collect unique protocol names across all cells
        protocols = set()
        if 'protocols' in df.columns:
            for proto_dict in df['protocols']:
                if isinstance(proto_dict, dict):
                    protocols.update(proto_dict.keys())
        proto_list = sorted(protocols)
        self._protocol_dropdown.options = proto_list
        if proto_list:
            self._protocol_dropdown.value = proto_list[0]

    def _on_clear_output(self, btn):
        """Clear the status output area."""
        with self.output:
            clear_output()

    def _on_done(self, btn):
        """Write opsin classification results back to the source DataFrame via get_results()."""
        with self.output:
            clear_output(wait=True)
            results = self.get_results()
            if results is None:
                print("❌ No results yet — run analysis first.")
                return
            name = self._df_dropdown.value
            if name is not None and name in self._df_dict:
                src_df = self._df_dict[name]
                src_df['opsin_fit'] = results.reindex(src_df.index)['opsin_fit']
                src_df['opsin_class'] = results.reindex(src_df.index)['opsin_class']
            n_pos = int(self.led_params_df['opsin_positive'].sum()) \
                if 'opsin_positive' in self.led_params_df.columns else '?'
            print(f"✅ Done — results written back to '{name}'.")
            print(f"   {n_pos} / {len(self.led_params_df)} cells opsin-positive.")

    # ---- Run analysis --------------------------------------------------

    def _on_run_analysis(self, btn):
        """Load, format, and fit LED responses for every cell."""
        with self.output:
            clear_output(wait=True)
            name = self._df_dropdown.value
            if name is None or name not in self._df_dict:
                print("❌ Select a DataFrame first.")
                return

            protocol = self._protocol_dropdown.value
            if not protocol:
                print("❌ Select a protocol first.")
                return

            self.cells_df = self._df_dict[name].copy()

            # Parse LED channel list
            raw_ch = self._led_channel_input.value
            led_channels = [c.strip() for c in raw_ch.split(',') if c.strip()]
            if len(led_channels) == 1:
                led_channels = led_channels[0]

            threshold = self._threshold_input.value
            min_duration = self._min_dur_input.value
            duration = self._duration_input.value
            onset = self._onset_input.value
            a_sust_fixed = self._a_sust_fixed_check.value

            led_results = []
            self.led_fits = {}
            self.rec_fmts = {}

            total = len(self.cells_df)
            n_no_protocol = 0
            n_error = 0
            print(f"🔄 Processing {total} cells for protocol '{protocol}'...")

            for idx, row in self.cells_df.iterrows():
                proto = row.get('protocols', {})
                if not isinstance(proto, dict):
                    n_no_protocol += 1
                    continue
                files = proto.get(protocol, [])
                if not files:
                    n_no_protocol += 1
                    continue

                rec_idx = self._rec_index_input.value
                if rec_idx < 0 or rec_idx >= len(files):
                    n_error += 1
                    print(f"  ⚠️ Skipped cell {idx}: recording index "
                          f"{rec_idx} out of range (0–{len(files)-1})")
                    continue
                abf_file = files[rec_idx]['file']
                try:
                    rec = load_abf(abf_file)
                    rec_fmt = format_blue_tonic_led(
                        rec,
                        led_channel=led_channels,
                        threshold=threshold,
                        min_duration=min_duration,
                        duration=duration,
                        onset=onset,
                    )
                    fit = fit_opsin_response(rec_fmt, A_sust_fixed=a_sust_fixed)

                    params = dict(fit.attrs)
                    params['cell_idx'] = idx
                    led_results.append(params)
                    self.led_fits[idx] = fit
                    self.rec_fmts[idx] = rec_fmt
                except Exception as e:
                    n_error += 1
                    print(f"  ⚠️ Skipped cell {idx}: {e}")

            if not led_results:
                print(f"❌ No cells yielded valid fits.")
                print(f"   {n_no_protocol} cells had no '{protocol}' files, "
                      f"{n_error} failed during processing.")
                self.led_params_df = None
                return

            self.led_params_df = pd.DataFrame(led_results).set_index('cell_idx')
            print(f"✅ Fit {len(self.led_params_df)} / "
                  f"{total - n_no_protocol} cells with '{protocol}' recordings.")
            if n_no_protocol > 0:
                print(f"   ({n_no_protocol} cells had no '{protocol}' files)")
            if n_error > 0:
                print(f"   ({n_error} cells failed — see warnings above)")

        # Populate table and plots
        self._refresh_table()
        self._apply_classification()

    # ---- Classification ------------------------------------------------

    def _on_threshold_changed(self, change):
        """Re-classify automatically when any threshold widget changes."""
        if self._suppress_classify:
            return
        if self.led_params_df is not None and len(self.led_params_df) > 0:
            self._apply_classification()

    def _apply_classification(self):
        """Read threshold widgets, classify, and refresh views."""
        if self.led_params_df is None:
            return

        mask = pd.Series(True, index=self.led_params_df.index)
        for col in _PARAM_COLS:
            txt, chk = self._thresh_widgets[col]
            val_str = txt.value.strip()
            if not val_str:
                continue
            try:
                threshold_val = float(val_str)
            except ValueError:
                continue
            if chk.value:  # >= (greater-than-or-equal)
                mask = mask & (self.led_params_df[col] >= threshold_val)
            else:          # <= (less-than-or-equal)
                mask = mask & (self.led_params_df[col] <= threshold_val)

        self.led_params_df['opsin_positive'] = mask

        # Merge back into cells_df
        if self.cells_df is not None:
            self.cells_df['opsin_class'] = 'no_recording'
            self.cells_df.loc[self.led_params_df.index, 'opsin_class'] = (
                self.led_params_df['opsin_positive'].map(
                    {True: 'opsin+', False: 'opsin-'}
                )
            )

        self._refresh_table()
        self._refresh_scatter()

    # ---- Table ---------------------------------------------------------

    def _refresh_table(self):
        """Rebuild the ipydatagrid DataGrid from led_params_df."""
        if self.led_params_df is None:
            return

        try:
            from ipydatagrid import DataGrid
        except ImportError:
            # Fallback: plain HTML table with a dropdown for row selection
            self._refresh_table_fallback()
            return

        display_df = self.led_params_df.copy().reset_index()

        grid_kwargs = dict(
            selection_mode='row',
            editable=False,
            auto_fit_columns=True,
            layout=widgets.Layout(height='300px', width='100%'),
        )

        # Green row highlight for opsin+ cells using Expr + cell.metadata.data
        if 'opsin_positive' in display_df.columns:
            try:
                from ipydatagrid import TextRenderer, Expr
                grid_kwargs['default_renderer'] = TextRenderer(
                    background_color=Expr(
                        "'#c8f7c8' if cell.metadata.data['opsin_positive'] "
                        "else 'white'"
                    )
                )
            except Exception:
                pass

        grid = DataGrid(display_df, **grid_kwargs)
        grid.on_cell_click(self._on_grid_cell_click)
        self._grid = grid
        self._table_container.children = [grid]

    def _refresh_table_fallback(self):
        """Fallback when ipydatagrid is unavailable."""
        if self.led_params_df is None:
            return

        cell_indices = list(self.led_params_df.index)
        if not hasattr(self, '_fallback_dropdown'):
            self._fallback_dropdown = widgets.Dropdown(
                options=cell_indices,
                description='Cell:',
                style={'description_width': '40px'},
            )
            self._fallback_dropdown.observe(
                self._on_fallback_cell_selected, names='value'
            )
        else:
            self._fallback_dropdown.options = cell_indices

        # Build HTML table with green rows for opsin_positive == True
        df_html = self.led_params_df.copy()
        styler = df_html.style.format(
            {c: '{:.4g}' for c in df_html.select_dtypes(include='number').columns}
        )
        if 'opsin_positive' in df_html.columns:
            styler = styler.apply(
                lambda row: [
                    'background-color: #c8f7c8' if row.get('opsin_positive', False)
                    else '' for _ in row
                ],
                axis=1,
            )
        html = styler.to_html()
        scrollable = widgets.HTML(
            f"<div style='max-height:250px; overflow-y:auto'>{html}</div>"
        )
        self._table_container.children = [
            self._fallback_dropdown,
            scrollable,
        ]

    def _on_fallback_cell_selected(self, change):
        """Handle cell selection from the fallback dropdown when ipydatagrid is unavailable."""
        self._selected_cell_idx = change['new']
        self._refresh_fit_plot()
        self._refresh_scatter()

    def _on_grid_cell_click(self, event):
        """Handle row selection in the DataGrid.

        ipydatagrid passes a dict with keys:
        'region', 'column', 'column_index', 'row', 'primary_key_row',
        'cell_value'.  'row' is an integer row index into the displayed
        DataFrame.
        """
        row_idx = event.get('row', None)
        if row_idx is None or self.led_params_df is None:
            return
        try:
            cell_idx = self.led_params_df.index[row_idx]
        except (IndexError, KeyError):
            return
        self._selected_cell_idx = cell_idx
        self._refresh_fit_plot()
        self._refresh_scatter()

    # ---- Scatter plot --------------------------------------------------

    def _on_scatter_param_changed(self, change):
        """Redraw the scatter plot when the X or Y axis dropdown changes."""
        self._refresh_scatter()

    def _refresh_scatter(self):
        """Redraw the scatter plot of two selected parameters."""
        if self.led_params_df is None or len(self.led_params_df) == 0:
            return

        x_col = self._scatter_x_dropdown.value
        y_col = self._scatter_y_dropdown.value
        df = self.led_params_df

        with self._scatter_output:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(4, 3))

            if 'opsin_positive' in df.columns:
                pos = df[df['opsin_positive']]
                neg = df[~df['opsin_positive']]
                ax.scatter(
                    neg[x_col], neg[y_col],
                    c='gray', alpha=0.6, s=30, label='opsin−',
                )
                ax.scatter(
                    pos[x_col], pos[y_col],
                    c='dodgerblue', alpha=0.8, s=50, marker='D',
                    edgecolors='navy', linewidths=0.5, label='opsin+',
                )
                ax.legend(fontsize=8, loc='best')
            else:
                ax.scatter(df[x_col], df[y_col], c='steelblue', s=30)

            # Highlight the currently selected cell
            sel = self._selected_cell_idx
            if sel is not None and sel in df.index:
                ax.scatter(
                    [df.loc[sel, x_col]], [df.loc[sel, y_col]],
                    s=120, facecolors='none', edgecolors='red',
                    linewidths=2, zorder=5, label='selected',
                )

            # Decision boundary lines for visible parameters
            for col, axis in [(x_col, 'x'), (y_col, 'y')]:
                txt, chk = self._thresh_widgets[col]
                val_str = txt.value.strip()
                if not val_str:
                    continue
                try:
                    tv = float(val_str)
                except ValueError:
                    continue
                direction_label = '≥' if chk.value else '≤'
                if axis == 'x':
                    ax.axvline(tv, color='red', ls='--', lw=1,
                               label=f'{col} {direction_label} {tv}')
                else:
                    ax.axhline(tv, color='red', ls='--', lw=1,
                               label=f'{col} {direction_label} {tv}')

            ax.set_xlabel(_PARAM_LABELS.get(x_col, x_col), fontsize=9)
            ax.set_ylabel(_PARAM_LABELS.get(y_col, y_col), fontsize=9)
            ax.set_title('Parameter Scatter', fontsize=10)
            fig.tight_layout()
            plt.show()

    # ---- Fit plot ------------------------------------------------------

    def _refresh_fit_plot(self):
        """Plot the trial-averaged trace and fit for the selected cell."""
        idx = self._selected_cell_idx
        if idx is None or idx not in self.led_fits or idx not in self.rec_fmts:
            return

        fit = self.led_fits[idx]
        rec_fmt = self.rec_fmts[idx]

        # Trial-averaged trace — select channel matching the fit's response_units
        ch_units = rec_fmt['ch_units'].to_numpy().tolist()
        units = fit.attrs.get('response_units', 'mV')
        target_indices = [i for i, u in enumerate(ch_units) if u == units]
        if not target_indices:
            # fall back to mV for older fits
            target_indices = [i for i, u in enumerate(ch_units) if u == 'mV']
        target_ch = target_indices[0] if target_indices else 0
        v_mean = rec_fmt.sel({DIM_CHAN: target_ch}).mean(dim=DIM_EPOCH)

        with self._fit_output:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(4, 3))

            t_data = v_mean.coords['time'].to_numpy()
            ax.plot(t_data, v_mean.values, color='black', lw=0.8,
                    label='Trial avg')
            ax.plot(fit.coords['time'].to_numpy(), fit.values,
                    color='orange', lw=1.5, label='Fit')

            ax.set_xlabel('Time (s)', fontsize=9)

            peak = fit.attrs.get('peak_amp', 0)
            tau = fit.attrs.get('tau_inact_ms', 0)
            units = fit.attrs.get('response_units', 'mV')
            ax.set_ylabel(f'Response ({units})', fontsize=9)
            ax.set_title(
                f'Cell {idx}  |  peak={peak:.1f} {units}, τ={tau:.1f} ms',
                fontsize=9,
            )
            ax.legend(fontsize=7, loc='best')
            fig.tight_layout()
            plt.show()
