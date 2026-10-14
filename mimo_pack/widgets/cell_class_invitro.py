"""Cell classification widget for in vitro IV-curve analysis.

Provides an interactive ipywidgets interface that wraps the analysis
pipeline from ``format_ivic`` and ``extract_iv_properties``.  Users
select a cells DataFrame, choose the IV protocol name, run the
extraction loop, inspect per-cell traces by clicking table rows, and
classify neurons (FSI / RS / Unknown) using adjustable thresholds —
all without writing loop code.

Requires ``ipydatagrid`` for the interactive parameter table.
"""

import datetime as dt

import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from mimo_pack.fileio.abf import load_abf, load_abf_settings
from mimo_pack.preprocess.invitro import format_ivic
from mimo_pack.analysis.invitro_ephys_props import extract_iv_properties
from mimo_pack.plot.timeseries import stacked_lines
from mimo_pack.consts import DIM_TIME, DIM_CHAN, DIM_EPOCH

# -----------------------------------------------------------------------
# Module-level constants
# -----------------------------------------------------------------------

# Columns shown in the DataGrid (display names differ from dict keys)
_TABLE_COLS = [
    'ap_halfwidth_ms',
    'mem_t_ms',
    'max_ap_rate_hz',
    'mem_r_MOhm',
    'sag_ratio',
    'max_reb_aps',
    'cell_class',
]

# Human-readable labels
_PARAM_LABELS = {
    'ap_halfwidth_ms': 'AP half-width (ms)',
    'mem_t_ms':        'Membrane τ (ms)',
    'max_ap_rate_hz':  'Max firing rate (Hz)',
    'mem_r_MOhm':      'Input resistance (MΩ)',
    'sag_ratio':       'Sag ratio',
    'max_reb_aps':     'Max rebound APs',
    'cell_class':      'Class',
}

# The three parameters that vote on FSI vs RS
_CLASSIFY_PARAMS = ['ap_halfwidth_ms', 'mem_t_ms', 'max_ap_rate_hz']

# For each FSI param: True means ">= threshold" is the FSI criterion,
# False means "<= threshold" is the FSI criterion.
_FSI_DIRECTION_GTE = {
    'ap_halfwidth_ms': False,  # FSI: halfwidth ≤ threshold
    'mem_t_ms':        False,  # FSI: τ ≤ threshold
    'max_ap_rate_hz':  True,   # FSI: rate ≥ threshold
}

# Default FSI threshold values
_DEFAULT_THRESHOLDS = {
    'ap_halfwidth_ms': '0.4',
    'mem_t_ms':        '5.0',
    'max_ap_rate_hz':  '100.0',
}

# LTS criteria — all non-empty LTS criteria must be met.
# LTS is evaluated first and takes priority over FSI / RS.
_LTS_PARAMS = ['sag_ratio', 'max_reb_aps']

# True means ">= threshold" is the LTS criterion.
_LTS_DIRECTION_GTE = {
    'sag_ratio':   False,  # LTS: sag_ratio ≤ threshold (prominent sag)
    'max_reb_aps': True,   # LTS: max_reb_aps ≥ threshold (rebound firing)
}

# Default LTS threshold values (leave blank to ignore a criterion)
_DEFAULT_LTS_THRESHOLDS = {
    'sag_ratio':   '0.9',
    'max_reb_aps': '1',
}

# Row colours per class
_CLASS_COLOURS = {
    'FSI':     '#c8e6ff',   # blue
    'RS':      '#ffddb3',   # orange
    'LTS':     '#c8f7c8',   # green
    'Unknown': 'white',
}


class CellClassificationWidget:
    """Interactive widget for classifying neurons from IV-curve recordings.

    Wraps the IVIC loading, formatting, and property-extraction pipeline
    into a single ipywidgets interface with:

    - Protocol-name input populated from the cells DataFrame
    - ``ipydatagrid`` parameter table with row selection and colour coding
    - Automatic reclassification when thresholds change
    - Scatter plot of any two parameters with classification decision lines
    - Per-cell IV-curve traces shown for the selected row

    After calling ``_on_run_analysis`` the widget writes
    ``cells_df['class']`` back to the source DataFrame.

    Parameters
    ----------
    None

    Examples
    --------
    >>> from mimo_pack.widgets.cell_class_invitro import CellClassificationWidget
    >>> w = CellClassificationWidget()
    >>> w.set_dataframes({'cells': cells})
    >>> w.display()
    """

    def __init__(self):
        """Initialize the cell classification widget.

        Creates an empty widget ready to receive a cells DataFrame
        via ``set_dataframes``.  No data is loaded until the user
        clicks *Run Analysis*.
        """
        self.cells_df = None          # copy of source df, updated after run
        self.props_df = None          # extracted properties, indexed by cell_idx
        self.ivic_data = {}           # {cell_idx: formatted xr.DataArray}
        self._df_dict = {}
        self._selected_cell_idx = None
        self._suppress_classify = False
        self._grid = None             # ipydatagrid DataGrid reference
        self._scatter_artists = {}    # {artist: [cell_idx, ...]} for pick events

        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_dataframes(self, df_dict: dict):
        """Populate the DataFrame dropdown.

        Parameters
        ----------
        df_dict : dict
            Mapping of ``{'label': pd.DataFrame, ...}``.
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
        """Return cell metadata merged with classification results.

        Returns
        -------
        pd.DataFrame or None
            Copy of the source DataFrame with two added columns:

            * ``'ephys_props'`` — dict of all extracted IV properties per cell
              (``ap_halfwidth_ms``, ``mem_t_ms``, ``max_ap_rate_hz``,
              ``mem_r_MOhm``, ``sag_ratio``, ``max_reb_aps``)
            * ``'cell_class'`` — classification result string
              (``'FSI'``, ``'RS'``, ``'LTS'``, or ``'Unknown'``)

            Returns ``None`` if analysis has not been run.
        """
        if self.cells_df is None or self.props_df is None:
            return None
        prop_cols = [c for c in self.props_df.columns if c != 'cell_class']
        result = self.cells_df.copy()
        result['ephys_props'] = [
            self.props_df.loc[idx, prop_cols].to_dict()
            if idx in self.props_df.index else None
            for idx in result.index
        ]
        result['cell_class'] = self.props_df.reindex(result.index)['cell_class']
        return result

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Construct all ipywidget controls and assemble the layout."""
        # ---- Step 1: DataFrame, protocol, run -------------------------
        step1_label = widgets.HTML("<h4>📥 Step 1: Select Data</h4>")

        self._df_dropdown = widgets.Dropdown(
            options=[],
            description='DataFrame:',
            style={'description_width': '90px'},
            layout=widgets.Layout(width='280px'),
        )
        self._df_dropdown.observe(self._on_df_selected, names='value')

        self._protocol_input = widgets.Text(
            value='Left_IVIC',
            description='Protocol:',
            style={'description_width': '70px'},
            layout=widgets.Layout(width='260px'),
        )

        self._run_btn = widgets.Button(
            description='▶ Run Analysis',
            button_style='primary',
            layout=widgets.Layout(width='160px'),
        )
        self._run_btn.on_click(self._on_run_analysis)

        step1_box = widgets.VBox([
            step1_label,
            widgets.HBox([self._df_dropdown, self._protocol_input, self._run_btn]),
        ])

        # ---- Step 2: Classification thresholds ------------------------
        step2_label = widgets.HTML("<h4>🎯 Step 2: Classification Thresholds</h4>")
        step2_help = widgets.HTML(
            "<small>"
            "<b>FSI</b>: AP half-width ≤, membrane τ ≤, max firing rate ≥ — all three must agree "
            "(→ FSI); all three oppose (→ RS); otherwise Unknown.<br>"
            "<b>LTS</b>: sag ratio ≤ <i>and</i> max rebound APs ≥ — tested first and takes "
            "priority over FSI / RS. Leave a field blank to ignore that criterion.<br>"
            "Defaults based on Neske, Patrick &amp; Connors (2015)."
            "</small>"
        )

        self._thresh_widgets = {}   # param -> Text widget (FSI + LTS combined)

        # FSI rows
        fsi_header = widgets.HTML("<b>FSI / RS criteria</b>")
        fsi_rows = []
        for col in _CLASSIFY_PARAMS:
            gte = _FSI_DIRECTION_GTE[col]
            direction_sym = '≥' if gte else '≤'
            txt = widgets.Text(
                value=_DEFAULT_THRESHOLDS[col],
                layout=widgets.Layout(width='80px'),
            )
            lbl = widgets.HTML(
                f"<span style='width:220px; display:inline-block'>"
                f"FSI if {_PARAM_LABELS[col]} {direction_sym}</span>"
            )
            txt.observe(self._on_threshold_changed, names='value')
            self._thresh_widgets[col] = txt
            fsi_rows.append(widgets.HBox([lbl, txt]))

        # LTS rows
        lts_header = widgets.HTML("<b>LTS criteria</b> (checked first, takes priority)")
        lts_rows = []
        for col in _LTS_PARAMS:
            gte = _LTS_DIRECTION_GTE[col]
            direction_sym = '≥' if gte else '≤'
            txt = widgets.Text(
                value=_DEFAULT_LTS_THRESHOLDS[col],
                layout=widgets.Layout(width='80px'),
            )
            lbl = widgets.HTML(
                f"<span style='width:220px; display:inline-block'>"
                f"LTS if {_PARAM_LABELS[col]} {direction_sym}</span>"
            )
            txt.observe(self._on_threshold_changed, names='value')
            self._thresh_widgets[col] = txt
            lts_rows.append(widgets.HBox([lbl, txt]))

        step2_box = widgets.VBox(
            [step2_label, step2_help, fsi_header] + fsi_rows + [lts_header] + lts_rows
        )

        # ---- Results: table ------------------------------------------
        self._table_placeholder = widgets.HTML(
            "<i>Run analysis to populate the table.</i>"
        )
        self._table_container = widgets.VBox(
            [self._table_placeholder],
            layout=widgets.Layout(width='100%', min_height='300px', overflow_y='auto'),
        )

        # ---- Results: scatter plot ------------------------------------
        _scatter_opts = _TABLE_COLS[:-1]   # all but 'cell_class'
        self._scatter_x_dropdown = widgets.Dropdown(
            options=_scatter_opts, value='ap_halfwidth_ms',
            description='X:', style={'description_width': '20px'},
            layout=widgets.Layout(width='220px'),
        )
        self._scatter_y_dropdown = widgets.Dropdown(
            options=_scatter_opts, value='max_ap_rate_hz',
            description='Y:', style={'description_width': '20px'},
            layout=widgets.Layout(width='220px'),
        )
        self._scatter_x_dropdown.observe(self._on_scatter_param_changed, names='value')
        self._scatter_y_dropdown.observe(self._on_scatter_param_changed, names='value')

        self._scatter_output = widgets.Output(
            layout=widgets.Layout(height='340px'),
        )

        scatter_controls = widgets.HBox([
            self._scatter_x_dropdown,
            self._scatter_y_dropdown,
        ])
        scatter_panel = widgets.VBox([
            widgets.HTML("<b>Classification Scatter</b>"),
            scatter_controls,
            self._scatter_output,
        ], layout=widgets.Layout(width='50%'))

        # ---- Results: traces plot ------------------------------------
        self._traces_output = widgets.Output(
            layout=widgets.Layout(height='340px'),
        )
        traces_panel = widgets.VBox([
            widgets.HTML("<b>IV Traces (selected cell)</b>"),
            self._traces_output,
        ], layout=widgets.Layout(width='50%'))

        plots_row = widgets.HBox([scatter_panel, traces_panel])

        results_panel = widgets.VBox([self._table_container, plots_row])

        # ---- Status output & clear -----------------------------------
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

        # ---- Assemble ------------------------------------------------
        self.widget = widgets.VBox([
            widgets.HTML("<h3>🧬 Cell Classification (Ephys)</h3>"),
            step1_box,
            step2_box,
            results_panel,
            widgets.HBox([self._clear_btn, self._done_btn]),
            self.output,
        ])

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_df_selected(self, change):
        """When a DataFrame is chosen, suggest a protocol name."""
        name = self._df_dropdown.value
        if name is None or name not in self._df_dict:
            return
        df = self._df_dict[name]
        # Find first protocol name that looks like an IV-clamp protocol
        protocols: set = set()
        if 'protocols' in df.columns:
            for proto_dict in df['protocols']:
                if isinstance(proto_dict, dict):
                    protocols.update(proto_dict.keys())
        ivic_candidates = sorted(
            p for p in protocols if 'IVIC' in p.upper() or 'IV_IC' in p.upper()
        )
        if ivic_candidates:
            self._protocol_input.value = ivic_candidates[0]

    def _on_clear_output(self, btn):
        """Clear the status output area."""
        with self.output:
            clear_output()

    def _on_done(self, btn):
        """Write classification results back to the source DataFrame via get_results()."""
        with self.output:
            clear_output(wait=True)
            results = self.get_results()
            if results is None:
                print("\u274c No results yet \u2014 run analysis first.")
                return
            name = self._df_dropdown.value
            if name is not None and name in self._df_dict:
                src_df = self._df_dict[name]
                src_df['ephys_props'] = results.reindex(src_df.index)['ephys_props']
                src_df['cell_class'] = results.reindex(src_df.index)['cell_class']
            counts = self.props_df['cell_class'].value_counts().to_dict()
            summary = ', '.join(f'{k}: {v}' for k, v in counts.items())
            print(f"✅ Done — results written back to '{name}'.")
            print(f"   {len(self.props_df)} cells classified: {summary}")

    # ---- Run analysis ------------------------------------------------

    def _on_run_analysis(self, btn):
        """Load, format, and extract IV properties for every cell."""
        with self.output:
            clear_output(wait=True)

            name = self._df_dropdown.value
            if name is None or name not in self._df_dict:
                print("❌ Select a DataFrame first.")
                return

            protocol = self._protocol_input.value.strip()
            if not protocol:
                print("❌ Enter a protocol name first.")
                return

            self.cells_df = self._df_dict[name].copy()
            self.ivic_data = {}
            props_list = []

            total = len(self.cells_df)
            n_no_protocol = 0
            n_error = 0
            print(f"🔄 Processing {total} cells for protocol '{protocol}'…")

            for idx, row in self.cells_df.iterrows():
                proto = row.get('protocols', {})
                if not isinstance(proto, dict):
                    n_no_protocol += 1
                    continue
                files = proto.get(protocol, [])
                if not files:
                    n_no_protocol += 1
                    continue

                # Filter to I-Clamp files only
                files = [x for x in files if x.get('clamp_mode') == 'I-Clamp']
                if not files:
                    n_no_protocol += 1
                    continue

                # Pick last file recorded within 10 min of the first
                abf_file = self._select_ivic_file(files)

                try:
                    rec = load_abf(abf_file)
                    rec_fmt = format_ivic(rec)
                    iv_props = extract_iv_properties(rec_fmt)

                    # Derive scalar summary values for the table
                    _reb = iv_props.get('ap_counts_reb')
                    entry = {
                        'cell_idx':        idx,
                        'ap_halfwidth_ms': float(iv_props.get('ap_halfwidth_s', np.nan)) * 1000,
                        'mem_t_ms':        float(iv_props.get('mem_t_s', np.nan)) * 1000,
                        'max_ap_rate_hz':  float(np.nanmax(iv_props['ap_rates_hz']))
                                           if iv_props.get('ap_rates_hz') is not None
                                              and len(iv_props['ap_rates_hz']) > 0
                                           else np.nan,
                        'mem_r_MOhm':      float(iv_props.get('mem_r_MOhm', np.nan)),
                        'sag_ratio':       float(np.nanmean(iv_props['sag_ratio']))
                                           if iv_props.get('sag_ratio') is not None
                                              and len(np.atleast_1d(iv_props['sag_ratio'])) > 0
                                           else np.nan,
                        'max_reb_aps':     float(np.nanmax(_reb))
                                           if _reb is not None and len(np.atleast_1d(_reb)) > 0
                                           else 0.0,
                        'cell_class':      'Unknown',
                    }
                    props_list.append(entry)
                    self.ivic_data[idx] = rec_fmt

                except Exception as e:
                    n_error += 1
                    print(f"  ⚠️ Skipped cell {idx}: {e}")

            if not props_list:
                print("❌ No cells yielded valid properties.")
                print(f"   {n_no_protocol} cells had no '{protocol}' I-Clamp files, "
                      f"{n_error} failed.")
                self.props_df = None
                return

            self.props_df = pd.DataFrame(props_list).set_index('cell_idx')
            print(f"✅ Processed {len(self.props_df)} / "
                  f"{total - n_no_protocol} cells with '{protocol}' recordings.")
            if n_no_protocol > 0:
                print(f"   ({n_no_protocol} cells had no '{protocol}' I-Clamp files)")
            if n_error > 0:
                print(f"   ({n_error} cells failed — see warnings above)")

        self._apply_classification()

    @staticmethod
    def _select_ivic_file(files: list) -> str:
        """Return the path of the best IVIC file from a list of file dicts.

        Picks the last recording captured within 10 minutes of the first
        (by ABF creation time).  Falls back to the first file if settings
        cannot be read.

        Parameters
        ----------
        files : list of dict
            Each dict has at least ``'file'`` (str path) and
            ``'clamp_mode'`` keys.

        Returns
        -------
        str
        """
        if len(files) == 1:
            return files[0]['file']
        try:
            settings = [load_abf_settings(x['file']) for x in files]
            created = [s['created'] for s in settings]
            # Pair (creation_time, original_index) and sort chronologically
            pairs = sorted(zip(created, range(len(files))))
            first_time = pairs[0][0]
            cutoff = first_time + dt.timedelta(minutes=10)
            valid = [orig_idx for t, orig_idx in pairs if t <= cutoff]
            return files[valid[-1]]['file']
        except Exception:
            return files[0]['file']

    # ---- Classification ---------------------------------------------

    def _on_threshold_changed(self, change):
        """Re-classify cells when any threshold text widget changes."""
        if self._suppress_classify:
            return
        if self.props_df is not None and len(self.props_df) > 0:
            self._apply_classification()

    def _apply_classification(self):
        """Read threshold widgets, classify each cell, refresh views.

        Classification priority: LTS > FSI > RS > Unknown.
        """
        if self.props_df is None:
            return

        fsi_thresholds = {}
        for col in _CLASSIFY_PARAMS:
            val_str = self._thresh_widgets[col].value.strip()
            try:
                fsi_thresholds[col] = float(val_str)
            except ValueError:
                fsi_thresholds[col] = None

        lts_thresholds = {}
        for col in _LTS_PARAMS:
            val_str = self._thresh_widgets[col].value.strip()
            try:
                lts_thresholds[col] = float(val_str)
            except ValueError:
                lts_thresholds[col] = None

        classes = []
        for _, row in self.props_df.iterrows():
            # 1. Test LTS criteria first (takes priority)
            lts_votes = []
            for col in _LTS_PARAMS:
                tv = lts_thresholds[col]
                val = row[col]
                if tv is None or (isinstance(val, float) and np.isnan(val)):
                    lts_votes.append(None)
                    continue
                gte = _LTS_DIRECTION_GTE[col]
                lts_votes.append(bool(val >= tv) if gte else bool(val <= tv))
            valid_lts = [v for v in lts_votes if v is not None]
            if valid_lts and all(valid_lts):
                classes.append('LTS')
                continue

            # 2. FSI / RS voting
            votes = []
            for col in _CLASSIFY_PARAMS:
                tv = fsi_thresholds[col]
                val = row[col]
                if tv is None or (isinstance(val, float) and np.isnan(val)):
                    votes.append(None)
                    continue
                gte = _FSI_DIRECTION_GTE[col]
                votes.append(bool(val >= tv) if gte else bool(val <= tv))

            valid_votes = [v for v in votes if v is not None]
            if not valid_votes:
                classes.append('Unknown')
            elif all(valid_votes):
                classes.append('FSI')
            elif not any(valid_votes):
                classes.append('RS')
            else:
                classes.append('Unknown')

        self.props_df['cell_class'] = classes

        # Write back to source DataFrame
        if self.cells_df is not None:
            self.cells_df['class'] = 'no_recording'
            self.cells_df.loc[self.props_df.index, 'class'] = (
                self.props_df['cell_class']
            )

        self._refresh_table()
        self._refresh_scatter()

    # ---- Table -------------------------------------------------------

    def _refresh_table(self):
        """Rebuild ipydatagrid DataGrid from props_df."""
        if self.props_df is None:
            return

        try:
            from ipydatagrid import DataGrid
        except ImportError:
            self._refresh_table_fallback()
            return

        display_df = self.props_df[_TABLE_COLS].copy().reset_index()

        # Round numeric columns for readability
        for col in _TABLE_COLS[:-1]:   # skip 'cell_class'
            if col in display_df.columns:
                display_df[col] = display_df[col].round(3)

        grid_kwargs = dict(
            selection_mode='row',
            editable=False,
            auto_fit_columns=True,
            layout=widgets.Layout(height='300px', width='100%'),
        )

        try:
            from ipydatagrid import TextRenderer, Expr
            grid_kwargs['default_renderer'] = TextRenderer(
                background_color=Expr(
                    "('#c8e6ff' if cell.metadata.data['cell_class'] == 'FSI' "
                    "else ('#ffddb3' if cell.metadata.data['cell_class'] == 'RS' "
                    "else ('#c8f7c8' if cell.metadata.data['cell_class'] == 'LTS' "
                    "else 'white')))"
                )
            )
        except Exception:
            pass

        grid = DataGrid(display_df, **grid_kwargs)
        grid.on_cell_click(self._on_grid_cell_click)
        self._grid = grid
        self._table_container.children = [grid]


    def _refresh_table_fallback(self):
        """Fallback table when ipydatagrid is unavailable."""
        if self.props_df is None:
            return
        cell_indices = list(self.props_df.index)
        if not hasattr(self, '_fallback_dropdown'):
            self._fallback_dropdown = widgets.Dropdown(
                options=cell_indices, description='Cell:',
                style={'description_width': '40px'},
            )
            self._fallback_dropdown.observe(
                self._on_fallback_cell_selected, names='value'
            )
        else:
            self._fallback_dropdown.options = cell_indices

        display_df = self.props_df[_TABLE_COLS].copy()
        styler = display_df.style.format(
            {c: '{:.3f}' for c in display_df.select_dtypes(include='number').columns}
        ).apply(
            lambda row: [
                f"background-color: {_CLASS_COLOURS.get(row.get('cell_class', 'Unknown'), 'white')}"
                for _ in row
            ], axis=1,
        )
        html = styler.to_html()
        scrollable = widgets.HTML(
            f"<div style='max-height:250px; overflow-y:auto'>{html}</div>"
        )
        self._table_container.children = [self._fallback_dropdown, scrollable]

    def _on_fallback_cell_selected(self, change):
        """Handle cell selection from the fallback dropdown when ipydatagrid is unavailable."""
        self._selected_cell_idx = change['new']
        self._refresh_traces()
        self._refresh_scatter()

    def _on_grid_cell_click(self, event):
        """Handle row selection in the ipydatagrid DataGrid and update plots."""
        row_idx = event.get('row', None)
        if row_idx is None or self.props_df is None:
            return
        try:
            cell_idx = self.props_df.index[row_idx]
        except (IndexError, KeyError):
            return
        self._selected_cell_idx = cell_idx
        self._refresh_traces()
        self._refresh_scatter()

    # ---- Scatter plot -----------------------------------------------

    def _on_scatter_param_changed(self, change):
        """Redraw the scatter plot when the X or Y axis dropdown changes."""
        self._refresh_scatter()

    def _refresh_scatter(self):
        """Redraw scatter coloured by class with threshold lines."""
        if self.props_df is None or len(self.props_df) == 0:
            return

        x_col = self._scatter_x_dropdown.value
        y_col = self._scatter_y_dropdown.value
        df = self.props_df

        _colour_map = {'FSI': '#4e8dd9', 'RS': '#f0933a', 'LTS': '#3dbd6b', 'Unknown': '#aaaaaa'}

        with self._scatter_output:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(4.5, 3.5))

            artist_map = {}
            for cell_class, colour in _colour_map.items():
                sub = df[df['cell_class'] == cell_class]
                if len(sub) == 0:
                    continue
                sc = ax.scatter(
                    sub[x_col], sub[y_col],
                    c=colour, label=cell_class,
                    s=40, alpha=0.8, edgecolors='white', linewidths=0.5,
                    picker=True, pickradius=5,
                )
                artist_map[sc] = list(sub.index)
            self._scatter_artists = artist_map

            # Highlight selected cell
            sel = self._selected_cell_idx
            if sel is not None and sel in df.index:
                ax.scatter(
                    [df.loc[sel, x_col]], [df.loc[sel, y_col]],
                    s=150, facecolors='none', edgecolors='red',
                    linewidths=2, zorder=5,
                )

            # Threshold lines for classification params
            for col, axis in [(x_col, 'x'), (y_col, 'y')]:
                if col not in self._thresh_widgets:
                    continue
                tv_str = self._thresh_widgets[col].value.strip()
                try:
                    tv = float(tv_str)
                except ValueError:
                    continue
                if col in _CLASSIFY_PARAMS:
                    gte = _FSI_DIRECTION_GTE[col]
                    sym = '≥' if gte else '≤'
                    kw = dict(color='#4e8dd9', ls='--', lw=1,
                              label=f'FSI: {_PARAM_LABELS[col]} {sym} {tv}')
                elif col in _LTS_PARAMS:
                    gte = _LTS_DIRECTION_GTE[col]
                    sym = '≥' if gte else '≤'
                    kw = dict(color='#3dbd6b', ls=':', lw=1.5,
                              label=f'LTS: {_PARAM_LABELS[col]} {sym} {tv}')
                else:
                    continue
                if axis == 'x':
                    ax.axvline(tv, **kw)
                else:
                    ax.axhline(tv, **kw)

            ax.set_xlabel(_PARAM_LABELS.get(x_col, x_col), fontsize=9)
            ax.set_ylabel(_PARAM_LABELS.get(y_col, y_col), fontsize=9)
            ax.set_title('Cell Classification  (click a point to select)', fontsize=10)
            ax.legend(fontsize=8, loc='best')
            fig.tight_layout()
            fig.canvas.mpl_connect('pick_event', self._on_scatter_pick)
            plt.show()

    def _on_scatter_pick(self, event):
        """Handle a click on a scatter point — select that cell in the table and plots."""
        if not hasattr(event, 'ind') or len(event.ind) == 0:
            return
        artist = event.artist
        if artist not in self._scatter_artists:
            return
        cell_idx = self._scatter_artists[artist][int(event.ind[0])]
        if cell_idx == self._selected_cell_idx:
            return
        self._selected_cell_idx = cell_idx
        # Sync table row highlight if ipydatagrid is active
        if self._grid is not None and self.props_df is not None:
            try:
                row_pos = self.props_df.index.get_loc(cell_idx)
                n_cols = len(_TABLE_COLS)
                self._grid.selections = [
                    {'r1': row_pos, 'c1': 0, 'r2': row_pos, 'c2': n_cols - 1}
                ]
            except Exception:
                pass
        self._refresh_traces()
        self._refresh_scatter()

    # ---- Traces plot ------------------------------------------------

    def _refresh_traces(self):
        """Plot stacked IV-curve traces for the selected cell using stacked_lines()."""
        idx = self._selected_cell_idx
        if idx is None or idx not in self.ivic_data:
            return

        rec = self.ivic_data[idx]

        # Find voltage channel
        ch_units = rec['ch_units'].values
        mv_indices = np.where(ch_units == 'mV')[0]
        if len(mv_indices) == 0:
            return
        v_ch = int(mv_indices[0])
        v_traces = rec.sel({DIM_CHAN: v_ch})   # shape: (time, epoch)

        # Sort epochs by injected current
        if 'i_inject' in v_traces.coords:
            i_vals = v_traces.coords['i_inject'].values
        elif 'i_inject' in rec.coords:
            i_vals = rec.coords['i_inject'].values
        else:
            i_vals = np.arange(v_traces.sizes[DIM_EPOCH])

        sort_order = np.argsort(i_vals)
        i_sorted = i_vals[sort_order]
        v_sorted = v_traces.isel({DIM_EPOCH: sort_order})

        # Round i_inject coordinate to integers for right-axis tick labels
        i_rounded = np.round(i_sorted).astype(int)
        v_sorted = v_sorted.assign_coords({'i_inject': (DIM_EPOCH, i_rounded)})

        # Build per-epoch colours from a coolwarm colormap
        n_epochs = len(sort_order)
        i_min, i_max = float(i_sorted.min()), float(i_sorted.max())
        i_range = i_max - i_min if i_max != i_min else 1.0
        cmap = cm.get_cmap('coolwarm', n_epochs)
        colors = [cmap((i_sorted[k] - i_min) / i_range) for k in range(n_epochs)]

        with self._traces_output:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(4.5, 3.5))

            stacked_lines(
                v_sorted,
                time_dim=DIM_TIME,
                chan_dim=DIM_EPOCH,
                label_coord='i_inject',
                disp_color=colors,
                sep_ratio=1,
                ax=ax,
                lw=0.8,
                alpha=0.85,
            )

            # Add 100 mV voltage scale bar at bottom-right
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x_sb = xlim[1] - (xlim[1] - xlim[0]) * 0.04
            y_sb = ylim[0] + (ylim[1] - ylim[0]) * 0.03
            ax.plot([x_sb, x_sb], [y_sb, y_sb + 100],
                    color='black', lw=2, clip_on=False, solid_capstyle='butt')
            ax.text(x_sb - (xlim[1] - xlim[0]) * 0.01, y_sb + 50,
                    '100 mV', ha='right', va='center', fontsize=8)

            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel('Voltage + offset (mV)', fontsize=9)
            ax.set_yticks([])

            cls = self.props_df.loc[idx, 'cell_class'] if self.props_df is not None else ''
            hw = self.props_df.loc[idx, 'ap_halfwidth_ms'] if self.props_df is not None else np.nan
            ax.set_title(
                f'Cell {idx}  |  class = {cls},  AP hw = {hw:.2f} ms',
                fontsize=9,
            )
            fig.tight_layout()
            plt.show()
