"""NIDQ channel configuration widget for mimo_pack.

Provides :class:`NidqChannelConfigWidget`, an interactive Jupyter widget
for visually inspecting all NIDQ channels in a recording session and
assigning per-channel metadata (name, type, units, notes).  Configured
values are persisted to ``session_config.json`` and can be read back by
:func:`mimo_pack.fileio.nidq.load_nidq` via its ``chan_names`` parameter.

Author: Drew B. Headley
"""

import io
import json

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output, Image

import xarray as xr

from mimo_pack.consts import DIM_TIME, DIM_CHAN, ATTR_FS
from mimo_pack.plot.timeseries import stacked_lines
from mimo_pack.plot.colors import MIMO_PALETTE
from mimo_pack.fileio.nidq import (
    _sglx_prefix,
    _SGLX_DIGITAL_PREFIXES,
    _SGLX_ALL_PREFIXES,
)


# Channel type vocabulary (enforced via the type column)
_CHAN_TYPES = ('analog', 'digital', 'ttl', 'sync', 'other')


class NidqChannelConfigWidget:
    """Interactive widget for inspecting and labelling NIDQ channels.

    Loads a NIDQ recording from the dclut file discovered via the session,
    renders a full-session overview and a scrollable time-window preview
    of all channels, and lets you configure a *name*, *type*, *units*, and
    *notes* for every channel.  Clicking **Save to Config** writes the
    configuration to ``session_config.json`` under the key
    ``parameters.nidq_channels``.

    Parameters
    ----------
    session : Session
        Loaded :class:`~mimo_pack.core.Session` object for the recording.

    Examples
    --------
    >>> from mimo_pack.core import Session
    >>> from mimo_pack.widgets import NidqChannelConfigWidget
    >>> sess = Session('/data/DH51')
    >>> nidq_widget = NidqChannelConfigWidget(sess)
    >>> nidq_widget.display()
    """

    def __init__(self, session):
        self.session = session
        self._data: xr.DataArray | None = None
        self._chan_ids: list = []
        self._chan_type_map: dict = {}   # cid -> 'analog' | 'digital'
        self._grid = None    # kept for backward compat; unused
        self._row_widgets: list = []   # list of per-row widget dicts
        self._df: pd.DataFrame | None = None
        self.create_widgets()

    # ------------------------------------------------------------------
    # Widget construction
    # ------------------------------------------------------------------

    def create_widgets(self):
        """Build all ipywidget controls and assemble the layout."""
        # ---- Header -------------------------------------------------------
        title = widgets.HTML("<h3>&#x1F4E1; NIDQ Channel Configuration</h3>")
        session_info = widgets.HTML(
            "<div style='background:#f0f8ff;padding:8px;border:1px solid #ccc;"
            "border-radius:4px;margin-bottom:6px;'>"
            f"<b>Session:</b> {self.session.dir}"
            "</div>"
        )

        # ---- Load section -------------------------------------------------
        load_label = widgets.HTML("<h4>Load NIDQ File</h4>")
        default_path = self._discover_nidq_path()

        self.path_text = widgets.Text(
            value=default_path,
            description='NIDQ path:',
            placeholder='Path to *.nidq_dclut.json',
            layout=widgets.Layout(width='620px'),
            style={'description_width': '90px'},
        )
        self.load_btn = widgets.Button(
            description='Load',
            button_style='primary',
            layout=widgets.Layout(width='80px'),
        )
        self.status_output = widgets.Output(
            layout=widgets.Layout(width='100%')
        )

        # ---- Overview section ---------------------------------------------
        overview_label = widgets.HTML(
            "<h4>Overview <small style='color:#888;font-weight:normal;'>"
            "(full session, all channels, downsampled)</small></h4>"
        )
        self.overview_output = widgets.Output(
            layout=widgets.Layout(width='100%', min_height='200px')
        )

        # ---- Channel configuration section --------------------------------
        config_label = widgets.HTML("<h4>Channel Configuration</h4>")
        config_help = widgets.HTML(
            "<small><b>name</b> — SpikeGLX channel name (read-only).  "
            "<b>label</b> — your custom display name (editable).  "
            "<b>type</b> — dropdown (auto-detected from XD/XA).  "
            "<b>units</b> &amp; <b>notes</b> — free text.  "
            "Click <b>Save to Config</b> to persist.</small>"
        )
        self.grid_container = widgets.Output(
            layout=widgets.Layout(width='100%', min_height='80px')
        )

        # ---- Preview section ----------------------------------------------
        preview_label = widgets.HTML("<h4>Channel Preview</h4>")
        self.win_dur_text = widgets.BoundedFloatText(
            value=10.0,
            min=0.1,
            max=3600.0,
            step=1.0,
            description='Window (s):',
            continuous_update=False,
            layout=widgets.Layout(width='180px'),
            style={'description_width': '85px'},
            disabled=True,
        )
        self.center_slider = widgets.FloatSlider(
            value=5.0,
            min=0.0,
            max=10.0,
            step=0.5,
            description='Centre (s):',
            continuous_update=False,
            readout_format='.2f',
            layout=widgets.Layout(width='500px'),
            style={'description_width': '80px'},
            disabled=True,
        )
        self.update_preview_btn = widgets.Button(
            description='Update Preview',
            button_style='info',
            layout=widgets.Layout(width='150px'),
            disabled=True,
        )
        self.preview_output = widgets.Output(
            layout=widgets.Layout(width='100%', min_height='200px')
        )
        self.chan_range_slider = widgets.IntRangeSlider(
            value=[0, 0],
            min=0,
            max=0,
            step=1,
            description='Chans:',
            orientation='vertical',
            continuous_update=False,
            readout=True,
            layout=widgets.Layout(height='300px', width='80px'),
            style={'description_width': '50px'},
            disabled=True,
        )

        # ---- Action section -----------------------------------------------
        self.save_btn = widgets.Button(
            description='\U0001f4be Save to Config',
            button_style='success',
            layout=widgets.Layout(width='160px'),
            disabled=True,
        )
        self.write_dclut_btn = widgets.Button(
            description='\U0001f4dd Write to dclut JSON',
            button_style='primary',
            tooltip='Add label and chan_type list scales to the dclut JSON file',
            layout=widgets.Layout(width='190px'),
            disabled=True,
        )
        self.clear_btn = widgets.Button(
            description='\U0001f9f9 Clear Output',
            button_style='warning',
            layout=widgets.Layout(width='140px'),
        )
        self.action_output = widgets.Output(
            layout=widgets.Layout(width='100%')
        )

        # ---- Wire events --------------------------------------------------
        self.load_btn.on_click(self._on_load)
        self.center_slider.observe(self._on_slider_change, names='value')
        self.win_dur_text.observe(self._on_slider_change, names='value')
        self.update_preview_btn.on_click(self._on_update_preview)
        self.chan_range_slider.observe(self._on_slider_change, names='value')
        self.save_btn.on_click(self._on_save_config)
        self.write_dclut_btn.on_click(self._on_write_dclut)
        self.clear_btn.on_click(self._on_clear)

        # ---- Assemble layout ----------------------------------------------
        self.container = widgets.VBox([
            title,
            session_info,
            load_label,
            widgets.HBox([self.path_text, self.load_btn]),
            self.status_output,
            overview_label,
            self.overview_output,
            config_label,
            config_help,
            self.grid_container,
            preview_label,
            widgets.HBox([self.win_dur_text, self.update_preview_btn]),
            self.center_slider,
            widgets.HBox(
                [self.preview_output, self.chan_range_slider],
                layout=widgets.Layout(align_items='stretch'),
            ),
            widgets.HBox([self.save_btn, self.write_dclut_btn, self.clear_btn]),
            self.action_output,
        ])

        # Auto-load if a path was detected
        if default_path:
            self._load_data(default_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _discover_nidq_path(self) -> str:
        """Return the first NIDQ dclut path found in the session, or ''."""
        try:
            bin_files = self.session.find_files('nidq_pattern')
            if bin_files:
                return str(self.session.get_dclut_path(str(bin_files[0])))
        except Exception:
            pass
        return ''

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _on_load(self, btn):
        """Load button handler."""
        path = self.path_text.value.strip()
        if not path:
            with self.status_output:
                clear_output(wait=True)
                print("⚠️  Please enter a NIDQ dclut path.")
            return
        self._load_data(path)

    def _load_data(self, path: str):
        """Load the NIDQ DataArray and refresh all UI sections."""
        with self.status_output:
            clear_output(wait=True)
            print(f"Loading {path} …")

        try:
            from mimo_pack.fileio.nidq import load_nidq
            data = load_nidq(path)
        except Exception as exc:
            with self.status_output:
                clear_output(wait=True)
                print(f"❌ Failed to load:\n{exc}")
            return

        self._data = data

        # Channel IDs as strings (used as dict keys throughout)
        self._chan_ids = [str(v) for v in data[DIM_CHAN].values]
        n_chans = len(self._chan_ids)

        # Status summary
        with self.status_output:
            clear_output(wait=True)
            fs = float(data.attrs.get(ATTR_FS, 0))
            t_start = float(data[DIM_TIME].values[0])
            t_end = float(data[DIM_TIME].values[-1])
            print(
                f"✅  {n_chans} channels  |  "
                f"fs = {fs:.1f} Hz  |  "
                f"duration = {t_end - t_start:.1f} s  "
                f"({t_start:.2f} – {t_end:.2f} s)"
            )

        # Build config DataFrame from saved values or defaults
        saved = (
            self.session.config
            .get('parameters', {})
            .get('nidq_channels', {})
        )

        # Build chan_type map from the 'chan_type' coordinate (added by expand_digital)
        chan_type_map: dict = {}
        if 'chan_type' in data.coords:
            for cid, ct in zip(self._chan_ids, data.coords['chan_type'].values):
                chan_type_map[str(cid)] = str(ct)
        self._chan_type_map = chan_type_map

        # Extract SpikeGLX names from any extra channel-indexed coordinate
        # (the name scale in the dclut JSON), skipping the chan_type coord.
        _SKIP_COORDS = {'chan_type'}
        dclut_names: dict = {}
        for coord_name, coord_arr in data.coords.items():
            if coord_name in _SKIP_COORDS or coord_name == DIM_CHAN:
                continue
            if DIM_CHAN not in coord_arr.dims:
                continue
            for cid, v in zip(self._chan_ids, coord_arr.values):
                s_val = str(v)
                if s_val and s_val.lower() not in ('nan', 'none', ''):
                    dclut_names[cid] = s_val
            if dclut_names:   # use first useful coord
                break

        rows = []
        for cid in self._chan_ids:
            s = saved.get(cid, {})
            ctype = chan_type_map.get(cid, 'analog')
            default_type  = s.get('type',  ctype)
            default_units = s.get('units', '' if ctype == 'digital' else 'V')
            rows.append({
                'channel': cid,
                'name':    dclut_names.get(cid, cid),  # SpikeGLX name (fixed)
                'label':   s.get('label', ''),          # user display label
                'type':    default_type,
                'units':   default_units,
                'notes':   s.get('notes', ''),
            })
        self._df = pd.DataFrame(rows)

        # Refresh grid
        self._build_grid()

        # Configure preview controls
        dur = t_end - t_start
        window = min(10.0, dur)
        half = window / 2.0
        step = max(0.1, round(dur / 400, 1))
        # Set slider bounds/step first, then value (avoids clamping)
        self.center_slider.min = t_start
        self.center_slider.max = t_end
        self.center_slider.step = step
        self.center_slider.value = t_start + half
        self.win_dur_text.max = dur
        self.win_dur_text.value = window
        self.center_slider.disabled = False
        self.win_dur_text.disabled = False
        self.update_preview_btn.disabled = False
        self.save_btn.disabled = False
        self.write_dclut_btn.disabled = False

        # Configure channel-range slider
        n_ch = len(self._chan_ids)
        self.chan_range_slider.min = 0
        self.chan_range_slider.max = max(0, n_ch - 1)
        self.chan_range_slider.value = [0, max(0, n_ch - 1)]
        slider_h = max(120, min(600, 20 + n_ch * 20))
        self.chan_range_slider.layout.height = f'{slider_h}px'
        self.chan_range_slider.disabled = False

        # Render plots
        self._render_overview()
        t_c = self.center_slider.value
        half_w = self.win_dur_text.value / 2.0
        self._render_preview(t_c - half_w, t_c + half_w)

    # ------------------------------------------------------------------
    # DataGrid (channel config table)
    # ------------------------------------------------------------------

    # Column layout widths — single source of truth used by both header and data rows
    _COL_WIDTHS = {
        'channel': '82px',
        'name':    '142px',
        'label':   '142px',
        'type':    '122px',
        'units':   '72px',
        'notes':   '242px',
    }

    def _build_grid(self):
        """Build an editable per-channel widget table with a Dropdown for type."""
        if self._df is None:
            return

        cw = self._COL_WIDTHS

        # Header row — HTML widgets share the same layout widths as data cells
        def _hdr(label, width, title=''):
            tip = f" title='{title}'" if title else ''
            return widgets.HTML(
                value=f"<b style='display:inline-block'{tip}>{label}</b>",
                layout=widgets.Layout(width=width, margin='0 2px 2px 0'),
            )
        header = widgets.HBox([
            _hdr('channel', cw['channel']),
            _hdr('name',    cw['name'],  'SpikeGLX channel name (read-only)'),
            _hdr('label',   cw['label'], 'Your custom display name'),
            _hdr('type',    cw['type']),
            _hdr('units',   cw['units']),
            _hdr('notes',   cw['notes']),
        ], layout=widgets.Layout(border_bottom='2px solid #bbb', margin_bottom='2px'))

        self._row_widgets = []  # list of dicts per row
        row_boxes = []
        for _, row in self._df.iterrows():
            cid = str(row['channel'])
            is_digital = self._chan_type_map.get(cid, 'analog') == 'digital'
            row_bg = "background:#f8f8f8;" if is_digital else ""

            chan_lbl = widgets.Label(
                value=cid,
                layout=widgets.Layout(width=cw['channel'], margin='0 2px 0 0'),
            )
            # name is read-only — show as greyed-out HTML label
            w_name = widgets.HTML(
                value=(
                    f"<span style='font-family:monospace;font-size:0.85em;"
                    f"color:#555;display:inline-block;overflow:hidden;"
                    f"white-space:nowrap;{row_bg}'>"
                    f"{str(row.get('name', cid))}</span>"
                ),
                layout=widgets.Layout(width=cw['name'], margin='0 2px 0 0'),
            )
            w_label = widgets.Text(
                value=str(row.get('label', '')),
                placeholder='display name',
                layout=widgets.Layout(width=cw['label'], margin='0 2px 0 0'),
            )
            type_val = row.get('type', 'analog')
            w_type = widgets.Dropdown(
                options=list(_CHAN_TYPES),
                value=type_val if type_val in _CHAN_TYPES else 'analog',
                layout=widgets.Layout(width=cw['type'], margin='0 2px 0 0'),
            )
            w_units = widgets.Text(
                value=str(row.get('units', '')),
                layout=widgets.Layout(width=cw['units'], margin='0 2px 0 0'),
            )
            w_notes = widgets.Text(
                value=str(row.get('notes', '')),
                layout=widgets.Layout(width=cw['notes'], margin='0 2px 0 0'),
            )
            self._row_widgets.append({
                'channel': cid,
                'name':    str(row.get('name', cid)),  # stored string, not widget
                'label':   w_label,
                'type':    w_type,
                'units':   w_units,
                'notes':   w_notes,
            })
            row_boxes.append(widgets.HBox(
                [chan_lbl, w_name, w_label, w_type, w_units, w_notes]
            ))

        total_h = 52 + len(row_boxes) * 40  # 40 px per row, 52 px for header
        table_box = widgets.VBox(
            [header] + row_boxes,
            layout=widgets.Layout(
                height=f'{total_h}px',
                border='1px solid #ccc',
                padding='4px',
            ),
        )
        with self.grid_container:
            clear_output(wait=True)
            display(table_box)

    def _get_grid_df(self) -> pd.DataFrame:
        """Return the current channel config table as a DataFrame."""
        if self._row_widgets:
            rows = []
            for rw in self._row_widgets:
                rows.append({
                    'channel': rw['channel'],
                    'name':    rw['name'],           # plain string
                    'label':   rw['label'].value,
                    'type':    rw['type'].value,
                    'units':   rw['units'].value,
                    'notes':   rw['notes'].value,
                })
            return pd.DataFrame(rows)
        if self._df is not None:
            return self._df.copy()
        return pd.DataFrame(
            columns=['channel', 'name', 'label', 'type', 'units', 'notes']
        )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _chan_colors(self, chan_ids: list) -> list[str]:
        """Per-channel color list: black for digital, palette for analog."""
        palette_idx = 0
        colors = []
        for cid in chan_ids:
            if self._chan_type_map.get(str(cid), 'analog') == 'digital':
                colors.append('black')
            else:
                colors.append(MIMO_PALETTE[palette_idx % len(MIMO_PALETTE)])
                palette_idx += 1
        return colors

    def _render_overview(self):
        """Render a downsampled full-session stacked overview."""
        if self._data is None:
            return

        data = self._data
        n_time = data.sizes[DIM_TIME]
        skip = max(1, n_time // 2000)
        ds = data.isel({DIM_TIME: slice(None, None, skip)})

        chan_ids = [str(v) for v in ds[DIM_CHAN].values]
        colors = self._chan_colors(chan_ids)

        fig_h = max(3.0, len(chan_ids) * 0.35)
        fig, ax = plt.subplots(figsize=(12, fig_h))
        stacked_lines(
            ds,
            disp_color=colors,
            sep_ratio=1.2,
            ax=ax,
        )
        ax.set_xlabel('Time (s)')
        ax.set_title('NIDQ Overview — all channels (downsampled)')
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        with self.overview_output:
            clear_output(wait=True)
            display(Image(data=buf.read()))

    def _render_preview(self, t_start: float, t_end: float):
        """Render a zoomed stacked-trace preview for *t_start*–*t_end*."""
        if self._data is None:
            return

        # Apply channel-range selection
        all_chans = list(self._data[DIM_CHAN].values)
        lo, hi = self.chan_range_slider.value
        lo = max(0, min(lo, len(all_chans) - 1))
        hi = max(lo, min(hi, len(all_chans) - 1))
        sel_chans = all_chans[lo:hi + 1]
        data = self._data.sel({DIM_CHAN: sel_chans, DIM_TIME: slice(t_start, t_end)})
        if data.sizes[DIM_TIME] == 0:
            with self.preview_output:
                clear_output(wait=True)
                print("⚠️  No samples in selected time window.")
            return

        # Get channel display labels: prefer user label, fall back to SpikeGLX name
        df = self._get_grid_df()
        chan_vals = [str(v) for v in data[DIM_CHAN].values]
        labels = []
        for cid in chan_vals:
            row = df[df['channel'].astype(str) == cid]
            if not row.empty:
                user_label = str(row.iloc[0].get('label', ''))
                sglx_name  = str(row.iloc[0].get('name', ''))
                labels.append(user_label or sglx_name or cid)
            else:
                labels.append(cid)

        colors = self._chan_colors(chan_vals)

        # Temporarily attach display labels as a coordinate for stacked_lines
        data = data.assign_coords({'_label': (DIM_CHAN, labels)})

        fig_h = max(3.0, len(chan_vals) * 0.40)
        fig, ax = plt.subplots(figsize=(12, fig_h))
        stacked_lines(
            data,
            label_coord='_label',
            disp_color=colors,
            sep_ratio=1.2,
            ax=ax,
        )
        ax.set_xlabel('Time (s)')
        ax.set_title(f'Preview: {t_start:.2f} – {t_end:.2f} s')
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        with self.preview_output:
            clear_output(wait=True)
            display(Image(data=buf.read()))

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_slider_change(self, change):
        """Re-render preview when centre or duration changes."""
        t_c = self.center_slider.value
        half_w = self.win_dur_text.value / 2.0
        self._render_preview(t_c - half_w, t_c + half_w)

    def _on_update_preview(self, btn):
        """Update Preview button handler."""
        t_c = self.center_slider.value
        half_w = self.win_dur_text.value / 2.0
        self._render_preview(t_c - half_w, t_c + half_w)

    def _on_save_config(self, btn):
        """Write channel configuration to session_config.json."""
        df = self._get_grid_df()

        nidq_channels: dict = {}
        for _, row in df.iterrows():
            cid = str(row['channel'])
            nidq_channels[cid] = {
                'name':  str(row.get('name', '')),
                'label': str(row.get('label', '')),
                'type':  str(row.get('type', 'analog')),
                'units': str(row.get('units', '')),
                'notes': str(row.get('notes', '')),
            }

        if 'parameters' not in self.session.config:
            self.session.config['parameters'] = {}
        self.session.config['parameters']['nidq_channels'] = nidq_channels

        with open(self.session.config_path, 'w') as f:
            json.dump(self.session.config, f, indent=2)

        with self.action_output:
            clear_output(wait=True)
            print(
                f"✅  Saved {len(nidq_channels)} channel configs → "
                f"{self.session.config_path}"
            )

    def _on_write_dclut(self, btn):
        """Write 'label' and 'chan_type' list scales into the dclut JSON file.

        Each entry corresponds to one *original* channel in the binary file
        (pre-digital-expansion).  For XD digital-word channels the 16 bit
        labels are packed as a semicolon-separated string
        ``"bit00_label;bit01_label;…;bit15_label"``.
        """
        from pathlib import Path

        path = self.path_text.value.strip()
        if not path or not Path(path).exists():
            with self.action_output:
                clear_output(wait=True)
                print("\u26a0\ufe0f  No valid dclut path loaded.")
            return

        with open(path, 'r') as fh:
            dcl_json = json.load(fh)

        shape = dcl_json.get('file', {}).get('shape', [])
        if len(shape) < 2:
            with self.action_output:
                clear_output(wait=True)
                print("\u26a0\ufe0f  Cannot determine channel dimension from dclut JSON.")
            return

        chan_dim_idx = 1          # NIDQ is always (time, channel)
        n_orig_chans = shape[chan_dim_idx]
        scales = dcl_json.get('scales', {})

        # Find the existing SpikeGLX name scale (list, dim=1) to map back to
        # original channels
        orig_names = None
        for sc in scales.values():
            if sc.get('dim') == chan_dim_idx and sc.get('type') == 'list':
                vals = sc.get('values') or []
                if len(vals) == n_orig_chans:
                    sample = str(vals[0]) if vals else ''
                    if ';' in sample or _sglx_prefix(sample) in _SGLX_ALL_PREFIXES:
                        orig_names = [str(v) for v in vals]
                        break
        if orig_names is None:
            orig_names = [str(i) for i in range(n_orig_chans)]

        # Current widget config keyed by channel ID string
        df = self._get_grid_df()
        cfg_by_chan = {str(r['channel']): r for _, r in df.iterrows()}
        # Also index by SpikeGLX name (for analog channels looked up by name)
        cfg_by_name = {str(r.get('name', '')): r for _, r in df.iterrows()}

        label_vals: list[str] = []
        type_vals: list[str] = []

        for orig_name in orig_names:
            prefix = _sglx_prefix(orig_name)
            if prefix in _SGLX_DIGITAL_PREFIXES:
                xd_base = orig_name.split(';')[0]   # e.g. 'XD0'
                bit_labels = []
                bit_type = 'digital'
                for bit in range(16):
                    bit_id = f"{xd_base}_b{bit:02d}"
                    row = cfg_by_chan.get(bit_id, {})
                    bit_labels.append(str(row.get('label', '')))
                    if bit == 0:
                        bit_type = str(row.get('type', 'digital'))
                label_vals.append(';'.join(bit_labels))
                type_vals.append(bit_type)
            else:
                row = cfg_by_name.get(orig_name) or cfg_by_chan.get(orig_name, {})
                label_vals.append(str(row.get('label', '')))
                type_vals.append(str(row.get('type', 'analog')))

        # Remove stale label / chan_type scales then write fresh ones
        for key in list(scales.keys()):
            if key in ('label', 'chan_type'):
                del scales[key]

        scales['label'] = {
            'dim': chan_dim_idx, 'unit': '', 'type': 'list',
            'values': label_vals, 'params': {},
        }
        scales['chan_type'] = {
            'dim': chan_dim_idx, 'unit': '', 'type': 'list',
            'values': type_vals, 'params': {},
        }
        dcl_json['scales'] = scales

        with open(path, 'w') as fh:
            json.dump(dcl_json, fh, indent=2)

        with self.action_output:
            clear_output(wait=True)
            print(
                f"\u2705  Wrote 'label' and 'chan_type' scales "
                f"({n_orig_chans} original channels) \u2192 {path}"
            )

    def _on_clear(self, btn):
        """Clear status and action output areas."""
        self.status_output.clear_output()
        self.action_output.clear_output()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def display(self):
        """Display the widget in a Jupyter notebook cell."""
        display(self.container)
