"""Interactive widget for filtering cells by protocol and clamp mode.

Provides a row-per-(protocol, clamp_mode) checklist with optional group
tags, per-row file-count thresholds, AND / OR filter logic, and
automatic gray-out of rows that cannot contribute to the current filter.
"""

import ipywidgets as widgets
from IPython.display import display
import pandas as pd

from mimo_pack.fileio.invitro import summarize_invitro_protocols


class CellSelectorWidget:
    """Filter a cells DataFrame by protocol and clamp mode combinations.

    Each row in the checklist corresponds to a unique (protocol,
    clamp_mode) combination.  Optional group tags and per-row file-count
    thresholds can be set interactively.

    Grouping
    --------
    Rows sharing the same non-empty group tag are OR-ed together within
    that group.  The AND / OR toggle then applies *across* groups (and
    standalone rows).  Leading / trailing whitespace in group names is
    stripped.

    Graying out
    -----------
    * **AND mode** — rows absent from cells already satisfying the
      current AND selection are grayed out and disabled.
    * **OR mode** — rows whose (protocol, clamp_mode) exists in zero
      source cells are grayed out and disabled.

    File-count threshold
    --------------------
    Each row has a text box for an integer ``n`` and a ``≥`` / ``<``
    toggle.  When set, a cell passes that row only if its file count for
    that (protocol, clamp_mode) satisfies the threshold.

    Usage
    -----
    >>> widget = CellSelectorWidget()
    >>> widget.set_dataframes({'cells': cells_df})
    >>> widget.display()
    # interact — click Apply when done
    >>> filtered = widget.cells_df
    """

    def __init__(self) -> None:
        self.cells_df: pd.DataFrame | None = None
        self._source_df: pd.DataFrame | None = None
        self._rows: list[dict] = []
        self._block_updates: int = 0
        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_dataframes(self, df_dict: dict) -> None:
        """Load a cells DataFrame into the widget.

        Parameters
        ----------
        df_dict : dict
            Must contain key ``'cells'`` mapping to a DataFrame produced by
            :func:`~mimo_pack.fileio.invitro.invitro_files_df`.
        """
        if 'cells' not in df_dict:
            raise KeyError("df_dict must contain key 'cells'.")
        self._source_df = df_dict['cells']
        self._populate_checklist(self._source_df)

    def display(self) -> None:
        """Render the widget in the current notebook cell."""
        display(self._container)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Construct all ipywidgets; checklist is empty until set_dataframes."""
        header = widgets.HTML(
            "<b>Select protocol / clamp-mode combinations</b>"
        )

        self._logic_toggle = widgets.ToggleButtons(
            options=['AND', 'OR'],
            value='OR',
            description='Logic:',
            style={'description_width': 'initial'},
            tooltips=[
                'Keep cells that have ALL selected groups',
                'Keep cells that have ANY selected group',
            ],
        )
        self._logic_toggle.observe(lambda _: self._on_change(), names='value')

        btn_all = widgets.Button(
            description='Select All',
            layout=widgets.Layout(width='110px'),
        )
        btn_none = widgets.Button(
            description='Deselect All',
            layout=widgets.Layout(width='110px'),
        )
        btn_all.on_click(lambda _: self._set_all(True))
        btn_none.on_click(lambda _: self._set_all(False))

        # Column header
        col_header = widgets.HBox(
            [
                widgets.HTML(
                    '',
                    layout=widgets.Layout(width='34px', min_width='34px'),
                ),
                widgets.HTML(
                    "<b style='font-size:0.88em'>Protocol [Clamp] — cells, files</b>",
                    layout=widgets.Layout(width='340px', min_width='340px'),
                ),
                widgets.HTML(
                    "<b style='font-size:0.88em'>Group</b>",
                    layout=widgets.Layout(width='112px', min_width='112px'),
                ),
                widgets.HTML(
                    "<b style='font-size:0.88em'>N files</b>",
                    layout=widgets.Layout(width='74px', min_width='74px'),
                ),
                widgets.HTML(
                    "<b style='font-size:0.88em'>≥ / &lt;</b>",
                    layout=widgets.Layout(width='48px', min_width='48px'),
                ),
            ],
            layout=widgets.Layout(align_items='center', padding='2px 4px'),
        )

        self._checklist_box = widgets.VBox(
            [],
            layout=widgets.Layout(
                max_height='380px',
                overflow_y='auto',
                border='1px solid #ccc',
                padding='2px 4px',
            ),
        )

        self._summary_label = widgets.Label('')

        self._apply_btn = widgets.Button(
            description='Apply Filter',
            button_style='primary',
            layout=widgets.Layout(width='140px'),
        )
        self._apply_btn.on_click(lambda _: self._apply_filter())

        self._output = widgets.Output()

        self._container = widgets.VBox([
            header,
            self._logic_toggle,
            widgets.HBox([btn_all, btn_none]),
            col_header,
            self._checklist_box,
            self._summary_label,
            self._apply_btn,
            self._output,
        ])

    # ------------------------------------------------------------------
    # Checklist population
    # ------------------------------------------------------------------

    def _populate_checklist(self, df: pd.DataFrame) -> None:
        """Build one row per (protocol, clamp_mode) combination."""
        summary = summarize_invitro_protocols(df)
        summary = summary.sort_values(
            ['protocol', 'clamp_mode'], ascending=[True, True]
        ).reset_index(drop=True)
        self._rows = []
        row_widgets = []

        for _, s_row in summary.iterrows():
            proto = str(s_row['protocol'])
            mode = str(s_row['clamp_mode'])
            n_cells = int(s_row['n_cells'])
            n_total = int(s_row['n_total'])
            label_text = f"{proto}  [{mode}]  — {n_cells} cells, {n_total} files"

            # Selection checkbox (no text — label is a separate HTML widget)
            cb = widgets.Checkbox(
                value=False,
                description='',
                indent=False,
                layout=widgets.Layout(width='30px', min_width='30px', height='28px'),
            )
            cb.observe(lambda _: self._on_change(), names='value')

            label = widgets.HTML(
                f"<span style='font-size:0.88em'>{label_text}</span>",
                layout=widgets.Layout(width='340px', min_width='340px'),
            )

            group_input = widgets.Text(
                value='',
                placeholder='group',
                layout=widgets.Layout(width='108px', min_width='108px', height='28px'),
            )
            group_input.observe(lambda _: self._on_change(), names='value')

            count_input = widgets.Text(
                value='',
                placeholder='n',
                layout=widgets.Layout(width='70px', min_width='70px', height='28px'),
            )
            count_input.observe(lambda _: self._on_change(), names='value')

            gte_btn = widgets.ToggleButton(
                value=True,
                description='≥',
                layout=widgets.Layout(width='44px', min_width='44px', height='28px'),
                button_style='',
            )

            def _make_gte_obs(btn: widgets.ToggleButton):
                def _obs(change: dict) -> None:
                    btn.description = '≥' if change['new'] else '<'
                    self._on_change()
                return _obs

            gte_btn.observe(_make_gte_obs(gte_btn), names='value')

            rd = {
                'protocol': proto,
                'clamp_mode': mode,
                'label_text': label_text,
                'cb': cb,
                'label': label,
                'group_input': group_input,
                'count_input': count_input,
                'gte_btn': gte_btn,
            }
            self._rows.append(rd)

            hbox = widgets.HBox(
                [cb, label, group_input, count_input, gte_btn],
                layout=widgets.Layout(
                    align_items='center',
                    min_height='36px',
                    margin='1px 0px',
                ),
            )
            row_widgets.append(hbox)

        self._checklist_box.children = row_widgets
        self._on_change()

    # ------------------------------------------------------------------
    # Change handler
    # ------------------------------------------------------------------

    def _on_change(self) -> None:
        """Called on any UI interaction: update gray-out state and summary."""
        if self._block_updates > 0:
            return
        self._block_updates += 1
        try:
            self._update_availability()
            self._update_summary()
        finally:
            self._block_updates -= 1

    def _set_all(self, value: bool) -> None:
        for rd in self._rows:
            if not rd['cb'].disabled:
                rd['cb'].value = value


    # ------------------------------------------------------------------
    # Availability (gray-out) logic
    # ------------------------------------------------------------------

    def _update_availability(self) -> None:
        if self._source_df is None:
            return
        if self._logic_toggle.value == 'AND':
            self._update_availability_and()
        else:
            self._update_availability_or()

    def _update_availability_and(self) -> None:
        """Gray out rows absent from cells passing the current AND selection."""
        checked = [
            rd for rd in self._rows
            if rd['cb'].value and not rd['cb'].disabled
        ]
        if checked:
            mask = [
                self._row_passes_combo_list(row, checked, 'AND')
                for _, row in self._source_df.iterrows()
            ]
            candidate_df = self._source_df[mask]
        else:
            candidate_df = self._source_df

        for rd in self._rows:
            if rd['cb'].value and not rd['cb'].disabled:
                # Currently selected — always keep enabled
                self._set_row_disabled(rd, False)
                continue
            n = sum(
                1 for _, row in candidate_df.iterrows()
                if self._cell_satisfies_row(row, rd)
            )
            self._set_row_disabled(rd, n == 0)

    def _update_availability_or(self) -> None:
        """In OR mode, gray out rows with no cells in the full source DataFrame."""
        for rd in self._rows:
            n = sum(
                1 for _, row in self._source_df.iterrows()
                if self._cell_has_combo(
                    row.get('protocols') or {}, rd['protocol'], rd['clamp_mode']
                )
            )
            self._set_row_disabled(rd, n == 0)

    def _set_row_disabled(self, rd: dict, disabled: bool) -> None:
        if rd['cb'].disabled == disabled:
            return  # no change — skip to avoid spurious observer calls
        rd['cb'].disabled = disabled
        if disabled:
            rd['cb'].value = False
        style = "color:#aaa;" if disabled else ""
        rd['label'].value = (
            f"<span style='font-size:0.88em;{style}'>{rd['label_text']}</span>"
        )

    # ------------------------------------------------------------------
    # Summary label
    # ------------------------------------------------------------------

    def _update_summary(self) -> None:
        if self._source_df is None:
            self._summary_label.value = ''
            return
        logic = self._logic_toggle.value
        groups = self._get_selected_groups()
        if not groups:
            self._summary_label.value = (
                f"No filter active — {len(self._source_df)} cells"
            )
            return
        n_pass = sum(
            1 for _, row in self._source_df.iterrows()
            if self._row_passes_groups(row, groups, logic)
        )
        self._summary_label.value = (
            f"{n_pass} / {len(self._source_df)} cells pass  ({logic})"
        )

    # ------------------------------------------------------------------
    # Apply filter
    # ------------------------------------------------------------------

    def _apply_filter(self) -> None:
        self._output.clear_output()
        with self._output:
            if self._source_df is None:
                print("No DataFrame loaded. Call set_dataframes() first.")
                return
            logic = self._logic_toggle.value
            groups = self._get_selected_groups()
            if not groups:
                self.cells_df = self._source_df.copy()
                print(
                    f"No combinations selected — returning all "
                    f"{len(self.cells_df)} cells."
                )
                return
            mask = [
                self._row_passes_groups(row, groups, logic)
                for _, row in self._source_df.iterrows()
            ]
            self.cells_df = self._source_df[mask].copy()
            print(
                f"Filter applied ({logic}): {len(self.cells_df)} / "
                f"{len(self._source_df)} cells kept."
            )

    # ------------------------------------------------------------------
    # Filter logic
    # ------------------------------------------------------------------

    def _get_selected_groups(self) -> dict[str, list[dict]]:
        """Return active filter groups.

        Rows sharing a non-empty group tag are treated as one OR group.
        Standalone rows (blank tag) each get their own unique key.
        """
        groups: dict[str, list[dict]] = {}
        standalone_idx = 0
        for rd in self._rows:
            if not rd['cb'].value or rd['cb'].disabled:
                continue
            tag = rd['group_input'].value.strip()
            if not tag:
                key = f'__standalone_{standalone_idx}__'
                standalone_idx += 1
                groups[key] = [rd]
            else:
                if tag not in groups:
                    groups[tag] = []
                groups[tag].append(rd)
        return groups

    def _row_passes_groups(
        self,
        row: pd.Series,
        groups: dict[str, list[dict]],
        logic: str,
    ) -> bool:
        """Return True if *row* satisfies the grouped filter."""
        group_results = [
            any(self._cell_satisfies_row(row, rd) for rd in rds)
            for rds in groups.values()
        ]
        if logic == 'AND':
            return all(group_results)
        return any(group_results)

    def _row_passes_combo_list(
        self,
        row: pd.Series,
        row_dicts: list[dict],
        logic: str,
    ) -> bool:
        results = [self._cell_satisfies_row(row, rd) for rd in row_dicts]
        if logic == 'AND':
            return all(results)
        return any(results)

    def _cell_satisfies_row(self, row: pd.Series, rd: dict) -> bool:
        """Return True if *row* satisfies the (protocol, clamp_mode, n_files) criteria."""
        proto_dict = row.get('protocols') or {}
        if not isinstance(proto_dict, dict):
            return False
        files = proto_dict.get(rd['protocol'], [])
        if not isinstance(files, list):
            return False
        count = sum(
            1 for f in files
            if (
                f.get('clamp_mode', 'unknown') if isinstance(f, dict) else 'unknown'
            ) == rd['clamp_mode']
        )
        if count == 0:
            return False
        # Apply optional file-count threshold
        thresh_str = rd['count_input'].value.strip()
        if thresh_str:
            try:
                thresh = int(thresh_str)
                if rd['gte_btn'].value:   # ≥
                    return count >= thresh
                else:                      # <
                    return count < thresh
            except ValueError:
                pass  # non-integer input → ignore threshold
        return True

    @staticmethod
    def _cell_has_combo(
        proto_dict: dict,
        protocol: str,
        clamp_mode: str,
    ) -> bool:
        """Return True if the cell has at least one file for (protocol, clamp_mode)."""
        files = proto_dict.get(protocol, [])
        if not isinstance(files, list):
            return False
        return any(
            (f.get('clamp_mode', 'unknown') if isinstance(f, dict) else 'unknown')
            == clamp_mode
            for f in files
        )
