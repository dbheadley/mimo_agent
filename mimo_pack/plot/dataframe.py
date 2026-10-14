"""DataFrame-driven multi-panel figure layout.

Provides :class:`PlotGrid`, a lightweight class that iterates over the rows
of a :class:`~pandas.DataFrame` and draws one subplot per row using a
user-supplied callback.  Inspired by seaborn's ``FacetGrid`` but designed
for DataFrames whose columns contain rich objects (``xr.DataArray``,
fit results, etc.) rather than tidy long-form data.

Two layout modes:

**Grid mode** (default): A single figure with one subplot per
DataFrame row, arranged in an ``nrow x ncol`` grid.

**Facet mode** (any of ``fig``, ``row``, ``col`` is set): Layout
is driven by DataFrame column values.  ``fig`` creates a separate
figure per unique value, ``row`` / ``col`` determine the subplot
row / column within each figure.  The ``.map()`` callback is called
once per DataFrame row; multiple rows that map to the same panel
naturally overlay on the same axes.

Features borrowed from FacetGrid / plotnine ``facet_wrap``:
- ``ncol`` wrapping (auto-compute rows from panel count)
- ``sharex`` / ``sharey`` for aligned axes across panels
- ``height`` / ``aspect`` per-panel sizing
- ``.map()`` chaining for composable plot layers
- ``.refline()`` for adding reference lines to every panel
- ``.set()`` / ``.set_axis_labels()`` / ``.set_titles()`` for bulk styling
- Per-panel error handling consistent with ``apply_to_columns``
- Column-based faceting: ``fig``, ``row``, ``col`` for multi-level layout
"""

import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PlotGrid:
    """Multi-panel figure driven by DataFrame rows.

    Parameters
    ----------
    df : pandas.DataFrame
        One subplot per row (grid mode), or multiple rows per panel
        (facet mode).
    ncol : int or None, optional
        Maximum columns before wrapping to a new row.  ``None``
        (default) puts all panels in one row.  Ignored in facet mode
        when *col* is given.
    fig : str or None, optional
        Column name whose unique values create separate figures.
    row : str or None, optional
        Column name whose unique values determine subplot rows.
    col : str or None, optional
        Column name whose unique values determine subplot columns.
    col_order : list or None, optional
        Explicit order (and set) of column-facet values.  Forces a
        global grid when combined with *fig*.
    row_order : list or None, optional
        Explicit order (and set) of row-facet values.
    height : float, optional
        Height of each panel in inches.  Default is 3.5.
    aspect : float, optional
        Width = ``height * aspect``.  Default is 1.5.
    sharex : bool or str, optional
        Share x-axes across panels.  Default is ``False``.
    sharey : bool or str, optional
        Share y-axes across panels.  Default is ``False``.
    title_fn : callable or None, optional
        ``title_fn(row) -> str`` sets the title for each panel.
        Takes precedence over *title_cols*.
    title_cols : list of str or None, optional
        Column names whose values are joined with *title_sep* to form
        each panel's title.  Ignored when *title_fn* is given.
    title_sep : str, optional
        Separator when building titles from *title_cols*.
        Default is ``'_'``.
    suptitle : str or None, optional
        Figure-level super-title.  In facet mode with *fig*, each
        figure gets the *fig* column value as suptitle instead.
    errors : {'warn', 'raise'}, optional
        How to handle exceptions inside ``.map()`` callbacks.

    Attributes
    ----------
    figure : matplotlib.figure.Figure or None
        The single figure (grid mode or single-fig facet mode).
        ``None`` when facet mode creates multiple figures.
    figures : list of matplotlib.figure.Figure
        All figures (always populated in both modes).
    axes : numpy.ndarray of matplotlib.axes.Axes
        Shape ``(nrow, ncol)``.  Grid mode only.
    axes_flat : numpy.ndarray
        1-D view of *axes*.  Grid mode only.
    facet_mode : bool
        ``True`` when any of *fig*, *row*, *col* is set.
    n_panels : int
        Total number of panels across all figures.
    df : pandas.DataFrame
    """

    def __init__(
        self,
        df,
        *,
        ncol=None,
        fig=None,
        row=None,
        col=None,
        col_order=None,
        row_order=None,
        height=3.5,
        aspect=1.5,
        sharex=False,
        sharey=False,
        title_fn=None,
        title_cols=None,
        title_sep='_',
        suptitle=None,
        errors='warn',
    ):
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected pandas.DataFrame, got {type(df).__name__}."
            )
        if errors not in ('warn', 'raise'):
            raise ValueError(f"errors must be 'warn' or 'raise', got {errors!r}")
        if len(df) == 0:
            raise ValueError("DataFrame is empty -- nothing to plot.")

        self.df = df
        self.errors = errors
        self._title_fn = title_fn
        self._title_cols = title_cols
        self._title_sep = title_sep
        self._failed = set()

        self.facet_mode = any(x is not None for x in (fig, row, col))

        if self.facet_mode:
            self._init_facet(
                df, fig_col=fig, row_col=row, col_col=col,
                col_order=col_order, row_order=row_order,
                height=height, aspect=aspect,
                sharex=sharex, sharey=sharey,
                suptitle=suptitle,
            )
        else:
            self._init_grid(
                df, ncol=ncol,
                height=height, aspect=aspect,
                sharex=sharex, sharey=sharey,
                suptitle=suptitle,
            )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_grid(self, df, *, ncol, height, aspect, sharex, sharey,
                   suptitle):
        """Set up a single figure with one panel per DataFrame row."""
        self.n_panels = len(df)

        if ncol is None:
            ncol = self.n_panels
        ncol = min(ncol, self.n_panels)
        nrow = math.ceil(self.n_panels / ncol)

        self._ncol = ncol
        self._nrow = nrow

        figw = height * aspect * ncol
        figh = height * nrow
        self.figure, axes_2d = plt.subplots(
            nrow, ncol, figsize=(figw, figh),
            sharex=sharex, sharey=sharey, squeeze=False,
        )
        self.axes = axes_2d
        self.axes_flat = axes_2d.flatten()
        self.figures = [self.figure]

        self._all_axes = list(self.axes_flat[:self.n_panels])
        self._facet_panels = None
        self._axes_per_fig = [axes_2d]
        self._nrows_per_fig = [nrow]
        self._ncols_per_fig = [ncol]

        for i in range(self.n_panels, len(self.axes_flat)):
            self.axes_flat[i].set_visible(False)

        self._apply_titles()

        if suptitle is not None:
            self.figure.suptitle(suptitle, fontweight='bold', y=1.02)

    def _init_facet(self, df, *, fig_col, row_col, col_col,
                    col_order, row_order,
                    height, aspect, sharex, sharey, suptitle):
        """Set up one or more figures driven by column values."""
        if fig_col is not None:
            fig_groups = list(df.groupby(fig_col, sort=True))
        else:
            fig_groups = [(None, df)]

        self.figures = []
        self._all_axes = []
        self._facet_panels = []
        self._axes_per_fig = []
        self._nrows_per_fig = []
        self._ncols_per_fig = []

        for fig_val, fig_df in fig_groups:
            if col_col is not None:
                col_vals = (col_order if col_order is not None
                            else sorted(fig_df[col_col].unique()))
            else:
                col_vals = [None]
            if row_col is not None:
                row_vals = (row_order if row_order is not None
                            else sorted(fig_df[row_col].unique()))
            else:
                row_vals = [None]

            nr = len(row_vals)
            nc = len(col_vals)

            figw = height * aspect * nc
            figh = height * nr
            figure, axes_2d = plt.subplots(
                nr, nc, figsize=(figw, figh),
                sharex=sharex, sharey=sharey, squeeze=False,
            )
            self.figures.append(figure)
            self._axes_per_fig.append(axes_2d)
            self._nrows_per_fig.append(nr)
            self._ncols_per_fig.append(nc)

            for ri, rv in enumerate(row_vals):
                for ci, cv in enumerate(col_vals):
                    ax = axes_2d[ri, ci]
                    mask = pd.Series(True, index=fig_df.index)
                    if row_col is not None:
                        mask &= fig_df[row_col] == rv
                    if col_col is not None:
                        mask &= fig_df[col_col] == cv
                    panel_df = fig_df.loc[mask]

                    if panel_df.empty:
                        ax.set_visible(False)
                    else:
                        self._all_axes.append(ax)
                        self._facet_panels.append((panel_df, ax))

            if col_col is not None:
                for ci, cv in enumerate(col_vals):
                    axes_2d[0, ci].set_title(str(cv))

            if fig_val is not None:
                figure.suptitle(str(fig_val), fontweight='bold', y=1.02)
            elif suptitle is not None:
                figure.suptitle(suptitle, fontweight='bold', y=1.02)

        self.n_panels = len(self._facet_panels)

        if len(self.figures) == 1:
            self.figure = self.figures[0]
        else:
            self.figure = None

        self.axes = None
        self.axes_flat = None

        self._apply_titles()

    # ------------------------------------------------------------------
    # Public API -- chaining methods
    # ------------------------------------------------------------------

    def map(self, func, *args, **kwargs):
        """Apply *func* to every panel.

        Parameters
        ----------
        func : callable
            Signature ``func(row, ax, *args, **kwargs)``.
            *row* is the :class:`~pandas.Series` for that panel.
            In facet mode, called once per DataFrame row within each
            panel -- multiple calls overlay on the same axes.
        *args, **kwargs
            Forwarded to *func* after ``row`` and ``ax``.

        Returns
        -------
        self
            For method chaining.
        """
        if self.facet_mode:
            for panel_df, ax in self._facet_panels:
                for idx, r in panel_df.iterrows():
                    if idx in self._failed:
                        continue
                    try:
                        func(r, ax, *args, **kwargs)
                    except Exception as exc:
                        self._failed.add(idx)
                        if self.errors == 'raise':
                            raise
                        warnings.warn(
                            f"PlotGrid.map -- index={idx}: {exc}"
                        )
                        ax.text(
                            0.5, 0.5, f"Error:\n{exc}",
                            transform=ax.transAxes, ha='center', va='center',
                            fontsize=8, color='red', wrap=True,
                        )
        else:
            for i, (idx, r) in enumerate(self.df.iterrows()):
                if idx in self._failed:
                    continue
                ax = self.axes_flat[i]
                try:
                    func(r, ax, *args, **kwargs)
                except Exception as exc:
                    self._failed.add(idx)
                    if self.errors == 'raise':
                        raise
                    warnings.warn(
                        f"PlotGrid.map -- panel {i} (index={idx}): {exc}"
                    )
                    ax.text(
                        0.5, 0.5, f"Error:\n{exc}",
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=8, color='red', wrap=True,
                    )
        return self

    def refline(self, *, x=None, y=None, **kwargs):
        """Add reference line(s) to every visible panel.

        Parameters
        ----------
        x : float or None
            Draw a vertical line at this x-value.
        y : float or None
            Draw a horizontal line at this y-value.
        **kwargs
            Forwarded to ``axvline`` / ``axhline`` (e.g. *color*,
            *ls*, *lw*, *label*).

        Returns
        -------
        self
        """
        for ax in self._all_axes:
            if x is not None:
                ax.axvline(x, **kwargs)
            if y is not None:
                ax.axhline(y, **kwargs)
        return self

    def set(self, **kwargs):
        """Call ``ax.set(**kwargs)`` on every visible panel.

        Useful for ``xlim``, ``ylim``, ``xlabel``, ``ylabel``, etc.

        Returns
        -------
        self
        """
        for ax in self._all_axes:
            ax.set(**kwargs)
        return self

    def set_axis_labels(self, x_label=None, y_label=None, *, clear_inner=True):
        """Set axis labels, optionally clearing inner labels when axes are shared.

        Parameters
        ----------
        x_label : str or None
            Label for x-axes.
        y_label : str or None
            Label for y-axes.
        clear_inner : bool, optional
            When ``True`` (default) and axes are shared, only outer axes
            keep their labels.

        Returns
        -------
        self
        """
        for axes_2d, nr, nc in zip(
            self._axes_per_fig, self._nrows_per_fig, self._ncols_per_fig
        ):
            for ri in range(nr):
                for ci in range(nc):
                    ax = axes_2d[ri, ci]
                    if not ax.get_visible():
                        continue
                    if x_label is not None:
                        if clear_inner and ri < nr - 1:
                            ax.set_xlabel('')
                        else:
                            ax.set_xlabel(x_label)
                    if y_label is not None:
                        if clear_inner and ci > 0:
                            ax.set_ylabel('')
                        else:
                            ax.set_ylabel(y_label)
        return self

    def set_titles(self, template=None, func=None):
        """Override panel titles.

        Parameters
        ----------
        template : str or None
            Format string with access to row fields via keyword
            expansion.  E.g. ``"{subject} / {cell}"``.
        func : callable or None
            ``func(row) -> str``.  Takes precedence over *template*.

        Returns
        -------
        self
        """
        if self.facet_mode:
            for panel_df, ax in self._facet_panels:
                first_row = panel_df.iloc[0]
                if func is not None:
                    ax.set_title(func(first_row))
                elif template is not None:
                    ax.set_title(template.format(
                        **first_row.to_dict(), _index=panel_df.index[0]))
        else:
            for i, (idx, r) in enumerate(self.df.iterrows()):
                ax = self.axes_flat[i]
                if func is not None:
                    ax.set_title(func(r))
                elif template is not None:
                    ax.set_title(template.format(**r.to_dict(), _index=idx))
        return self

    def add_legend(self, *, panel=0, **kwargs):
        """Add a legend to one panel (avoids duplicate legends across panels).

        Parameters
        ----------
        panel : int, optional
            Panel index (0-based) to draw the legend on.  Default 0.
        **kwargs
            Forwarded to ``ax.legend()``.

        Returns
        -------
        self
        """
        kwargs.setdefault('fontsize', 8)
        self._all_axes[panel].legend(**kwargs)
        return self

    def tight_layout(self, **kwargs):
        """Call ``tight_layout()`` on all figures.

        Returns
        -------
        self
        """
        for f in self.figures:
            f.tight_layout(**kwargs)
        return self

    def show(self):
        """Redraw all figure canvases and call ``plt.show()``.

        Returns
        -------
        self
        """
        for f in self.figures:
            f.canvas.draw_idle()
        plt.show()
        return self

    def savefigs(self, directory: str, label: str,
                 fmt: str = 'pdf', **savefig_kwargs) -> 'PlotGrid':
        """Save all figures to *directory* as ``{label}.{fmt}`` files.

        When the grid produces multiple figures (one per unique *fig*
        column value), files are numbered ``{label}_01.{fmt}``,
        ``{label}_02.{fmt}``, …  A single-figure grid uses
        ``{label}.{fmt}`` with no suffix.

        The directory is created if it does not already exist.

        Parameters
        ----------
        directory : str
            Directory to write files into.
        label : str
            Base filename (without extension).
        fmt : str, optional
            File format accepted by ``matplotlib.Figure.savefig``.
            Default is ``'pdf'``.
        **savefig_kwargs
            Additional keyword arguments forwarded to
            ``Figure.savefig`` (e.g. ``dpi``, ``bbox_inches``).

        Returns
        -------
        self
        """
        import os
        savefig_kwargs.setdefault('bbox_inches', 'tight')
        os.makedirs(directory, exist_ok=True)
        figs = self.figures
        for i, fig in enumerate(figs):
            suffix = f'_{i + 1:02d}' if len(figs) > 1 else ''
            path = os.path.join(directory, f'{label}{suffix}.{fmt}')
            fig.savefig(path, **savefig_kwargs)
        return self

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_titles(self):
        """Set initial titles from title_fn or title_cols."""
        if self._title_fn is None and self._title_cols is None:
            return
        if self.facet_mode:
            for panel_df, ax in self._facet_panels:
                first_row = panel_df.iloc[0]
                if self._title_fn is not None:
                    ax.set_title(self._title_fn(first_row), fontweight='bold')
                elif self._title_cols is not None:
                    parts = [str(first_row[c]) for c in self._title_cols]
                    ax.set_title(self._title_sep.join(parts), fontweight='bold')
        else:
            for i, (idx, r) in enumerate(self.df.iterrows()):
                ax = self.axes_flat[i]
                if self._title_fn is not None:
                    ax.set_title(self._title_fn(r), fontweight='bold')
                elif self._title_cols is not None:
                    parts = [str(r[c]) for c in self._title_cols]
                    ax.set_title(self._title_sep.join(parts), fontweight='bold')
