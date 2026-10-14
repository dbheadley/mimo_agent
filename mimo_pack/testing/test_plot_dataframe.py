"""Tests for :class:`mimo_pack.plot.dataframe.PlotGrid`."""

import warnings

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mimo_pack.plot.dataframe import PlotGrid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Simple 4-row DataFrame with plottable data."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        'x': [rng.normal(size=20) for _ in range(4)],
        'y': [rng.normal(size=20) for _ in range(4)],
        'subject': ['mouse1', 'mouse1', 'mouse2', 'mouse2'],
        'cell': ['c1', 'c2', 'c1', 'c2'],
    })


@pytest.fixture(autouse=True)
def _close_figs():
    """Close all figures after each test."""
    yield
    plt.close('all')


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_basic(self, sample_df):
        g = PlotGrid(sample_df)
        assert g.n_panels == 4
        assert g.axes.shape == (1, 4)
        assert g.figure is not None

    def test_ncol_wrapping(self, sample_df):
        g = PlotGrid(sample_df, ncol=2)
        assert g.axes.shape == (2, 2)

    def test_ncol_wrapping_uneven(self, sample_df):
        g = PlotGrid(sample_df.iloc[:3], ncol=2)
        assert g.axes.shape == (2, 2)
        # Last axis should be hidden
        assert not g.axes_flat[3].get_visible()

    def test_type_error(self):
        with pytest.raises(TypeError, match="Expected pandas.DataFrame"):
            PlotGrid([1, 2, 3])

    def test_empty_error(self):
        with pytest.raises(ValueError, match="empty"):
            PlotGrid(pd.DataFrame({'a': []}))

    def test_invalid_errors(self, sample_df):
        with pytest.raises(ValueError, match="errors must be"):
            PlotGrid(sample_df, errors='skip')

    def test_height_aspect(self, sample_df):
        g = PlotGrid(sample_df.iloc[:2], height=4.0, aspect=2.0, ncol=2)
        fig_w, fig_h = g.figure.get_size_inches()
        assert abs(fig_w - 16.0) < 0.1   # 4*2*2 = 16
        assert abs(fig_h - 4.0) < 0.1    # 4*1 = 4

    def test_suptitle(self, sample_df):
        g = PlotGrid(sample_df, suptitle='My Title')
        texts = [t.get_text() for t in g.figure.texts]
        assert 'My Title' in texts


# ---------------------------------------------------------------------------
# Titles
# ---------------------------------------------------------------------------

class TestTitles:
    def test_title_cols(self, sample_df):
        g = PlotGrid(sample_df, title_cols=['subject', 'cell'])
        titles = [g.axes_flat[i].get_title() for i in range(4)]
        assert titles[0] == 'mouse1_c1'
        assert titles[3] == 'mouse2_c2'

    def test_title_fn(self, sample_df):
        g = PlotGrid(sample_df, title_fn=lambda r: f"S={r['subject']}")
        assert g.axes_flat[0].get_title() == 'S=mouse1'

    def test_title_fn_overrides_cols(self, sample_df):
        g = PlotGrid(
            sample_df,
            title_cols=['subject'],
            title_fn=lambda r: 'override',
        )
        assert g.axes_flat[0].get_title() == 'override'

    def test_title_sep(self, sample_df):
        g = PlotGrid(sample_df, title_cols=['subject', 'cell'], title_sep=' / ')
        assert g.axes_flat[0].get_title() == 'mouse1 / c1'

    def test_set_titles_template(self, sample_df):
        g = PlotGrid(sample_df)
        g.set_titles(template="{subject} — {cell}")
        assert g.axes_flat[0].get_title() == 'mouse1 — c1'

    def test_set_titles_func(self, sample_df):
        g = PlotGrid(sample_df)
        g.set_titles(func=lambda r: r['cell'].upper())
        assert g.axes_flat[0].get_title() == 'C1'


# ---------------------------------------------------------------------------
# .map()
# ---------------------------------------------------------------------------

class TestMap:
    def test_basic_map(self, sample_df):
        g = PlotGrid(sample_df, ncol=2)
        g.map(lambda row, ax: ax.plot(row['x'], row['y'], 'o'))
        # Verify each panel has a plot
        for i in range(4):
            assert len(g.axes_flat[i].lines) > 0

    def test_chaining(self, sample_df):
        g = PlotGrid(sample_df, ncol=2)
        result = g.map(lambda row, ax: ax.plot(row['x']))
        assert result is g  # returns self

    def test_errors_warn(self, sample_df):
        def bad_fn(row, ax):
            if row['cell'] == 'c2':
                raise ValueError("test error")
            ax.plot(row['x'])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            g = PlotGrid(sample_df, errors='warn')
            g.map(bad_fn)

        warn_msgs = [str(x.message) for x in w]
        assert any('test error' in m for m in warn_msgs)
        # Panels 0 and 2 should have plots, 1 and 3 should not
        assert len(g.axes_flat[0].lines) > 0
        assert len(g.axes_flat[1].lines) == 0

    def test_errors_raise(self, sample_df):
        def bad_fn(row, ax):
            raise RuntimeError("boom")

        g = PlotGrid(sample_df, errors='raise')
        with pytest.raises(RuntimeError, match="boom"):
            g.map(bad_fn)

    def test_failed_panel_skipped_in_subsequent_map(self, sample_df):
        """A panel that failed in one .map() call is skipped in later calls."""
        call_count = [0]

        def counting_fn(row, ax):
            call_count[0] += 1

        def failing_fn(row, ax):
            if row.name == sample_df.index[0]:
                raise ValueError("fail first")

        g = PlotGrid(sample_df, errors='warn')
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            g.map(failing_fn)

        g.map(counting_fn)
        assert call_count[0] == 3  # first panel skipped

    def test_map_extra_args(self, sample_df):
        captured = []
        g = PlotGrid(sample_df.iloc[:1])
        g.map(lambda row, ax, c: captured.append(c), 'red')
        assert captured == ['red']


# ---------------------------------------------------------------------------
# .refline()
# ---------------------------------------------------------------------------

class TestRefline:
    def test_vertical(self, sample_df):
        g = PlotGrid(sample_df.iloc[:2])
        g.refline(x=0, color='red', ls='--')
        for i in range(2):
            lines = g.axes_flat[i].lines
            assert len(lines) >= 1

    def test_horizontal(self, sample_df):
        g = PlotGrid(sample_df.iloc[:2])
        g.refline(y=1.5, color='blue')
        for i in range(2):
            assert len(g.axes_flat[i].lines) >= 1

    def test_both(self, sample_df):
        g = PlotGrid(sample_df.iloc[:1])
        g.refline(x=0, y=0)
        assert len(g.axes_flat[0].lines) == 2

    def test_chaining(self, sample_df):
        g = PlotGrid(sample_df.iloc[:1])
        result = g.refline(x=0)
        assert result is g


# ---------------------------------------------------------------------------
# .set() / .set_axis_labels()
# ---------------------------------------------------------------------------

class TestSetMethods:
    def test_set(self, sample_df):
        g = PlotGrid(sample_df.iloc[:2])
        g.set(xlim=(-1, 1), ylim=(0, 5))
        for i in range(2):
            assert g.axes_flat[i].get_xlim() == (-1, 1)
            assert g.axes_flat[i].get_ylim() == (0, 5)

    def test_set_axis_labels(self, sample_df):
        g = PlotGrid(sample_df.iloc[:2], ncol=2)
        g.set_axis_labels('Time (s)', 'Voltage (mV)')
        assert g.axes_flat[0].get_xlabel() == 'Time (s)'
        assert g.axes_flat[0].get_ylabel() == 'Voltage (mV)'

    def test_clear_inner_ylabel(self, sample_df):
        g = PlotGrid(sample_df, ncol=2)
        g.set_axis_labels('X', 'Y', clear_inner=True)
        # Second column should have empty y-label
        assert g.axes_flat[1].get_ylabel() == ''
        assert g.axes_flat[0].get_ylabel() == 'Y'

    def test_clear_inner_xlabel(self, sample_df):
        g = PlotGrid(sample_df, ncol=2)
        g.set_axis_labels('X', 'Y', clear_inner=True)
        # Top row should have empty x-label (panels exist below them)
        assert g.axes_flat[0].get_xlabel() == ''
        # Bottom row keeps x-label
        assert g.axes_flat[2].get_xlabel() == 'X'


# ---------------------------------------------------------------------------
# .add_legend() / .tight_layout() / .show()
# ---------------------------------------------------------------------------

class TestMisc:
    def test_add_legend(self, sample_df):
        g = PlotGrid(sample_df.iloc[:2])
        g.map(lambda row, ax: ax.plot([0, 1], label='test'))
        g.add_legend(panel=0)
        assert g.axes_flat[0].get_legend() is not None

    def test_tight_layout(self, sample_df):
        g = PlotGrid(sample_df.iloc[:2])
        result = g.tight_layout()
        assert result is g

    def test_show(self, sample_df):
        g = PlotGrid(sample_df.iloc[:1])
        result = g.show()
        assert result is g


# ---------------------------------------------------------------------------
# Integrated usage
# ---------------------------------------------------------------------------

class TestIntegrated:
    def test_full_workflow(self, sample_df):
        """Simulates typical notebook usage."""
        g = (
            PlotGrid(
                sample_df,
                ncol=2,
                height=3,
                aspect=1.5,
                sharex=True,
                sharey=True,
                title_cols=['subject', 'cell'],
                suptitle='Test Grid',
                errors='warn',
            )
            .map(lambda row, ax: ax.plot(row['x'], row['y'], 'o'))
            .refline(x=0, color='steelblue', ls='--', label='origin')
            .set_axis_labels('X value', 'Y value')
            .add_legend()
            .tight_layout()
        )
        assert g.n_panels == 4
        assert g.axes.shape == (2, 2)


# ---------------------------------------------------------------------------
# Facet mode
# ---------------------------------------------------------------------------

@pytest.fixture
def facet_df():
    """DataFrame with columns suitable for faceting."""
    rng = np.random.default_rng(99)
    rows = []
    for grp in ['A', 'B']:
        for cond in [1, 2, 3]:
            rows.append({
                'group': grp,
                'cond': cond,
                'x': rng.normal(size=10),
                'y': rng.normal(size=10),
                'label': f'{grp}_{cond}',
            })
    return pd.DataFrame(rows)


class TestFacetMode:
    def test_col_only(self, facet_df):
        g = PlotGrid(facet_df, col='cond')
        assert g.facet_mode is True
        assert len(g.figures) == 1
        assert g.figure is not None
        # 3 unique cond values → 3 panels
        assert g.n_panels == 3

    def test_fig_only(self, facet_df):
        g = PlotGrid(facet_df, fig='group')
        assert g.facet_mode is True
        assert len(g.figures) == 2
        assert g.figure is None  # multiple figures

    def test_fig_and_col(self, facet_df):
        g = PlotGrid(facet_df, fig='group', col='cond')
        assert len(g.figures) == 2
        # Each figure: 1 row × 3 cols = 3 panels
        assert g.n_panels == 6

    def test_row_and_col(self, facet_df):
        g = PlotGrid(facet_df, row='group', col='cond')
        assert len(g.figures) == 1
        assert g.figure is not None
        # 2 rows × 3 cols = 6 panels
        assert g.n_panels == 6

    def test_fig_row_col(self):
        """All three faceting levels."""
        rows = []
        for fig_val in ['F1', 'F2']:
            for r_val in ['R1', 'R2']:
                for c_val in ['C1', 'C2']:
                    rows.append({
                        'f': fig_val, 'r': r_val, 'c': c_val,
                        'val': 1.0,
                    })
        df = pd.DataFrame(rows)
        g = PlotGrid(df, fig='f', row='r', col='c')
        assert len(g.figures) == 2
        assert g.n_panels == 8

    def test_variable_col_counts(self):
        """Different fig groups have different col counts (per-figure sizing)."""
        df = pd.DataFrame({
            'grp': ['A', 'A', 'B', 'B', 'B'],
            'cond': [1, 2, 1, 2, 3],
            'val': range(5),
        })
        g = PlotGrid(df, fig='grp', col='cond')
        assert len(g.figures) == 2
        # Group A: 2 cols, Group B: 3 cols
        assert g._ncols_per_fig[0] == 2
        assert g._ncols_per_fig[1] == 3

    def test_col_order_explicit(self):
        """col_order enforces global grid with empty panels for missing combos."""
        df = pd.DataFrame({
            'grp': ['A', 'A', 'B'],
            'cond': [1, 2, 1],
            'val': range(3),
        })
        g = PlotGrid(df, fig='grp', col='cond', col_order=[1, 2, 3])
        # Both figures have 3 columns
        assert g._ncols_per_fig[0] == 3
        assert g._ncols_per_fig[1] == 3
        # Some panels should be hidden (cond=3 for A, cond=2,3 for B)

    def test_row_order_explicit(self):
        df = pd.DataFrame({
            'r': ['R1', 'R1', 'R2'],
            'c': ['C1', 'C2', 'C1'],
            'val': range(3),
        })
        g = PlotGrid(df, row='r', col='c', row_order=['R2', 'R1'])
        assert g._nrows_per_fig[0] == 2

    def test_map_overlay(self, facet_df):
        """Multiple rows per panel → callback called multiple times per panel."""
        call_count = {}
        def counter(row, ax):
            ax_id = id(ax)
            call_count[ax_id] = call_count.get(ax_id, 0) + 1
        g = PlotGrid(facet_df, col='cond')
        g.map(counter)
        # 6 rows, 3 panels → 2 calls per panel
        for cnt in call_count.values():
            assert cnt == 2

    def test_auto_titles_col(self, facet_df):
        """Col values should appear as panel titles."""
        g = PlotGrid(facet_df, col='cond')
        titles = [ax.get_title() for ax in g._all_axes]
        assert titles == ['1', '2', '3']

    def test_auto_titles_fig_suptitle(self, facet_df):
        """Fig column values should become figure suptitles."""
        g = PlotGrid(facet_df, fig='group', col='cond')
        suptitles = [fig._suptitle.get_text() for fig in g.figures]
        assert suptitles == ['A', 'B']

    def test_title_fn_override(self, facet_df):
        """title_fn should override auto-titles on panels."""
        g = PlotGrid(facet_df, col='cond',
                     title_fn=lambda r: f"COND-{r['cond']}")
        titles = [ax.get_title() for ax in g._all_axes]
        assert all(t.startswith('COND-') for t in titles)

    def test_refline(self, facet_df):
        g = PlotGrid(facet_df, fig='group', col='cond')
        g.refline(x=0, color='k')
        for ax in g._all_axes:
            assert len(ax.lines) >= 1

    def test_set_axis_labels_facet(self, facet_df):
        g = PlotGrid(facet_df, fig='group', col='cond')
        g.set_axis_labels('X', 'Y', clear_inner=True)
        # First panel of each fig should have y-label
        for axes_2d in g._axes_per_fig:
            assert axes_2d[0, 0].get_ylabel() == 'Y'
            if axes_2d.shape[1] > 1:
                assert axes_2d[0, 1].get_ylabel() == ''

    def test_error_handling_warn(self, facet_df):
        def bad_func(row, ax):
            if row['cond'] == 2:
                raise RuntimeError("boom")
            ax.plot(row['x'], row['y'])
        g = PlotGrid(facet_df, col='cond', errors='warn')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            g.map(bad_func)
        assert any('boom' in str(wi.message) for wi in w)

    def test_error_handling_raise(self, facet_df):
        def bad_func(row, ax):
            raise RuntimeError("boom")
        g = PlotGrid(facet_df, col='cond', errors='raise')
        with pytest.raises(RuntimeError, match='boom'):
            g.map(bad_func)

    def test_tight_layout_multi_fig(self, facet_df):
        g = PlotGrid(facet_df, fig='group', col='cond')
        result = g.tight_layout()
        assert result is g

    def test_empty_panels_hidden(self):
        """Missing row×col combos should have hidden axes."""
        df = pd.DataFrame({
            'r': ['R1', 'R2'],
            'c': ['C1', 'C1'],
            'val': [1, 2],
        })
        g = PlotGrid(df, row='r', col='c', col_order=['C1', 'C2'])
        # C2 has no data → those axes should be hidden
        for axes_2d in g._axes_per_fig:
            assert not axes_2d[0, 1].get_visible()
            assert not axes_2d[1, 1].get_visible()
