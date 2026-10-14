"""Tests for mimo_pack.plot.probe — probe layout and channel-spatial plotting.

Uses synthetic_lfp from mimo_pack.testing.data for reproducible test data.
All tests use matplotlib's Agg backend to avoid opening GUI windows.
"""

import pytest
import numpy as np
import xarray as xr
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mimo_pack.testing.data import synthetic_lfp
from mimo_pack.consts import DIM_TIME, DIM_CHAN, COORD_X, COORD_Y, COORD_SHANK, ATTR_UNITS
from mimo_pack.plot.probe import (
    probe_layout,
    probe_layout_subplots,
    channel_values,
    depth_lines,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def lfp_multi_shank():
    """32-channel LFP with 4 shanks."""
    return synthetic_lfp(duration=0.2, fs=1000, n_chans=32, seed=42)


@pytest.fixture
def lfp_single_shank():
    """8-channel LFP on one shank (drop COORD_SHANK)."""
    lfp = synthetic_lfp(duration=0.2, fs=1000, n_chans=8, seed=42)
    return lfp.drop_vars(COORD_SHANK)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after each test."""
    yield
    plt.close('all')


# ===================================================================
# probe_layout
# ===================================================================

class TestProbeLayout:
    def test_basic(self, lfp_multi_shank):
        ax = probe_layout(lfp_multi_shank)
        assert isinstance(ax, plt.Axes)

    def test_no_shank(self, lfp_single_shank):
        ax = probe_layout(lfp_single_shank)
        assert isinstance(ax, plt.Axes)

    def test_no_labels(self, lfp_multi_shank):
        ax = probe_layout(lfp_multi_shank, chan_labels=False)
        assert isinstance(ax, plt.Axes)

    def test_custom_ax(self, lfp_multi_shank):
        fig, ax_in = plt.subplots()
        ax_out = probe_layout(lfp_multi_shank, ax=ax_in)
        assert ax_out is ax_in

    def test_scat_text_args(self, lfp_multi_shank):
        ax = probe_layout(
            lfp_multi_shank,
            scat_args={'s': 50, 'marker': 'o'},
            text_args={'fontsize': 8},
        )
        assert isinstance(ax, plt.Axes)

    def test_type_error(self):
        with pytest.raises(TypeError, match="Expected xr.DataArray"):
            probe_layout(np.random.randn(100, 8))

    def test_missing_coord(self):
        da = xr.DataArray(np.zeros((10, 4)), dims=[DIM_TIME, DIM_CHAN])
        with pytest.raises(ValueError, match="missing required coordinate"):
            probe_layout(da)


# ===================================================================
# probe_layout_subplots
# ===================================================================

class TestProbeLayoutSubplots:
    def test_basic(self, lfp_multi_shank):
        fig, axs, _ = probe_layout_subplots(lfp_multi_shank)
        assert isinstance(fig, plt.Figure)
        n_shanks = len(np.unique(lfp_multi_shank.coords[COORD_SHANK].values))
        assert len(axs) >= n_shanks

    def test_type_error(self):
        with pytest.raises(TypeError, match="Expected xr.DataArray"):
            probe_layout_subplots(np.zeros(10))

    def test_missing_shank(self, lfp_single_shank):
        with pytest.raises(ValueError, match="missing required coordinate"):
            probe_layout_subplots(lfp_single_shank)


# ===================================================================
# channel_values
# ===================================================================

class TestChannelValues:
    def test_basic(self, lfp_multi_shank):
        rms = (lfp_multi_shank ** 2).mean(DIM_TIME) ** 0.5
        ax, scat = channel_values(rms)
        assert isinstance(ax, plt.Axes)
        assert scat is not None

    def test_no_colorbar(self, lfp_multi_shank):
        rms = (lfp_multi_shank ** 2).mean(DIM_TIME) ** 0.5
        ax, scat = channel_values(rms, colorbar=False)
        assert isinstance(ax, plt.Axes)

    def test_units_label(self, lfp_multi_shank):
        rms = (lfp_multi_shank ** 2).mean(DIM_TIME) ** 0.5
        rms.attrs[ATTR_UNITS] = 'µV'
        ax, scat = channel_values(rms)
        assert isinstance(ax, plt.Axes)

    def test_type_error(self):
        with pytest.raises(TypeError, match="Expected xr.DataArray"):
            channel_values(np.zeros(10))

    def test_not_1d(self, lfp_multi_shank):
        with pytest.raises(ValueError, match="1-D"):
            channel_values(lfp_multi_shank)

    def test_single_channel(self):
        lfp = synthetic_lfp(duration=0.1, fs=1000, n_chans=1, seed=42)
        rms = (lfp ** 2).mean(DIM_TIME) ** 0.5
        ax, scat = channel_values(rms)
        assert isinstance(ax, plt.Axes)


# ===================================================================
# depth_lines
# ===================================================================

class TestDepthLines:
    def test_basic(self, lfp_multi_shank):
        ax = depth_lines(lfp_multi_shank)
        assert isinstance(ax, plt.Axes)

    def test_sorted_by_depth(self, lfp_multi_shank):
        """Verify channels are reordered by COORD_Y."""
        y = lfp_multi_shank.coords[COORD_Y].values
        sort_idx = np.argsort(y)
        # depth_lines should sort internally — just verify it runs
        ax = depth_lines(lfp_multi_shank)
        # Right axis tick labels should be sorted y-values
        ax_right = ax.figure.axes[1]  # twinx axis
        tick_labels = [t.get_text() for t in ax_right.get_yticklabels()]
        if tick_labels and tick_labels[0]:
            depth_vals = [float(t) for t in tick_labels]
            assert depth_vals == sorted(depth_vals)

    def test_custom_ax(self, lfp_multi_shank):
        fig, ax_in = plt.subplots()
        ax_out = depth_lines(lfp_multi_shank, ax=ax_in)
        assert ax_out is ax_in

    def test_type_error(self):
        with pytest.raises(TypeError, match="Expected xr.DataArray"):
            depth_lines(np.zeros((100, 8)))

    def test_missing_dim(self):
        da = xr.DataArray(np.zeros((10, 4)), dims=['x', 'y'])
        with pytest.raises(ValueError, match="Dimension"):
            depth_lines(da)

    def test_missing_depth_coord(self):
        da = xr.DataArray(np.zeros((10, 4)), dims=[DIM_TIME, DIM_CHAN])
        with pytest.raises(ValueError, match="Coordinate"):
            depth_lines(da)
