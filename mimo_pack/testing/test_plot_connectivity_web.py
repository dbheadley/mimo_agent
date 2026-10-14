"""Tests for :mod:`mimo_pack.plot.connectivity_web`."""

import json
import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mimo_pack.consts import DIM_TIME, DIM_CHAN, COORD_X, COORD_Y
from mimo_pack.plot.connectivity_web import connectivity_page, connectivity_index_page


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_LAGS = 81


@pytest.fixture
def units():
    return pd.DataFrame({
        'unit_id': [1, 2, 3, 4],
        'x': [0.0, 30.0, 60.0, 90.0],
        'y': [100.0, 400.0, 800.0, 1200.0],
        'cell_type': ['RS', 'FS', 'RS', 'UN'],
        'region': ['M1', 'M1', 'M1', 'M2'],
        'layer': ['L2/3', 'L5', 'L5', 'L6'],
        'shank': [0, 0, 1, 1],
    })


@pytest.fixture
def lags_s():
    return np.linspace(-0.02, 0.02, N_LAGS)


@pytest.fixture
def connections(lags_s):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        'pre_unit': [1, 2],
        'post_unit': [2, 3],
        'sign': [1.0, -1.0],
        'lag_s': [0.002, 0.0015],
        'amplitude': [0.8, -0.4],
        'cc': [rng.poisson(5, N_LAGS).astype(float) for _ in range(2)],
    })


@pytest.fixture
def waveforms():
    t = np.arange(-30, 60)
    wf8 = xr.DataArray(
        np.random.default_rng(1).normal(size=(len(t), 8)),
        dims=(DIM_TIME, DIM_CHAN),
        coords={DIM_TIME: t, DIM_CHAN: np.arange(8),
                COORD_X: (DIM_CHAN, np.tile([0.0, 32.0], 4)),
                COORD_Y: (DIM_CHAN, np.repeat(np.arange(4) * 15.0, 2))})
    full = xr.DataArray(
        np.random.default_rng(2).normal(size=(len(t), 32)),
        dims=(DIM_TIME, DIM_CHAN),
        coords={DIM_TIME: t, DIM_CHAN: np.arange(32),
                COORD_X: (DIM_CHAN, np.tile([0.0, 32.0], 16)),
                COORD_Y: (DIM_CHAN, np.repeat(np.arange(16) * 15.0, 2))})
    return {1: {'waveform': wf8, 'full_waveform': full}, 2: wf8}


def _extract_payload(page):
    m = re.search(r'const DATA = (\{.*\});', page)
    assert m, 'embedded data not found in page'
    return json.loads(m.group(1))


# ---------------------------------------------------------------------------
# connectivity_page
# ---------------------------------------------------------------------------

class TestConnectivityPage:
    def test_returns_standalone_html(self, units, connections, lags_s):
        page = connectivity_page(units, connections, lags_s, title='probeA')
        assert isinstance(page, str)
        assert page.startswith('<!DOCTYPE html>')
        assert 'probeA' in page
        assert '__TITLE__' not in page
        assert '__CMAP_STOPS__' not in page
        assert '/*__DATA__*/null' not in page

    def test_payload_contents(self, units, connections, lags_s, waveforms):
        page = connectivity_page(units, connections, lags_s,
                                 waveforms=waveforms, title='probeA')
        data = _extract_payload(page)
        assert data['probe'] == 'probeA'
        assert len(data['units']) == 4
        assert len(data['conns']) == 2
        assert len(data['lagsMs']) == N_LAGS
        assert data['conns'][0] == pytest.approx(
            data['conns'][0] | {'pre': 1, 'post': 2, 'sign': 1, 'lagMs': 2.0})
        # matrix order covers exactly the connected units, RS/other before FS
        assert sorted(data['order']) == [1, 2, 3]
        assert data['order'].index(2) == len(data['order']) - 1
        # waveforms: unit 1 has site amplitudes from full_waveform, unit 2 not
        assert set(data['wf']) == {'1', '2'}
        assert len(data['wf']['1']['v']) == 8
        assert len(data['wf']['1']['amps']) == 32
        assert 'amps' not in data['wf']['2']
        assert len(data['sites']) == 32

    def test_unit_column_aliases_and_index(self, units, connections, lags_s):
        alt = (units.rename(columns={'region': 'region_name', 'layer': 'laminar_layer'})
               .set_index('unit_id'))
        data = _extract_payload(connectivity_page(alt, connections, lags_s))
        assert data['units'][0]['region'] == 'M1'
        assert data['units'][0]['layer'] == 'L2/3'

    def test_boundaries(self, units, connections, lags_s):
        bounds = pd.DataFrame({
            'shank': [0], 'surface_um': [0.0], 'L1_L23_um': [150.0],
            'L23_L5_um': [500.0], 'L5_L6_um': [900.0], 'floor_um': [1300.0],
            'floor_type': ['WM']})
        data = _extract_payload(
            connectivity_page(units, connections, lags_s, boundaries=bounds))
        assert len(data['bounds']) == 1
        assert data['bounds'][0]['depths'] == [0, 150, 500, 900, 1300]
        assert [lb['name'] for lb in data['bounds'][0]['labels']] == \
            ['L1', 'L2/3', 'L5', 'L6']

    def test_index_link(self, units, connections, lags_s):
        page = connectivity_page(units, connections, lags_s, index_href='index.html')
        assert '<a href="index.html">' in page
        page = connectivity_page(units, connections, lags_s)
        assert '<a href="index.html">' not in page

    def test_connections_outside_units_dropped(self, units, connections, lags_s):
        extra = pd.concat([connections, pd.DataFrame({
            'pre_unit': [99], 'post_unit': [1], 'sign': [1.0], 'lag_s': [0.002],
            'amplitude': [0.5], 'cc': [np.ones(N_LAGS)]})], ignore_index=True)
        data = _extract_payload(connectivity_page(units, extra, lags_s))
        assert len(data['conns']) == 2

    def test_invalid_inputs_raise(self, units, connections, lags_s):
        with pytest.raises(TypeError):
            connectivity_page('not a frame', connections, lags_s)
        with pytest.raises(ValueError):
            connectivity_page(units.drop(columns=['x']), connections, lags_s)
        with pytest.raises(ValueError):
            connectivity_page(units, connections.drop(columns=['cc']), lags_s)
        with pytest.raises(ValueError):
            connectivity_page(units, connections, lags_s[:10])

    def test_waveform_missing_coords_raises(self, units, connections, lags_s):
        bad = xr.DataArray(np.zeros((90, 8)), dims=(DIM_TIME, DIM_CHAN))
        with pytest.raises(ValueError):
            connectivity_page(units, connections, lags_s, waveforms={1: bad})


# ---------------------------------------------------------------------------
# connectivity_index_page
# ---------------------------------------------------------------------------

class TestIndexPage:
    def test_basic(self):
        page = connectivity_index_page([
            {'href': 'a.html', 'title': 'probeA', 'units': 10},
            {'href': 'b.html', 'title': 'probeB', 'units': 20},
        ])
        assert '<a href="a.html">probeA</a>' in page
        assert '<th>units</th>' in page
        assert '<td>20</td>' in page

    def test_requires_href_and_title(self):
        with pytest.raises(ValueError):
            connectivity_index_page([{'title': 'probeA'}])
