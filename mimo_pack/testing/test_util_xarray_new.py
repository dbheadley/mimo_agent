"""Tests for mimo_pack.util.xarray functions:
select_by_coord, save_dataarray_df, load_dataarray_df.
"""

import os
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mimo_pack.consts import DIM_TIME, DIM_CHAN, DIM_EPOCH, COORD_X, COORD_Y, ATTR_FS
from mimo_pack.testing.data import synthetic_lfp
from mimo_pack.util.xarray import (
    select_by_coord,
    save_dataarray_df,
    load_dataarray_df,
    explode_xarray,
)


# =====================================================================
# select_by_coord
# =====================================================================

class TestSelectByCoord:
    @staticmethod
    def _make_da():
        """DataArray with 4 channels, each having a ch_units coordinate."""
        return xr.DataArray(
            np.random.RandomState(0).randn(100, 4),
            dims=(DIM_TIME, DIM_CHAN),
            coords={
                DIM_TIME: np.linspace(0, 1, 100),
                'ch_units': (DIM_CHAN, ['mV', 'pA', 'mV', 'pA']),
                'ch_names': (DIM_CHAN, ['Vm', 'Im', 'Vm2', 'Im2']),
            },
            attrs={ATTR_FS: 100.0},
        )

    def test_single_value(self):
        da = self._make_da()
        result = select_by_coord(da, 'ch_units', 'pA')
        assert result.sizes[DIM_CHAN] == 2
        assert list(result['ch_units'].values) == ['pA', 'pA']

    def test_multiple_values(self):
        da = self._make_da()
        result = select_by_coord(da, 'ch_names', ['Vm', 'Im2'])
        assert result.sizes[DIM_CHAN] == 2

    def test_preserves_time(self):
        da = self._make_da()
        result = select_by_coord(da, 'ch_units', 'mV')
        assert result.sizes[DIM_TIME] == 100

    def test_preserves_attrs(self):
        da = self._make_da()
        result = select_by_coord(da, 'ch_units', 'mV')
        assert ATTR_FS in result.attrs

    def test_preserves_other_coords(self):
        da = self._make_da()
        result = select_by_coord(da, 'ch_units', 'pA')
        assert 'ch_names' in result.coords
        assert list(result['ch_names'].values) == ['Im', 'Im2']

    def test_no_match_raises(self):
        da = self._make_da()
        with pytest.raises(ValueError, match="No indices"):
            select_by_coord(da, 'ch_units', 'nA')

    def test_no_match_shows_available(self):
        da = self._make_da()
        with pytest.raises(ValueError, match="Available values"):
            select_by_coord(da, 'ch_units', 'nA')

    def test_missing_coord_raises(self):
        da = self._make_da()
        with pytest.raises(ValueError):
            select_by_coord(da, 'nonexistent', 'x')

    def test_explicit_dim(self):
        da = self._make_da()
        result = select_by_coord(da, 'ch_units', 'mV', dim=DIM_CHAN)
        assert result.sizes[DIM_CHAN] == 2

    def test_works_with_synthetic_lfp(self):
        lfp = synthetic_lfp(duration=0.5, fs=1000, n_chans=8, seed=0)
        # select_by_coord should work on spatial coords too
        shank_vals = lfp.coords[COORD_X].values
        target = shank_vals[0]
        result = select_by_coord(lfp, COORD_X, target)
        assert result.sizes[DIM_CHAN] > 0


# =====================================================================
# save_dataarray_df / load_dataarray_df
# =====================================================================

class TestSaveLoadDataarrayDf:
    @staticmethod
    def _make_df():
        """Build a small DataFrame with DataArray-valued column."""
        das = []
        for i in range(3):
            da = xr.DataArray(
                np.random.RandomState(i).randn(50, 2),
                dims=(DIM_TIME, DIM_CHAN),
                coords={DIM_TIME: np.linspace(0, 0.5, 50)},
                attrs={ATTR_FS: 100.0, 'description': f'test_{i}'},
            )
            das.append(da)
        return pd.DataFrame({
            'subject': ['m1', 'm1', 'm2'],
            'session': ['s1', 's2', 's1'],
            'rec': das,
        })

    def test_save_creates_files(self, tmp_path):
        df = self._make_df()
        paths = save_dataarray_df(df, 'rec', str(tmp_path),
                                  name_cols=['subject', 'session'],
                                  label='lfp')
        assert len(paths) == 3
        for p in paths:
            assert os.path.isfile(p)
            assert p.endswith('.nc')

    def test_filenames_correct(self, tmp_path):
        df = self._make_df()
        paths = save_dataarray_df(df, 'rec', str(tmp_path),
                                  name_cols=['subject', 'session'],
                                  label='lfp')
        basenames = sorted(os.path.basename(p) for p in paths)
        assert 'm1_s1_lfp.nc' in basenames
        assert 'm1_s2_lfp.nc' in basenames
        assert 'm2_s1_lfp.nc' in basenames

    def test_roundtrip(self, tmp_path):
        df = self._make_df()
        save_dataarray_df(df, 'rec', str(tmp_path),
                          name_cols=['subject', 'session'],
                          label='lfp')
        loaded = load_dataarray_df(str(tmp_path),
                                   name_fields=['subject', 'session', 'label'],
                                   da_col='rec')
        assert len(loaded) == 3
        assert 'rec' in loaded.columns
        assert 'subject' in loaded.columns
        assert 'session' in loaded.columns

    def test_roundtrip_data_integrity(self, tmp_path):
        df = self._make_df()
        save_dataarray_df(df, 'rec', str(tmp_path),
                          name_cols=['subject', 'session'],
                          label='lfp')
        loaded = load_dataarray_df(str(tmp_path),
                                   name_fields=['subject', 'session', 'label'],
                                   da_col='rec')
        # Check that the data round trips correctly
        for _, row in loaded.iterrows():
            assert isinstance(row['rec'], xr.DataArray)
            assert row['rec'].sizes[DIM_TIME] == 50

    def test_creates_dir(self, tmp_path):
        subdir = os.path.join(str(tmp_path), 'deep', 'nested')
        df = self._make_df()
        paths = save_dataarray_df(df, 'rec', subdir,
                                  name_cols=['subject'], label='x')
        assert os.path.isdir(subdir)
        assert len(paths) == 3

    def test_load_empty_dir(self, tmp_path):
        loaded = load_dataarray_df(str(tmp_path))
        assert len(loaded) == 0

    def test_load_no_name_fields(self, tmp_path):
        df = self._make_df()
        save_dataarray_df(df, 'rec', str(tmp_path),
                          name_cols=['subject', 'session'],
                          label='lfp')
        loaded = load_dataarray_df(str(tmp_path), da_col='rec')
        assert len(loaded) == 3
        assert 'rec' in loaded.columns



# ---------------------------------------------------------------------------
# explode_xarray
# ---------------------------------------------------------------------------

from mimo_pack.util.xarray import concat_by_attr


class TestExplodeXarray:
    """Tests for :func:`mimo_pack.util.xarray.explode_xarray`."""

    @staticmethod
    def _make_df():
        """Build a 2-row DataFrame with concatenated DataArrays.

        Each combined DataArray has dims (time, cond) with cond size 2.
        """
        da1 = xr.DataArray([1, 2], dims='time', attrs={'cond': 'A'})
        da2 = xr.DataArray([3, 4], dims='time', attrs={'cond': 'B'})
        combined1 = concat_by_attr([da1, da2], attr='cond', dim='cond')

        da3 = xr.DataArray([10, 20], dims='time', attrs={'cond': 'X'})
        da4 = xr.DataArray([30, 40], dims='time', attrs={'cond': 'Y'})
        combined2 = concat_by_attr([da3, da4], attr='cond', dim='cond')

        return pd.DataFrame({
            'data': [combined1, combined2],
            'subject': ['m1', 'm2'],
        })

    def test_basic(self):
        df = self._make_df()
        out = explode_xarray(df, col='data', dim='cond')
        # 2 rows x 2 cond values = 4 output rows
        assert len(out) == 4
        assert 'data' in out.columns
        assert 'subject' in out.columns

    def test_coord_col_values(self):
        df = self._make_df()
        out = explode_xarray(df, col='data', dim='cond')
        assert 'cond' in out.columns
        assert list(out['cond']) == ['A', 'B', 'X', 'Y']

    def test_scalar_columns_duplicated(self):
        df = self._make_df()
        out = explode_xarray(df, col='data', dim='cond')
        assert list(out['subject']) == ['m1', 'm1', 'm2', 'm2']

    def test_keep_coords_true(self):
        """Extra coordinates along dim should become columns."""
        da = xr.DataArray(
            [[1, 2], [3, 4], [5, 6]], dims=('trial', 'time'),
            coords={'trial': [0, 1, 2], 'stim': ('trial', ['A', 'B', 'A'])},
        )
        df = pd.DataFrame({'rec': [da]})
        out = explode_xarray(df, col='rec', dim='trial', keep_coords=True)
        assert 'stim' in out.columns
        assert list(out['stim']) == ['A', 'B', 'A']

    def test_keep_coords_false(self):
        da = xr.DataArray(
            [[1, 2], [3, 4], [5, 6]], dims=('trial', 'time'),
            coords={'trial': [0, 1, 2], 'stim': ('trial', ['A', 'B', 'A'])},
        )
        df = pd.DataFrame({'rec': [da]})
        out = explode_xarray(df, col='rec', dim='trial', keep_coords=False)
        assert 'stim' not in out.columns

    def test_custom_coord_col(self):
        df = self._make_df()
        out = explode_xarray(df, col='data', dim='cond', coord_col='condition')
        assert 'condition' in out.columns
        assert 'cond' not in out.columns

    def test_orig_index_preserved(self):
        df = self._make_df()
        out = explode_xarray(df, col='data', dim='cond')
        assert '_orig_index' in out.columns
        assert list(out['_orig_index']) == [0, 0, 1, 1]

    def test_preserves_other_dims(self):
        """Sliced DataArrays should keep non-split dimensions."""
        df = self._make_df()
        out = explode_xarray(df, col='data', dim='cond')
        for _, row in out.iterrows():
            assert row['data'].dims == ('time',)
            assert row['data'].shape == (2,)

    def test_missing_dim_error(self):
        da = xr.DataArray([1, 2], dims='time')
        df = pd.DataFrame({'rec': [da]})
        with pytest.raises(ValueError, match="does not have dimension"):
            explode_xarray(df, col='rec', dim='cond')

    def test_type_error(self):
        df = pd.DataFrame({'rec': [42]})
        with pytest.raises(TypeError, match="expected xr.DataArray"):
            explode_xarray(df, col='rec', dim='cond')

    def test_multi_col(self):
        """Multiple xarray columns should be split in sync."""
        da_a = xr.DataArray([[1, 2], [3, 4]], dims=('cond', 'time'))
        da_b = xr.DataArray([[10, 20], [30, 40]], dims=('cond', 'time'))
        df = pd.DataFrame({'a': [da_a], 'b': [da_b]})
        out = explode_xarray(df, col=['a', 'b'], dim='cond')
        assert len(out) == 2
        assert list(out.iloc[0]['a'].values) == [1, 2]
        assert list(out.iloc[0]['b'].values) == [10, 20]

    def test_variable_dim_sizes(self):
        """Rows with different sizes along dim are handled independently."""
        da1 = xr.DataArray([[1], [2]], dims=('cond', 'time'))
        da2 = xr.DataArray([[10], [20], [30]], dims=('cond', 'time'))
        df = pd.DataFrame({'rec': [da1, da2], 'id': ['a', 'b']})
        out = explode_xarray(df, col='rec', dim='cond')
        assert len(out) == 5
        assert list(out['id']) == ['a', 'a', 'b', 'b', 'b']
