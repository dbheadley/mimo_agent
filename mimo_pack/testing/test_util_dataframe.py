"""Tests for mimo_pack.util.dataframe: apply_to_columns, pipe_columns."""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from mimo_pack.consts import DIM_TIME, DIM_CHAN, ATTR_FS
from mimo_pack.util.dataframe import apply_to_columns, pipe_columns


# =====================================================================
# apply_to_columns — single input / single output
# =====================================================================

class TestApplyToColumnsSingleIO:
    @staticmethod
    def _make_df():
        return pd.DataFrame({'x': [1, 2, 3], 'label': ['a', 'b', 'c']})

    def test_basic_transform(self):
        df = self._make_df()
        result = apply_to_columns(df, 'x', lambda v: v * 10, result_cols='ten_x')
        assert list(result['ten_x']) == [10, 20, 30]

    def test_overwrite_default(self):
        df = self._make_df()
        result = apply_to_columns(df, 'x', lambda v: v + 1)
        assert list(result['x']) == [2, 3, 4]

    def test_preserves_other_columns(self):
        df = self._make_df()
        result = apply_to_columns(df, 'x', lambda v: v, result_cols='copy')
        assert list(result['label']) == ['a', 'b', 'c']

    def test_returns_copy(self):
        df = self._make_df()
        result = apply_to_columns(df, 'x', lambda v: v, result_cols='copy')
        assert result is not df

    def test_extra_args_kwargs(self):
        df = self._make_df()

        def scale(v, factor, offset=0):
            return v * factor + offset

        result = apply_to_columns(df, 'x', scale, 2, offset=5,
                                  result_cols='scaled')
        assert list(result['scaled']) == [7, 9, 11]


# =====================================================================
# apply_to_columns — multi input / multi output
# =====================================================================

class TestApplyToColumnsMultiIO:
    def test_multi_input(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [10, 20]})
        result = apply_to_columns(df, ['a', 'b'], lambda a, b: a + b,
                                  result_cols='sum')
        assert list(result['sum']) == [11, 22]

    def test_multi_output(self):
        df = pd.DataFrame({'x': [3, 6]})
        result = apply_to_columns(df, 'x',
                                  lambda v: (v * 2, v * 3),
                                  result_cols=['doubled', 'tripled'])
        assert list(result['doubled']) == [6, 12]
        assert list(result['tripled']) == [9, 18]

    def test_multi_input_multi_output(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [10, 20]})
        result = apply_to_columns(df, ['a', 'b'],
                                  lambda a, b: (a + b, a * b),
                                  result_cols=['sum', 'prod'])
        assert list(result['sum']) == [11, 22]
        assert list(result['prod']) == [10, 40]

    def test_wrong_tuple_length_raises(self):
        df = pd.DataFrame({'x': [1]})
        with pytest.raises(ValueError, match="tuple of length 2"):
            apply_to_columns(df, 'x', lambda v: (v,),
                             result_cols=['a', 'b'], errors='raise')


# =====================================================================
# apply_to_columns — error handling
# =====================================================================

class TestApplyToColumnsErrors:
    @staticmethod
    def _make_df():
        return pd.DataFrame({'x': [1, 2, 3]})

    def test_errors_warn(self):
        df = self._make_df()
        with pytest.warns(UserWarning, match="division by zero"):
            result = apply_to_columns(df, 'x',
                                      lambda v: 1 / 0,
                                      result_cols='out', errors='warn')
        assert all(v is None for v in result['out'])

    def test_errors_raise(self):
        df = self._make_df()
        with pytest.raises(ZeroDivisionError):
            apply_to_columns(df, 'x', lambda v: 1 / 0, errors='raise')

    def test_errors_skip(self):
        df = self._make_df()
        result = apply_to_columns(df, 'x', lambda v: 1 / 0,
                                  result_cols='out', errors='skip')
        assert all(v is None for v in result['out'])

    def test_invalid_errors_raises(self):
        df = self._make_df()
        with pytest.raises(ValueError, match="errors must be"):
            apply_to_columns(df, 'x', lambda v: v, errors='bad')

    def test_multi_output_error_sets_all_none(self):
        df = pd.DataFrame({'x': [1, 2]})

        def fail_second(v):
            if v == 2:
                raise RuntimeError("fail")
            return (v, v * 10)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = apply_to_columns(df, 'x', fail_second,
                                      result_cols=['a', 'b'], errors='warn')
        # First row succeeded
        assert result.iloc[0]['a'] == 1
        assert result.iloc[0]['b'] == 10
        # Second row failed — both outputs NaN (pandas coerces None → NaN)
        assert pd.isna(result.iloc[1]['a'])
        assert pd.isna(result.iloc[1]['b'])


# =====================================================================
# apply_to_columns — with xarray DataArrays (integration)
# =====================================================================

class TestApplyToColumnsXarray:
    @staticmethod
    def _make_df():
        das = []
        for i in range(3):
            da = xr.DataArray(
                np.random.RandomState(i).randn(50, 2),
                dims=(DIM_TIME, DIM_CHAN),
                coords={DIM_TIME: np.linspace(0, 0.5, 50)},
                attrs={ATTR_FS: 100.0},
            )
            das.append(da)
        return pd.DataFrame({'rec': das, 'label': ['a', 'b', 'c']})

    def test_basic_transform(self):
        df = self._make_df()
        result = apply_to_columns(df, 'rec', lambda da: da.mean(DIM_TIME),
                                  result_cols='mean')
        assert 'mean' in result.columns
        assert 'rec' in result.columns
        assert len(result) == 3

    def test_extra_args_kwargs(self):
        df = self._make_df()

        def scale(da, factor, offset=0):
            return da * factor + offset

        result = apply_to_columns(df, 'rec', scale, 2, offset=1,
                                  result_cols='scaled')
        original = df.iloc[0]['rec']
        scaled = result.iloc[0]['scaled']
        np.testing.assert_allclose(scaled.values, original.values * 2 + 1)


# =====================================================================
# pipe_columns — linear pipeline
# =====================================================================

class TestPipeColumnsLinear:
    def test_basic_chain(self):
        df = pd.DataFrame({'x': [1, 2, 3]})
        result = pipe_columns(df, [
            ['x',       lambda v: v * 2,  'doubled'],
            ['doubled', lambda v: v + 1,  'plus_one'],
        ])
        assert list(result['doubled']) == [2, 4, 6]
        assert list(result['plus_one']) == [3, 5, 7]

    def test_preserves_original_columns(self):
        df = pd.DataFrame({'x': [1], 'extra': ['hi']})
        result = pipe_columns(df, [
            ['x', lambda v: v * 2, 'doubled'],
        ])
        assert list(result['extra']) == ['hi']
        assert list(result['x']) == [1]

    def test_returns_copy(self):
        df = pd.DataFrame({'x': [1]})
        result = pipe_columns(df, [['x', lambda v: v, 'out']])
        assert result is not df


# =====================================================================
# pipe_columns — branching (processing tree)
# =====================================================================

class TestPipeColumnsBranching:
    def test_two_branches_from_same_source(self):
        df = pd.DataFrame({'x': [5]})
        result = pipe_columns(df, [
            ['x', lambda v: v * 2,  'branch_a'],
            ['x', lambda v: v + 10, 'branch_b'],
        ])
        assert result.iloc[0]['branch_a'] == 10
        assert result.iloc[0]['branch_b'] == 15

    def test_branch_from_earlier_step(self):
        df = pd.DataFrame({'x': [1]})
        result = pipe_columns(df, [
            ['x', lambda v: v * 2, 'a'],      # a = 2
            ['a', lambda v: v * 3, 'b'],       # b = 6
            ['a', lambda v: v + 10, 'c'],      # c = 12 (branches from a, not b)
        ])
        assert result.iloc[0]['a'] == 2
        assert result.iloc[0]['b'] == 6
        assert result.iloc[0]['c'] == 12

    def test_multi_input_step(self):
        df = pd.DataFrame({'a': [1], 'b': [10]})
        result = pipe_columns(df, [
            [['a', 'b'], lambda a, b: a + b, 'sum'],
        ])
        assert result.iloc[0]['sum'] == 11

    def test_multi_output_step(self):
        df = pd.DataFrame({'x': [5]})
        result = pipe_columns(df, [
            ['x', lambda v: (v * 2, v * 3), ['doubled', 'tripled']],
        ])
        assert result.iloc[0]['doubled'] == 10
        assert result.iloc[0]['tripled'] == 15

    def test_multi_input_multi_output_step(self):
        df = pd.DataFrame({'a': [2], 'b': [3]})
        result = pipe_columns(df, [
            [['a', 'b'], lambda a, b: (a + b, a * b), ['sum', 'prod']],
            ['sum', lambda v: v * 10, 'sum10'],
        ])
        assert result.iloc[0]['sum'] == 5
        assert result.iloc[0]['prod'] == 6
        assert result.iloc[0]['sum10'] == 50


# =====================================================================
# pipe_columns — error handling with cascade
# =====================================================================

class TestPipeColumnsErrors:
    def test_error_cascades_to_dependents_only(self):
        """Failure in branch A should not kill independent branch B."""
        df = pd.DataFrame({'x': [1]})

        def fail(v):
            raise RuntimeError("boom")

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = pipe_columns(df, [
                ['x',  fail,             'bad'],      # fails
                ['bad', lambda v: v + 1, 'after_bad'],  # depends on bad → skipped
                ['x',  lambda v: v * 99, 'good'],     # independent → succeeds
            ], errors='warn')

        assert result.iloc[0]['bad'] is None
        assert result.iloc[0]['after_bad'] is None  # cascade
        assert result.iloc[0]['good'] == 99          # independent

    def test_error_warn(self):
        df = pd.DataFrame({'x': [1]})
        with pytest.warns(UserWarning, match="boom"):
            result = pipe_columns(df, [
                ['x', lambda v: (_ for _ in ()).throw(RuntimeError("boom")), 'out'],
            ], errors='warn')
        assert result.iloc[0]['out'] is None

    def test_error_raise(self):
        df = pd.DataFrame({'x': [1]})
        with pytest.raises(RuntimeError, match="boom"):
            pipe_columns(df, [
                ['x', lambda v: (_ for _ in ()).throw(RuntimeError("boom")), 'out'],
            ], errors='raise')

    def test_error_skip(self):
        df = pd.DataFrame({'x': [1]})
        result = pipe_columns(df, [
            ['x', lambda v: 1 / 0, 'out'],
        ], errors='skip')
        assert result.iloc[0]['out'] is None

    def test_multi_output_error_sets_all_none(self):
        df = pd.DataFrame({'x': [1]})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = pipe_columns(df, [
                ['x', lambda v: 1 / 0, ['a', 'b']],
            ], errors='warn')
        assert result.iloc[0]['a'] is None
        assert result.iloc[0]['b'] is None


# =====================================================================
# pipe_columns — validation
# =====================================================================

class TestPipeColumnsValidation:
    def test_empty_steps_raises(self):
        df = pd.DataFrame({'x': [1]})
        with pytest.raises(ValueError, match="empty"):
            pipe_columns(df, [])

    def test_wrong_steps_type_raises(self):
        df = pd.DataFrame({'x': [1]})
        with pytest.raises(TypeError, match="list"):
            pipe_columns(df, "not_valid")

    def test_bad_step_length_raises(self):
        df = pd.DataFrame({'x': [1]})
        with pytest.raises(ValueError, match="length 3"):
            pipe_columns(df, [['x', lambda v: v]])

    def test_invalid_errors_raises(self):
        df = pd.DataFrame({'x': [1]})
        with pytest.raises(ValueError, match="'warn', 'raise', or 'skip'"):
            pipe_columns(df, [['x', lambda v: v, 'out']], errors='bad')

    def test_bad_source_col_raises(self):
        df = pd.DataFrame({'x': [1]})
        with pytest.raises(ValueError, match="input column"):
            pipe_columns(df, [
                ['nonexistent', lambda v: v, 'out'],
            ])

    def test_empty_df(self):
        df = pd.DataFrame({'x': pd.Series(dtype=object)})
        result = pipe_columns(df, [['x', lambda v: v, 'out']])
        assert 'out' in result.columns
        assert len(result) == 0


# =====================================================================
# pipe_columns — with xarray DataArrays (integration)
# =====================================================================

class TestPipeColumnsXarray:
    @staticmethod
    def _make_df(n_rows=3):
        rng = np.random.RandomState(42)
        rows = []
        for i in range(n_rows):
            da = xr.DataArray(
                rng.randn(50, 2),
                dims=('time', 'channel'),
                coords={'time': np.arange(50) / 100.0,
                        'channel': [0, 1]},
                attrs={'fs': 100, 'label': f'rec_{i}'},
            )
            rows.append({'id': i, 'rec': da})
        return pd.DataFrame(rows).set_index('id')

    def test_linear_chain(self):
        df = self._make_df()
        result = pipe_columns(df, [
            ['rec', lambda da: da * 2,  'doubled'],
            ['doubled', lambda da: da + 1, 'plus_one'],
        ])
        original = df.iloc[0]['rec']
        np.testing.assert_allclose(
            result.iloc[0]['plus_one'].values,
            original.values * 2 + 1,
        )

    def test_branching(self):
        df = self._make_df()
        result = pipe_columns(df, [
            ['rec', lambda da: da * 2, 'doubled'],
            ['rec', lambda da: da + 1, 'from_original'],
        ])
        original = df.iloc[0]['rec']
        np.testing.assert_allclose(
            result.iloc[0]['from_original'].values,
            original.values + 1,
        )

    def test_error_cascade_xarray(self):
        df = self._make_df(1)

        def fail(da):
            raise RuntimeError("boom")

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = pipe_columns(df, [
                ['rec',  fail,              'fail_step'],
                ['fail_step', lambda da: da, 'after_fail'],
                ['rec',  lambda da: da * 2, 'good_branch'],
            ], errors='warn')

        assert result.iloc[0]['fail_step'] is None
        assert result.iloc[0]['after_fail'] is None     # cascade
        assert result.iloc[0]['good_branch'] is not None  # independent
