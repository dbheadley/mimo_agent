"""Tests for mimo_pack.util.xarray — group_by_structure, select_mode."""

import numpy as np
import pytest
import xarray as xr

from mimo_pack.consts import DIM_CHAN, DIM_EPOCH, DIM_TIME, COORD_SHANK, COORD_X, COORD_Y
from mimo_pack.testing.data import synthetic_lfp
from mimo_pack.util.xarray import group_by_structure, select_mode


class TestGroupByStructure:
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make(duration, fs=1000, n_chans=4, seed=0):
        return synthetic_lfp(duration=duration, fs=fs, n_chans=n_chans, seed=seed)

    # ------------------------------------------------------------------
    # Basic functionality
    # ------------------------------------------------------------------
    def test_all_same_structure_one_group(self):
        """Arrays with identical dims, sizes, and coords land in one group."""
        das = [self._make(2.0, seed=i) for i in range(3)]
        groups = group_by_structure(das)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_different_duration_two_groups(self):
        """Arrays with different time-axis lengths split into separate groups."""
        lfp_a = self._make(2.0, seed=0)
        lfp_b = self._make(2.0, seed=1)
        lfp_c = self._make(1.0, seed=2)  # different n_time
        groups = group_by_structure([lfp_a, lfp_b, lfp_c])
        assert len(groups) == 2
        assert len(groups[0]) == 2  # lfp_a and lfp_b
        assert len(groups[1]) == 1  # lfp_c alone

    def test_different_n_chans_two_groups(self):
        lfp_4 = self._make(1.0, n_chans=4, seed=0)
        lfp_8 = self._make(1.0, n_chans=8, seed=0)
        groups = group_by_structure([lfp_4, lfp_8])
        assert len(groups) == 2

    def test_single_element_list(self):
        das = [self._make(1.0, seed=0)]
        groups = group_by_structure(das)
        assert len(groups) == 1
        assert groups[0][0] is das[0]

    def test_output_preserves_input_order(self):
        """Group members appear in the same order as the input list."""
        das = [self._make(2.0, seed=i) for i in range(4)]
        groups = group_by_structure(das)
        assert len(groups) == 1
        for result, original in zip(groups[0], das):
            assert result is original

    def test_return_type(self):
        das = [self._make(1.0, seed=0), self._make(1.0, seed=1)]
        groups = group_by_structure(das)
        assert isinstance(groups, list)
        assert all(isinstance(g, list) for g in groups)
        assert all(isinstance(da, xr.DataArray) for g in groups for da in g)

    # ------------------------------------------------------------------
    # check_coords=False
    # ------------------------------------------------------------------
    def test_check_coords_false_ignores_coord_values(self):
        """With check_coords=False, arrays that differ only in coord values
        (e.g. different time offsets) are treated as the same structure."""
        lfp_a = self._make(1.0, fs=1000, n_chans=4, seed=0)
        # Shift the time coordinate so values differ but shape is identical
        lfp_shifted = lfp_a.assign_coords(
            {DIM_TIME: lfp_a[DIM_TIME].values + 100.0}
        )
        groups_with = group_by_structure([lfp_a, lfp_shifted], check_coords=True)
        groups_without = group_by_structure([lfp_a, lfp_shifted], check_coords=False)
        assert len(groups_with) == 2       # coord values differ → split
        assert len(groups_without) == 1    # shape-only → merged

    def test_check_coords_false_groups_by_shape_only(self):
        das_2s = [self._make(2.0, seed=i) for i in range(2)]
        lfp_shifted = das_2s[0].assign_coords(
            {DIM_TIME: das_2s[0][DIM_TIME].values + 50.0}
        )
        groups = group_by_structure(das_2s + [lfp_shifted], check_coords=False)
        assert len(groups) == 1

    # ------------------------------------------------------------------
    # Tolerance (rtol / atol)
    # ------------------------------------------------------------------
    def test_atol_merges_nearly_equal_coords(self):
        """Arrays whose coords differ by less than atol should be in the same group."""
        lfp_a = self._make(1.0, seed=0)
        # Add a tiny numeric jitter (< default atol=1e-8 × 10)
        jitter = 1e-9
        lfp_jittered = lfp_a.assign_coords(
            {DIM_TIME: lfp_a[DIM_TIME].values + jitter}
        )
        groups = group_by_structure([lfp_a, lfp_jittered], rtol=0.0, atol=1e-8)
        assert len(groups) == 1

    def test_atol_splits_when_diff_exceeds_tolerance(self):
        lfp_a = self._make(1.0, seed=0)
        lfp_b = lfp_a.assign_coords(
            {DIM_TIME: lfp_a[DIM_TIME].values + 1e-4}
        )
        groups = group_by_structure([lfp_a, lfp_b], rtol=0.0, atol=1e-8)
        assert len(groups) == 2

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    def test_empty_list_raises_value_error(self):
        with pytest.raises(ValueError, match="empty"):
            group_by_structure([])

    def test_non_dataarray_raises_type_error(self):
        lfp = self._make(1.0, seed=0)
        with pytest.raises(TypeError, match="xr.DataArray"):
            group_by_structure([lfp, np.random.randn(100, 4)])

    def test_non_dataarray_error_message_includes_index(self):
        lfp = self._make(1.0, seed=0)
        with pytest.raises(TypeError, match="index 1"):
            group_by_structure([lfp, "not an array"])

    # ------------------------------------------------------------------
    # Multi-group ordering
    # ------------------------------------------------------------------
    def test_group_order_follows_first_appearance(self):
        """First group should contain the array that appears first in input."""
        lfp_short = self._make(1.0, seed=0)
        lfp_long = self._make(2.0, seed=0)
        lfp_short2 = self._make(1.0, seed=1)
        groups = group_by_structure([lfp_short, lfp_long, lfp_short2])
        assert len(groups) == 2
        # First group seeds from lfp_short (index 0)
        assert groups[0][0] is lfp_short
        # Second group seeds from lfp_long (index 1)
        assert groups[1][0] is lfp_long

    def test_three_distinct_structures(self):
        lfp_a = self._make(1.0, fs=1000, n_chans=4, seed=0)
        lfp_b = self._make(2.0, fs=1000, n_chans=4, seed=0)
        lfp_c = self._make(1.0, fs=1000, n_chans=8, seed=0)
        groups = group_by_structure([lfp_a, lfp_b, lfp_c])
        assert len(groups) == 3

    # ------------------------------------------------------------------
    # Non-float (integer / string) coordinates
    # ------------------------------------------------------------------
    def test_integer_coord_exact_match(self):
        """Integer coordinates use exact equality, not allclose."""
        lfp_a = self._make(1.0, seed=0)
        # Shift the channel integer index by 1 — should produce a separate group
        lfp_b = lfp_a.assign_coords(
            {DIM_CHAN: lfp_a[DIM_CHAN].values + 1}
        )
        groups = group_by_structure([lfp_a, lfp_b])
        assert len(groups) == 2


class TestSelectMode:
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _make_with_two_shanks(n_per_shank=6, seed=0):
        """Return a synthetic LFP with n_per_shank channels on shank 0
        and (n_per_shank - 2) channels on shank 1, so mode shank is 0."""
        n_chans = n_per_shank + (n_per_shank - 2)
        lfp = synthetic_lfp(duration=1.0, fs=1000, n_chans=n_chans, seed=seed)
        shanks = np.array([0] * n_per_shank + [1] * (n_per_shank - 2))
        return lfp.assign_coords({COORD_SHANK: (DIM_CHAN, shanks)})

    # ------------------------------------------------------------------
    # Basic functionality
    # ------------------------------------------------------------------
    def test_single_coord_keeps_mode_shank(self):
        """Channels whose shank == mode shank are kept."""
        lfp = self._make_with_two_shanks(n_per_shank=6)
        result = select_mode(lfp, COORD_SHANK)
        assert result.sizes[DIM_CHAN] == 6  # only shank-0 channels
        assert set(result[COORD_SHANK].values.tolist()) == {0}

    def test_return_type(self):
        lfp = self._make_with_two_shanks()
        result = select_mode(lfp, COORD_SHANK)
        assert isinstance(result, xr.DataArray)

    def test_all_same_coord_value_keeps_all(self):
        """When all indices share the same coord value, nothing is dropped."""
        lfp = synthetic_lfp(duration=1.0, fs=1000, n_chans=4, seed=0)
        # ch_shank from synthetic_lfp: all channels on shank 0
        result = select_mode(lfp, COORD_SHANK)
        assert result.sizes[DIM_CHAN] == lfp.sizes[DIM_CHAN]

    def test_string_input_normalised_to_list(self):
        """Passing a single string should work identically to a list of one."""
        lfp = self._make_with_two_shanks()
        r_str  = select_mode(lfp, COORD_SHANK)
        r_list = select_mode(lfp, [COORD_SHANK])
        assert r_str.sizes[DIM_CHAN] == r_list.sizes[DIM_CHAN]

    # ------------------------------------------------------------------
    # Tie-breaking
    # ------------------------------------------------------------------
    def test_tie_broken_by_lowest_value(self):
        """When two shank counts are equal, the lowest shank ID wins."""
        n_chans = 4
        lfp = synthetic_lfp(duration=1.0, fs=1000, n_chans=n_chans, seed=0)
        # 2 channels on shank 0, 2 on shank 1 — tie → shank 0 wins
        shanks = np.array([0, 0, 1, 1])
        lfp = lfp.assign_coords({COORD_SHANK: (DIM_CHAN, shanks)})
        result = select_mode(lfp, COORD_SHANK)
        assert set(result[COORD_SHANK].values.tolist()) == {0}

    # ------------------------------------------------------------------
    # Multi-coord AND / OR on the same dimension
    # ------------------------------------------------------------------
    def test_two_coords_same_dim_and_mode(self):
        """AND: keep only indices matching the mode of BOTH coords."""
        n_chans = 8
        lfp = synthetic_lfp(duration=1.0, fs=1000, n_chans=n_chans, seed=0)
        # shank coord: 5× shank 0, 3× shank 1 → mode = 0
        shanks = np.array([0, 0, 0, 0, 0, 1, 1, 1])
        # x_group coord: 6× group 0, 2× group 1 → mode = 0
        # Channels 5,6,7 (shank 1) also happen to be group 1 — no overlap with mode shank.
        # Under AND: keep indices where shank==0 AND x_group==0 → indices 0-4
        x_group = np.array([0, 0, 0, 0, 0, 1, 1, 1])
        lfp = lfp.assign_coords({
            COORD_SHANK: (DIM_CHAN, shanks),
            COORD_X:     (DIM_CHAN, x_group.astype(float)),
        })
        result = select_mode(lfp, [COORD_SHANK, COORD_X], mode='and')
        assert result.sizes[DIM_CHAN] == 5

    def test_two_coords_same_dim_or_mode(self):
        """OR: keep indices matching the mode of ANY coord."""
        n_chans = 8
        lfp = synthetic_lfp(duration=1.0, fs=1000, n_chans=n_chans, seed=0)
        # shank: mode = 0 → indices 0-4
        # x_group: mode = 0 → indices 0-5 (6 channels with x_group=0)
        shanks  = np.array([0, 0, 0, 0, 0, 1, 1, 1])
        x_group = np.array([0, 0, 0, 0, 0, 0, 1, 1])
        lfp = lfp.assign_coords({
            COORD_SHANK: (DIM_CHAN, shanks),
            COORD_X:     (DIM_CHAN, x_group.astype(float)),
        })
        result_or  = select_mode(lfp, [COORD_SHANK, COORD_X], mode='or')
        result_and = select_mode(lfp, [COORD_SHANK, COORD_X], mode='and')
        # OR should keep more (or equal) channels than AND
        assert result_or.sizes[DIM_CHAN] >= result_and.sizes[DIM_CHAN]
        assert result_or.sizes[DIM_CHAN] == 6  # union of {0-4} and {0-5}

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------
    def test_description_updated(self):
        lfp = self._make_with_two_shanks()
        result = select_mode(lfp, COORD_SHANK)
        assert 'select_mode' in result.attrs.get('description', '')
        assert COORD_SHANK in result.attrs['description']

    def test_other_attrs_preserved(self):
        lfp = self._make_with_two_shanks()
        result = select_mode(lfp, COORD_SHANK)
        from mimo_pack.consts import ATTR_FS
        assert ATTR_FS in result.attrs

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    def test_type_error_on_non_dataarray(self):
        with pytest.raises(TypeError, match="xr.DataArray"):
            select_mode(np.random.randn(100, 4), COORD_SHANK)

    def test_value_error_on_unknown_coord(self):
        lfp = synthetic_lfp(duration=1.0, fs=1000, n_chans=4, seed=0)
        with pytest.raises(ValueError, match="nonexistent_coord"):
            select_mode(lfp, 'nonexistent_coord')

    def test_value_error_on_invalid_mode(self):
        lfp = synthetic_lfp(duration=1.0, fs=1000, n_chans=4, seed=0)
        with pytest.raises(ValueError, match="mode"):
            select_mode(lfp, COORD_SHANK, mode='xor')

    def test_value_error_on_empty_coords(self):
        lfp = synthetic_lfp(duration=1.0, fs=1000, n_chans=4, seed=0)
        with pytest.raises(ValueError):
            select_mode(lfp, [])
