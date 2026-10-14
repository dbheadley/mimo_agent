# Helper functions for working with xarray objects
# Author: Drew Headley
# Date: 2025-10-06

import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from mimo_pack.consts import COORD_X, COORD_Y, COORD_SHANK, DIM_TIME, DIM_CHAN, DIM_EPOCH, ATTR_FS


def normalize_to_epoch(data: xr.DataArray, stim_dims: list[str] = None) -> xr.DataArray:
    """Convert named stimulus dimensions to DIM_EPOCH + coordinates.
    
    This utility converts xarray data with stimulus features as named dimensions
    (e.g., from pynapple's compute_tuning_curves) into the mimo_pack standard
    format: DIM_EPOCH with stimulus properties as coordinates.
    
    Parameters
    ----------
    data : xr.DataArray
        The input DataArray with stimulus features as named dimensions.
    stim_dims : list of str, optional
        List of dimension names to convert to coordinates. If None, converts
        all dimensions except DIM_TIME, DIM_CHAN, and DIM_FREQ to coordinates.
        
    Returns
    -------
    xr.DataArray
        DataArray with stimulus dimensions stacked into DIM_EPOCH and their
        values stored as coordinates. The 'stimulus_properties' attribute
        is populated with the list of coordinate names from the stacked dims.
        
    Examples
    --------
    >>> # Pynapple returns orientation as a dimension
    >>> tuning = nap.compute_tuning_curves(spikes, position_tsd, bins=8)
    >>> tuning.dims  # ('unit', 'position')
    >>> 
    >>> # Convert to mimo_pack standard
    >>> tuning_norm = normalize_to_epoch(tuning, stim_dims=['position'])
    >>> tuning_norm.dims  # ('unit', 'epoch')
    >>> tuning_norm.coords  # 'position' is now a coordinate on 'epoch'
    >>> tuning_norm.attrs['stimulus_properties']  # ['position']
    
    Notes
    -----
    - If data already has DIM_EPOCH, new stacked dims will be combined with it.
    - The function preserves all other coordinates and attributes.
    - Original dimension values become coordinate arrays on the new DIM_EPOCH axis.
    """
    if stim_dims is None:
        # Auto-detect: stack all dims except standard data dims
        standard_dims = {DIM_TIME, DIM_CHAN, DIM_FREQ, DIM_EPOCH}
        stim_dims = [d for d in data.dims if d not in standard_dims]
    
    if len(stim_dims) == 0:
        # Nothing to normalize
        return data
    
    # If DIM_EPOCH already exists, we'll be combining it with stim_dims
    if DIM_EPOCH in stim_dims:
        stim_dims.remove(DIM_EPOCH)
    
    if len(stim_dims) == 0:
        # Already in correct format
        return data
    
    # Stack stimulus dimensions into DIM_EPOCH
    # This creates a MultiIndex which we'll flatten
    result = data.stack({DIM_EPOCH: stim_dims})
    
    # If the result has a MultiIndex on DIM_EPOCH, reset it to separate coordinates
    if isinstance(result.indexes.get(DIM_EPOCH), pd.MultiIndex):
        result = result.reset_index(DIM_EPOCH)
    
    # Propagate all attrs from input before adding/overwriting specific keys
    result.attrs = data.attrs.copy()

    # Update stimulus_properties attribute
    existing_stim_props = result.attrs.get('stimulus_properties', [])
    new_stim_props = list(set(existing_stim_props + stim_dims))
    result.attrs['stimulus_properties'] = new_stim_props
    
    # Update description for provenance
    desc_base = data.attrs.get('description', 'data')
    result.attrs['description'] = f"{desc_base} -> normalized to DIM_EPOCH"
    
    return result


def get_dim(da: xr.DataArray, coord_name: str) -> str:
    """Helper to resolve a coordinate name to its underlying dimension.
    
    Parameters
    ----------
    da : xr.DataArray
        The DataArray to examine.
    coord_name : str
        Name of coordinate to resolve.
        
    Returns
    -------
    str
        The underlying dimension name.
        
    Raises
    ------
    ValueError
        If the coordinate is not found.
    """
    if coord_name in da.dims:
        return coord_name
    elif coord_name in da.coords:
        return da.coords[coord_name].dims[0]
    else:
        raise ValueError(f"Coordinate '{coord_name}' not found in DataArray.")


def group_and_stack(data: xr.DataArray, group_coord: str) -> xr.DataArray:
    """
    Group a DataArray by a coordinate and concatenate groups into a new dimension.
    
    This function groups the data by the specified coordinate and concatenates 
    the groups back together, replacing the original dimension that the coordinate
    lived on with a new dimension named after the grouping coordinate.
    
    Parameters
    ----------
    data : xr.DataArray
        The input DataArray to group and stack.
    group_coord : str
        Name of the coordinate to group by.
        
    Returns
    -------
    xr.DataArray
        The regrouped and stacked DataArray with the original coordinate's 
        dimension replaced by group_coord as a new dimension.
    """
    # Find which dimension the grouping coordinate lives on
    if group_coord not in data.coords:
        raise ValueError(f"Coordinate '{group_coord}' not found in data")
        
    slices = []
    keys = []
    
    for k, g in data.groupby(group_coord):
        # Remove the grouping coordinate to avoid duplication
        g = g.drop_vars(group_coord, errors='ignore')
        slices.append(g)
        keys.append(k)
    
    # Stack along new dimension, replacing the original dimension
    idx = pd.Index(keys, name=group_coord)
    result = xr.concat(slices, dim=idx)
    
    # The result should have the same number of dimensions as input,
    # with original_dim replaced by group_coord

    # Propagate all attrs from input before overwriting specific keys
    result.attrs = data.attrs.copy()

    # Update description attribute for provenance
    desc_base = data.attrs.get('description', 'data')
    result.attrs['description'] = f"{desc_base} -> grouped by {group_coord}"
    
    return result


# ==========================================
# Validation and Accessor Functions
# ==========================================

def get_fs(da: xr.DataArray) -> float:
    """
    Get the sample rate from a DataArray's attributes.
    
    Parameters
    ----------
    da : xr.DataArray
        The DataArray to get the sample rate from.
        
    Returns
    -------
    fs : float
        The sample rate in Hz.
        
    Raises
    ------
    ValueError
        If the DataArray does not have a sample rate attribute.
    """
    if ATTR_FS not in da.attrs:
        raise ValueError(f"DataArray does not have '{ATTR_FS}' attribute. "
                        f"Available attrs: {list(da.attrs.keys())}")
    return float(da.attrs[ATTR_FS])


def validate_dims(da: xr.DataArray, required: list[str]) -> None:
    """
    Validate that a DataArray has the required dimensions.
    
    Parameters
    ----------
    da : xr.DataArray
        The DataArray to validate.
    required : list[str]
        List of required dimension names.
        
    Raises
    ------
    ValueError
        If any required dimensions are missing.
    """
    missing = [d for d in required if d not in da.dims]
    if missing:
        raise ValueError(f"DataArray missing required dimensions: {missing}. "
                        f"Available dims: {list(da.dims)}")


def validate_coords(da: xr.DataArray, required: list[str]) -> None:
    """
    Validate that a DataArray has the required coordinates.
    
    Parameters
    ----------
    da : xr.DataArray
        The DataArray to validate.
    required : list[str]
        List of required coordinate names.
        
    Raises
    ------
    ValueError
        If any required coordinates are missing.
    """
    missing = [c for c in required if c not in da.coords]
    if missing:
        raise ValueError(f"DataArray missing required coordinates: {missing}. "
                        f"Available coords: {list(da.coords.keys())}")


def expand_dict_attr(da: xr.DataArray, attr: str, *,
                     drop: bool = True, prefix: str = '') -> xr.DataArray:
    """Promote items of a dict-valued attribute to individual attributes.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray whose ``attrs[attr]`` is a ``dict``.
    attr : str
        Name of the attribute containing the dictionary.
    drop : bool, optional
        If ``True`` (default), remove the original dict attribute after
        expansion.  If ``False``, keep it alongside the new keys.
    prefix : str, optional
        String prepended to each dictionary key when creating the new
        attribute names, e.g. ``prefix='proto_'`` would produce
        ``'proto_key1'``, ``'proto_key2'``, etc.

    Returns
    -------
    xr.DataArray
        A copy of *da* with the dictionary entries added as top-level
        attributes.

    Raises
    ------
    KeyError
        If *attr* is not present in ``da.attrs``.
    TypeError
        If ``da.attrs[attr]`` is not a ``dict``.

    Examples
    --------
    >>> da.attrs['protocol_settings'] = {'led_dur': 0.5, 'led_power': 4.5}
    >>> da2 = expand_dict_attr(da, 'protocol_settings')
    >>> da2.attrs['led_dur']
    0.5
    >>> 'protocol_settings' not in da2.attrs
    True
    """
    if attr not in da.attrs:
        raise KeyError(f"Attribute '{attr}' not found in DataArray attrs. "
                       f"Available: {list(da.attrs.keys())}")
    d = da.attrs[attr]
    if not isinstance(d, dict):
        raise TypeError(f"attrs['{attr}'] must be a dict, got {type(d).__name__}.")

    da = da.copy()
    for key, value in d.items():
        da.attrs[f'{prefix}{key}'] = value
    if drop:
        del da.attrs[attr]
    return da


# ==========================================
# Channel Selection Functions
# ==========================================

def select_channels(da: xr.DataArray, x: float | list = None, 
                    y: float | list = None, shank: int | list = None,
                    dim: str = 'channel', tolerance: float = None) -> xr.DataArray:
    """
    Select channels from a DataArray by spatial coordinates.
    
    Parameters
    ----------
    da : xr.DataArray
        The DataArray to select channels from. Must have coordinates
        for ch_x, ch_y, and/or ch_shank along the channel dimension.
    x : float or list, optional
        X coordinate(s) to select. If tolerance is set, selects channels
        within tolerance of these values.
    y : float or list, optional
        Y coordinate(s) to select. If tolerance is set, selects channels
        within tolerance of these values.
    shank : int or list, optional
        Shank number(s) to select.
    dim : str, optional
        The dimension name for channels. Default is 'channel'.
    tolerance : float, optional
        Tolerance for coordinate matching. If None, exact matching is used.
        
    Returns
    -------
    xr.DataArray
        DataArray with only the selected channels.
        
    Examples
    --------
    >>> # Select channels on shank 0
    >>> da_shank0 = select_channels(da, shank=0)
    >>> 
    >>> # Select channels at x=0 and y between 100-200
    >>> da_subset = select_channels(da, x=0, y=[100, 150, 200])
    """
    if dim not in da.dims:
        raise ValueError(f"Dimension '{dim}' not found. Available: {list(da.dims)}")
    
    # Build mask for channel selection
    mask = np.ones(da.sizes[dim], dtype=bool)
    
    if x is not None:
        validate_coords(da, [COORD_X])
        x_vals = da.coords[COORD_X].values
        if tolerance is not None:
            x = [x] if np.isscalar(x) else x
            x_mask = np.zeros_like(mask)
            for xv in x:
                x_mask |= np.abs(x_vals - xv) <= tolerance
            mask &= x_mask
        else:
            x = [x] if np.isscalar(x) else x
            mask &= np.isin(x_vals, x)
    
    if y is not None:
        validate_coords(da, [COORD_Y])
        y_vals = da.coords[COORD_Y].values
        if tolerance is not None:
            y = [y] if np.isscalar(y) else y
            y_mask = np.zeros_like(mask)
            for yv in y:
                y_mask |= np.abs(y_vals - yv) <= tolerance
            mask &= y_mask
        else:
            y = [y] if np.isscalar(y) else y
            mask &= np.isin(y_vals, y)
    
    if shank is not None:
        validate_coords(da, [COORD_SHANK])
        shank_vals = da.coords[COORD_SHANK].values
        shank = [shank] if np.isscalar(shank) else shank
        mask &= np.isin(shank_vals, shank)
    
    result = da.isel({dim: mask})
    
    # Update description attribute for provenance
    desc_base = da.attrs.get('description', 'data')
    filters = []
    if x is not None:
        filters.append(f"x={x}")
    if y is not None:
        filters.append(f"y={y}")
    if shank is not None:
        filters.append(f"shank={shank}")
    filter_str = ", ".join(filters)
    result.attrs['description'] = f"{desc_base} -> selected channels ({filter_str})"
    
    return result


def select_by_coord(da: xr.DataArray, coord: str, values,
                    dim: str = None) -> xr.DataArray:
    """Select indices of a DataArray where a coordinate equals specific value(s).

    This is a generic selection utility that works with any coordinate
    (e.g., ``ch_units``, ``ch_names``, ``chan_type``, ``stim_type``).
    Unlike :func:`select_channels` which is limited to spatial coordinates,
    this function accepts an arbitrary coordinate name and one or more
    target values.

    Parameters
    ----------
    da : xr.DataArray
        The DataArray to subset.
    coord : str
        Name of the coordinate to filter on.  Must exist in ``da.coords``.
    values : scalar or array-like
        Value(s) to keep.  Indices where ``da[coord]`` equals any of these
        values are retained; all others are dropped.
    dim : str, optional
        The dimension to subset along.  If *None* the dimension is inferred
        from the coordinate (the coordinate must be one-dimensional).

    Returns
    -------
    xr.DataArray
        Subset of *da* containing only the matching indices.

    Raises
    ------
    ValueError
        If *coord* is not present, no indices match, or the dimension
        cannot be inferred.

    Examples
    --------
    >>> import xarray as xr, numpy as np
    >>> da = xr.DataArray(
    ...     np.random.randn(100, 3),
    ...     dims=('time', 'channel'),
    ...     coords={'ch_units': ('channel', ['mV', 'pA', 'mV'])},
    ... )
    >>> select_by_coord(da, 'ch_units', 'pA').sizes['channel']
    1
    >>> select_by_coord(da, 'ch_units', 'mV').sizes['channel']
    2
    """
    validate_coords(da, [coord])

    coord_vals = da.coords[coord].values

    # Normalise values to array for np.isin
    if np.isscalar(values) or isinstance(values, str):
        values = [values]

    mask = np.isin(coord_vals, values)

    if not mask.any():
        available = np.unique(coord_vals).tolist()
        raise ValueError(
            f"No indices where '{coord}' is in {list(values)}. "
            f"Available values: {available}"
        )

    # Determine the dimension this coordinate lives on
    if dim is None:
        coord_dims = da.coords[coord].dims
        if len(coord_dims) != 1:
            raise ValueError(
                f"Cannot infer dimension for multi-dimensional coordinate "
                f"'{coord}' (dims={coord_dims}).  Pass 'dim' explicitly."
            )
        dim = coord_dims[0]

    result = da.isel({dim: mask})

    # Update description attribute for provenance
    desc_base = da.attrs.get('description', 'data')
    result.attrs['description'] = f"{desc_base} -> selected by {coord}={list(values)}"

    return result


def select_mode(
    da: xr.DataArray,
    coords: str | list[str],
    mode: str = 'and',
) -> xr.DataArray:
    """Keep only indices where coordinate(s) equal their modal (most frequent) value.

    For each specified coordinate the mode is computed as the value with the
    highest occurrence count.  Ties are broken by the lowest value.  The
    resulting boolean mask is then used to index the DataArray with
    ``isel``.

    When multiple coordinates are provided they may map to the same or
    different underlying dimensions:

    - Coordinates on **different dimensions** always produce independent
      masks; both ``'and'`` and ``'or'`` behave identically (each dimension
      is filtered by its own coordintate's mask).
    - Coordinates on the **same dimension** are combined according to
      *mode*:

      - ``'and'`` *(default)*: keep indices where **all** specified coords
        equal their mode → intersection.
      - ``'or'``: keep indices where **any** specified coord equals its
        mode → union.

    Parameters
    ----------
    da : xr.DataArray
        Input DataArray.
    coords : str or list of str
        Coordinate name(s) to use for filtering.  Each must exist in
        ``da.coords``.
    mode : {'and', 'or'}, optional
        How to combine multiple coordinates that share the same underlying
        dimension.  Default is ``'and'``.

    Returns
    -------
    xr.DataArray
        Filtered DataArray with the same dimensions as *da* but only
        those indices where the specified coordinates equal their mode
        value(s).

    Raises
    ------
    TypeError
        If *da* is not an ``xr.DataArray``.
    ValueError
        If any coordinate name is not found, or *mode* is not
        ``'and'`` or ``'or'``.

    Examples
    --------
    >>> from mimo_pack.testing.data import synthetic_lfp
    >>> from mimo_pack.consts import COORD_SHANK
    >>> lfp = synthetic_lfp(duration=1.0, fs=1000, n_chans=16, seed=0)
    >>> # Keep only the channels whose shank ID is the most common
    >>> lfp_mode_shank = select_mode(lfp, COORD_SHANK)

    >>> # Keep epochs where BOTH stim_x AND stim_y equal their mode (AND)
    >>> result_and = select_mode(trials, ['stim_x', 'stim_y'], mode='and')

    >>> # Keep epochs where stim_x OR stim_y equals its mode (OR)
    >>> result_or = select_mode(trials, ['stim_x', 'stim_y'], mode='or')
    """
    if not isinstance(da, xr.DataArray):
        raise TypeError(
            f"Expected xr.DataArray, got {type(da).__name__}. "
            "Use mimo_pack.fileio functions to load data first."
        )

    if mode not in ('and', 'or'):
        raise ValueError(
            f"mode must be 'and' or 'or', got '{mode}'."
        )

    coords_list = [coords] if isinstance(coords, str) else list(coords)
    if not coords_list:
        raise ValueError("coords must contain at least one coordinate name.")

    validate_coords(da, coords_list)

    # ------------------------------------------------------------------
    # Compute mode value and boolean mask for each coordinate
    # ------------------------------------------------------------------
    def _mode_value(vals: np.ndarray):
        """Return the most frequent value; ties resolved by lowest value.

        NaN values are grouped together and counted as a single category.
        In a tie between NaN and a non-NaN value, the non-NaN value wins.
        """
        nan_mask = pd.isnull(vals)
        n_nan = int(nan_mask.sum())
        non_nan_vals = vals[~nan_mask]
        if non_nan_vals.size > 0:
            unique, counts = np.unique(non_nan_vals, return_counts=True)
            best_count = int(counts[np.argmax(counts)])
            if n_nan > best_count:
                return float('nan')
            # NaN loses ties; np.unique already returns sorted (lowest first)
            return unique[np.argmax(counts)]
        elif n_nan > 0:
            return float('nan')
        else:
            raise ValueError(
                f"Coordinate '{coord_name}' has no values; cannot compute a mode."
            )

    # dim_name → list of per-coord boolean masks for that dimension
    dim_masks: dict[str, list[np.ndarray]] = {}
    mode_vals: dict[str, object] = {}

    for coord_name in coords_list:
        vals = da.coords[coord_name].values
        mv = _mode_value(vals)
        mode_vals[coord_name] = mv
        try:
            is_nan_mode = np.isnan(float(mv))
        except (TypeError, ValueError):
            is_nan_mode = False
        mask = pd.isnull(vals) if is_nan_mode else vals == mv
        dim_name = get_dim(da, coord_name)
        dim_masks.setdefault(dim_name, []).append(mask)

    # ------------------------------------------------------------------
    # Combine masks within each dimension and apply isel
    # ------------------------------------------------------------------
    isel_dict: dict[str, np.ndarray] = {}
    for dim_name, masks in dim_masks.items():
        if len(masks) == 1:
            combined = masks[0]
        elif mode == 'and':
            combined = masks[0].copy()
            for m in masks[1:]:
                combined &= m
        else:  # 'or'
            combined = masks[0].copy()
            for m in masks[1:]:
                combined |= m
        isel_dict[dim_name] = combined

    result = da.isel(isel_dict)

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------
    desc_base = da.attrs.get('description', 'data')
    mode_str = ", ".join(f"{k}={v}" for k, v in mode_vals.items())
    result.attrs['description'] = (
        f"{desc_base} -> select_mode({{{mode_str}}}, mode='{mode}')"
    )

    return result


# ==========================================
# Structure Comparison Functions
# ==========================================

def _coords_match(da1: xr.DataArray, da2: xr.DataArray,
                  rtol: float, atol: float) -> bool:
    """Compare coordinate values of two DataArrays with the same coord names.

    Returns True if every coordinate array in *da1* matches the corresponding
    array in *da2* within the specified tolerance (float coords) or exactly
    (non-float coords).  Both DataArrays are assumed to already share the same
    set of coordinate names and the same dimension sizes (i.e. they passed the
    coarse-key check).
    """
    for name in da1.coords:
        c1 = da1.coords[name].values
        c2 = da2.coords[name].values
        if c1.shape != c2.shape:
            return False
        if np.issubdtype(c1.dtype, np.floating) or np.issubdtype(c2.dtype, np.floating):
            if not np.allclose(c1, c2, rtol=rtol, atol=atol, equal_nan=True):
                return False
        else:
            if not np.array_equal(c1, c2):
                return False
    return True


def group_by_structure(
    das: list[xr.DataArray],
    check_coords: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> list[list[xr.DataArray]]:
    """Group DataArrays by agreement between their dimensions and coordinates.

    Two DataArrays are considered structurally equivalent when they have:

    1. **Identical dimensions** — same names *and* same sizes (in any order).
    2. **Identical coordinate names** — the same set of coordinate variables.
    3. **Matching coordinate values** *(optional)* — when ``check_coords=True``,
       every coordinate array must also be numerically equal within the supplied
       tolerance.  Float coordinates use ``np.allclose``; integer/string/object
       coordinates use exact equality.  Set ``check_coords=False`` to skip this
       refinement and group solely by dims and coord names.

    Data values and ``.attrs`` are **never** compared.

    Parameters
    ----------
    das : list of xr.DataArray
        DataArrays to group.  Must be non-empty; all elements must be
        ``xr.DataArray``.
    check_coords : bool, optional
        If ``True`` (default), coordinate *values* must also match for two
        arrays to land in the same group.  If ``False``, arrays are grouped
        only by dimension names/sizes and coordinate names, making the
        function equivalent to a structural shape check.
    rtol : float, optional
        Relative tolerance for floating-point coordinate comparison.
        Passed to ``np.allclose``.  Default is ``1e-5``.
    atol : float, optional
        Absolute tolerance for floating-point coordinate comparison.
        Passed to ``np.allclose``.  Default is ``1e-8``.

    Returns
    -------
    list of list of xr.DataArray
        Each inner list is a group of structurally equivalent DataArrays.
        Groups are ordered by the index of their first member in *das*;
        order within each group matches the original input order.

    Raises
    ------
    ValueError
        If *das* is empty.
    TypeError
        If any element of *das* is not an ``xr.DataArray``.

    Examples
    --------
    >>> from mimo_pack.testing.data import synthetic_lfp
    >>> lfp_a = synthetic_lfp(duration=2.0, fs=1000, n_chans=4, seed=0)
    >>> lfp_b = synthetic_lfp(duration=2.0, fs=1000, n_chans=4, seed=1)
    >>> lfp_c = synthetic_lfp(duration=1.0, fs=1000, n_chans=4, seed=2)  # different duration
    >>> groups = group_by_structure([lfp_a, lfp_b, lfp_c])
    >>> len(groups)
    2
    >>> len(groups[0])   # lfp_a and lfp_b share structure
    2
    >>> len(groups[1])   # lfp_c is alone
    1

    Use ``check_coords=False`` to group by shape only (ignoring coord values):

    >>> groups_shape = group_by_structure([lfp_a, lfp_b, lfp_c], check_coords=False)
    >>> len(groups_shape)   # all have same n_chans and n_time samples
    1
    """
    if len(das) == 0:
        raise ValueError(
            "Cannot group an empty list. "
            "Provide at least one xr.DataArray."
        )
    for i, da in enumerate(das):
        if not isinstance(da, xr.DataArray):
            raise TypeError(
                f"Element at index {i} is {type(da).__name__}, expected xr.DataArray. "
                "Use mimo_pack.fileio functions to load data first."
            )

    # ------------------------------------------------------------------
    # Phase a — coarse partitioning by dims (name + size) and coord names
    # ------------------------------------------------------------------
    def _coarse_key(da: xr.DataArray):
        dims_key = tuple(sorted((d, da.sizes[d]) for d in da.dims))
        coord_key = frozenset(da.coords)
        return (dims_key, coord_key)

    # Map coarse key → list of (original_index, da)
    coarse: dict = {}
    for i, da in enumerate(das):
        key = _coarse_key(da)
        coarse.setdefault(key, []).append((i, da))

    # ------------------------------------------------------------------
    # Phase b — optional refinement by coordinate values within each
    # coarse group; greedy O(M²) pass (M = coarse group size, typ. small)
    # ------------------------------------------------------------------
    # Collect final groups in order of first-member index
    final_groups: dict[int, list[xr.DataArray]] = {}  # first_idx → members

    for bucket in coarse.values():
        if not check_coords or len(bucket) == 1:
            # No refinement needed — entire bucket is one group
            first_idx = bucket[0][0]
            final_groups[first_idx] = [da for _, da in bucket]
            continue

        # Greedy grouping: pick the first unassigned array, collect all
        # that match it, then repeat for any unmatched remainder.
        unassigned = list(bucket)  # [(orig_idx, da), ...]
        while unassigned:
            seed_idx, seed_da = unassigned[0]
            group_das = [seed_da]
            remaining = []
            for orig_idx, da in unassigned[1:]:
                if _coords_match(seed_da, da, rtol=rtol, atol=atol):
                    group_das.append(da)
                else:
                    remaining.append((orig_idx, da))
            final_groups[seed_idx] = group_das
            unassigned = remaining

    # Return groups sorted by the index of their first member
    return [final_groups[k] for k in sorted(final_groups)]


# ==========================================
# Concatenation Functions
# ==========================================

def _intersect_attrs(das: list[xr.DataArray]) -> dict:
    """Return attrs shared with equal values across all DataArrays.

    Only keys present in every DataArray whose values compare equal are kept.
    Numpy array values are compared with ``np.array_equal``; all other values
    are compared with ``==``.  If the comparison raises an exception (e.g.
    ambiguous truth value for an array) the key is excluded.

    Parameters
    ----------
    das : list of xr.DataArray
        At least one DataArray.

    Returns
    -------
    dict
        The intersection of attrs across all inputs.
    """
    if len(das) == 0:
        return {}
    if len(das) == 1:
        return das[0].attrs.copy()
    result = {}
    for key, val in das[0].attrs.items():
        all_equal = True
        for da in das[1:]:
            if key not in da.attrs:
                all_equal = False
                break
            other = da.attrs[key]
            try:
                if isinstance(val, np.ndarray):
                    if not (isinstance(other, np.ndarray) and np.array_equal(val, other)):
                        all_equal = False
                        break
                elif val != other:
                    all_equal = False
                    break
            except (ValueError, TypeError):
                all_equal = False
                break
        if all_equal:
            result[key] = val
    return result


def concat_by_attr(das: list[xr.DataArray], attr: str | list[str] | None = None,
                   dim: str = None, keep_attrs: str = 'identical') -> xr.DataArray:
    """
    Concatenate DataArrays using attribute values as coordinates on the new dimension.
    
    This is useful when combining data from multiple sessions, conditions,
    or files where each DataArray has metadata stored in attributes.
    
    Parameters
    ----------
    das : list[xr.DataArray]
        List of DataArrays to concatenate.
    attr : str, list of str, or None, optional
        Attribute name(s) to promote to coordinates on the concat dimension.

        - ``str``: single attribute; ``dim`` defaults to this name (backward-compatible).
        - ``list[str]``: multiple attributes, each becomes a coordinate; ``dim``
          must be provided (or there must be exactly one attr in the list).
        - ``None``: auto-detect all attribute keys shared across every DataArray
          and create a coordinate for each; ``dim`` must be provided unless
          exactly one shared attribute is found.
    dim : str, optional
        Dimension name to concatenate along. If None, inferred from ``attr`` when
        possible (single string or a single auto-detected attribute).
        If the dimension already exists in the DataArrays, concatenates along it
        and appends attribute values as extra coordinates; otherwise a new
        dimension is created.
    keep_attrs : str, optional
        How to handle attrs in the result:
        - ``'identical'``: keep only attrs identical across all DataArrays (default)
        - ``'first'``: use attrs from the first DataArray
        - ``'drop'``: discard all attrs
        
    Returns
    -------
    xr.DataArray
        Concatenated array with attribute values as coordinates along the
        concat dimension.
        
    Raises
    ------
    ValueError
        If any DataArray is missing a requested attribute, or if ``dim`` cannot
        be inferred.
        
    Examples
    --------
    >>> # Single attr — dim defaults to attr name; coord named after attr
    >>> combined = concat_by_attr(das, attr='session')
    >>> combined.coords['session']   # array(['A', 'B', 'C'])
    >>> combined.dims                # ('session', ...)  — integer indices
    >>>
    >>> # Multiple attrs along an existing 'epoch' dimension
    >>> combined = concat_by_attr(das, attr=['subject', 'condition'], dim='epoch')
    >>> # coords 'subject' and 'condition' are added on 'epoch'
    >>>
    >>> # Auto-detect all shared attrs
    >>> combined = concat_by_attr(das, attr=None, dim='recording')
    """
    if not das:
        raise ValueError("Cannot concatenate empty list of DataArrays")

    # ---- Normalise attr → attrs list -----------------------------------------
    if attr is None:
        # Auto-detect keys present in every DataArray
        common_keys = set(das[0].attrs.keys())
        for da in das[1:]:
            common_keys &= set(da.attrs.keys())
        attrs = sorted(common_keys)
        if not attrs:
            raise ValueError("No shared attributes found across DataArrays. "
                             "Specify 'attr' explicitly.")
    elif isinstance(attr, str):
        attrs = [attr]
    else:
        attrs = list(attr)

    # ---- Resolve dim ------------------------------------------------------
    if dim is None:
        if len(attrs) == 1:
            dim = attrs[0]
        else:
            raise ValueError(
                "Must specify 'dim' when 'attr' is a list or None resolves to "
                "multiple attributes."
            )

    # ---- Collect per-DataArray attr values ---------------------------------
    all_attr_values = []           # list of dicts, one per DataArray
    for i, da in enumerate(das):
        row = {}
        for a in attrs:
            if a not in da.attrs:
                raise ValueError(
                    f"DataArray at index {i} missing attribute '{a}'. "
                    f"Available attrs: {list(da.attrs.keys())}"
                )
            row[a] = da.attrs[a]
        all_attr_values.append(row)

    # ---- Concatenate -------------------------------------------------------
    dim_exists = any(dim in da.dims for da in das)

    if dim_exists:
        # Concat along existing dimension; add each attr as a named coordinate.
        # join='outer' unions mismatched coordinates on other dims, filling gaps with NaN.
        result = xr.concat(das, dim=dim, join='outer')
        for a in attrs:
            coord_values = []
            for da, av in zip(das, all_attr_values):
                coord_values.extend([av[a]] * da.sizes[dim])
            result = result.assign_coords({a: (dim, coord_values)})
    else:
        # Use integer indices for the new dim; add each attr as a named coordinate.
        # join='outer' handles DataArrays with different shapes on other dims.
        expanded = [da.expand_dims({dim: [i]}) for i, da in enumerate(das)]
        result = xr.concat(expanded, dim=dim, join='outer')
        for a in attrs:
            result = result.assign_coords(
                {a: (dim, [av[a] for av in all_attr_values])}
            )

    # ---- Handle result attrs -----------------------------------------------
    if keep_attrs == 'drop':
        result.attrs = {}
    elif keep_attrs == 'first':
        result.attrs = das[0].attrs.copy()
    elif keep_attrs == 'identical':
        result.attrs = _intersect_attrs(das)
    else:
        raise ValueError(f"Invalid keep_attrs value: '{keep_attrs}'. "
                         f"Must be 'identical', 'first', or 'drop'.")

    return result


def explode_xarray(df: pd.DataFrame, col: str | list[str], dim: str,
                   coord_col: str | None = None,
                   keep_coords: bool = True) -> pd.DataFrame:
    """Split DataArray column(s) along a dimension into separate rows.

    This is the inverse of :func:`concat_by_attr`.  For each row in *df*,
    the DataArray(s) in *col* are sliced along *dim* with ``.isel()``,
    producing one output row per position.  Scalar (non-xarray) columns
    are duplicated.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with one or more columns containing
        :class:`xarray.DataArray` objects.
    col : str or list of str
        Column name(s) holding DataArrays to split.  All specified
        columns must share the same size along *dim* within each row.
    dim : str
        Dimension name to split along.
    coord_col : str or None, optional
        Name of the new column that will hold the coordinate value at
        each position along *dim*.  Defaults to *dim*.
    keep_coords : bool, optional
        If ``True`` (default), other coordinates defined along *dim* in
        the first specified column are also added as new DataFrame
        columns (skipping any that already exist in *df*).

    Returns
    -------
    pandas.DataFrame
        Exploded DataFrame.  The index is reset to a default
        ``RangeIndex``; the original index values are preserved in an
        ``'_orig_index'`` column.

    Raises
    ------
    ValueError
        If a DataArray does not contain *dim*, or if multiple *col*
        entries disagree on the size of *dim* within the same row.

    Examples
    --------
    >>> import xarray as xr, pandas as pd, numpy as np
    >>> from mimo_pack.util.xarray import concat_by_attr, explode_xarray
    >>> da1 = xr.DataArray([1, 2], dims='time', attrs={'cond': 'A'})
    >>> da2 = xr.DataArray([3, 4], dims='time', attrs={'cond': 'B'})
    >>> combined = concat_by_attr([da1, da2], attr='cond', dim='cond')
    >>> df = pd.DataFrame({'data': [combined], 'subject': ['m1']})
    >>> explode_xarray(df, col='data', dim='cond')
       _orig_index  ... cond
    0            0  ...    A
    1            0  ...    B
    """
    if coord_col is None:
        coord_col = dim

    cols = [col] if isinstance(col, str) else list(col)

    rows_out: list[dict] = []
    other_cols = [c for c in df.columns if c not in cols]

    for idx, row in df.iterrows():
        # --- Determine size along dim from the first xarray column --------
        ref_da = row[cols[0]]
        if not isinstance(ref_da, xr.DataArray):
            raise TypeError(
                f"Column '{cols[0]}' at index {idx} is {type(ref_da).__name__}, "
                "expected xr.DataArray."
            )
        if dim not in ref_da.dims:
            raise ValueError(
                f"DataArray in column '{cols[0]}' at index {idx} does not "
                f"have dimension '{dim}'. Dims: {ref_da.dims}"
            )
        n = ref_da.sizes[dim]

        # Validate other cols have the same size
        for c in cols[1:]:
            da_c = row[c]
            if not isinstance(da_c, xr.DataArray):
                raise TypeError(
                    f"Column '{c}' at index {idx} is {type(da_c).__name__}, "
                    "expected xr.DataArray."
                )
            if dim not in da_c.dims:
                raise ValueError(
                    f"DataArray in column '{c}' at index {idx} does not "
                    f"have dimension '{dim}'. Dims: {da_c.dims}"
                )
            if da_c.sizes[dim] != n:
                raise ValueError(
                    f"Size mismatch along '{dim}' at index {idx}: "
                    f"column '{cols[0]}' has {n}, column '{c}' has "
                    f"{da_c.sizes[dim]}."
                )

        # --- Collect coordinate values along dim from reference -----------
        coord_values = (ref_da[dim].values if dim in ref_da.coords
                        else np.arange(n))

        extra_coord_names: list[str] = []
        if keep_coords:
            for cname in ref_da.coords:
                if cname == dim:
                    continue
                # Only coords that have `dim` as their sole dimension
                coord_obj = ref_da.coords[cname]
                if coord_obj.dims == (dim,) and cname not in df.columns:
                    extra_coord_names.append(cname)

        # --- Produce one output row per position --------------------------
        for i in range(n):
            new_row: dict = {'_orig_index': idx}
            # Copy scalar columns
            for c in other_cols:
                new_row[c] = row[c]
            # Slice xarray columns
            for c in cols:
                new_row[c] = row[c].isel({dim: i})
            # Add coordinate column
            new_row[coord_col] = coord_values[i]
            # Add extra coordinates
            for cname in extra_coord_names:
                new_row[cname] = ref_da.coords[cname].values[i]

            rows_out.append(new_row)

    result = pd.DataFrame(rows_out)
    result = result.reset_index(drop=True)
    return result


# ==========================================
# Interpolation Functions
# ==========================================

def interp(da: xr.DataArray, dim: str, new_coords: np.ndarray = None,
              new_spacing: float = None, method: str = 'linear',
              fill_value: float = np.nan) -> xr.DataArray:
    """
    Interpolate a DataArray along a specified dimension to new coordinates.
    
    Parameters
    ----------
    da : xr.DataArray
        The DataArray to interpolate.
    dim : str
        The dimension along which to interpolate.
    new_coords : np.ndarray, optional
        The new coordinate values to interpolate to. If None, new_spacing
        must be provided.
    new_spacing : float, optional
        Spacing for generating new coordinates. If provided, new coordinates
        are generated from min to max of the original coordinates with this
        spacing. Ignored if new_coords is provided.
    method : str, optional
        Interpolation method: 'linear', 'nearest', 'zero', 'slinear', 
        'quadratic', 'cubic'. Default is 'linear'.
    fill_value : float, optional
        Value to use for extrapolation. Default is np.nan.
        
    Returns
    -------
    xr.DataArray
        Interpolated DataArray with new coordinates along the specified dimension.
        
    Examples
    --------
    >>> # Resample time dimension to 1000 Hz
    >>> da_resampled = interp(da, 'time', new_spacing=0.001)
    >>> 
    >>> # Interpolate to specific time points
    >>> da_interp = interp(da, 'time', new_coords=np.array([0.1, 0.2, 0.3]))
    """
    validate_dims(da, [dim])
    
    # Get original coordinates
    orig_coords = da.coords[dim].values
    
    # Generate new coordinates if not provided
    if new_coords is None:
        if new_spacing is None:
            raise ValueError("Must provide either new_coords or new_spacing")
        new_coords = np.arange(orig_coords.min(), orig_coords.max() + new_spacing, new_spacing)
    
    # Use xarray's built-in interp for simple cases
    result = da.interp({dim: new_coords}, method=method, kwargs={'fill_value': fill_value})
    
    # Update description attribute for provenance
    desc_base = da.attrs.get('description', 'data')
    result.attrs['description'] = f"{desc_base} -> interpolated along {dim} ({method})"
    
    return result


def resample(da: xr.DataArray, new_fs: float, dim: str = None,
                method: str = 'linear') -> xr.DataArray:
    """
    Resample a DataArray to a new sample rate along the time dimension.
    
    Parameters
    ----------
    da : xr.DataArray
        The DataArray to resample. Must have 'fs' attribute.
    new_fs : float
        The new sample rate in Hz.
    dim : str, optional
        The dimension to resample along. Default is 'time'.
    method : str, optional
        Interpolation method. Default is 'linear'.
        
    Returns
    -------
    xr.DataArray
        Resampled DataArray with updated 'fs' attribute.
        
    Examples
    --------
    >>> # Downsample from 30000 Hz to 1000 Hz
    >>> da_downsampled = resample(da, new_fs=1000)
    """
    if dim is None:
        dim = DIM_TIME
    
    validate_dims(da, [dim])
    old_fs = get_fs(da)
    
    # Get time coordinates
    t = da.coords[dim].values
    t_min, t_max = t.min(), t.max()
    
    # Generate new time coordinates
    new_spacing = 1.0 / new_fs
    new_t = np.arange(t_min, t_max, new_spacing)
    
    # Interpolate
    da_resampled = da.interp({dim: new_t}, method=method)
    
    # Update sample rate attribute
    da_resampled.attrs[ATTR_FS] = new_fs
    
    # Update description attribute for provenance
    desc_base = da.attrs.get('description', 'data')
    da_resampled.attrs['description'] = f"{desc_base} -> resampled from {old_fs:.1f} to {new_fs:.1f} Hz"
    
    return da_resampled


def interp_to_grid(da: xr.DataArray, coord: str, new_values: np.ndarray = None,
                   spacing: float = None, dim: str = 'channel',
                   method: str = 'linear') -> xr.DataArray:
    """
    Interpolate a DataArray along a non-dimension coordinate to a regular grid.
    
    This is useful for interpolating probe data from irregular channel positions
    to a regular spatial grid.
    
    Parameters
    ----------
    da : xr.DataArray
        The DataArray to interpolate.
    coord : str
        The coordinate name to interpolate along (e.g., 'ch_y' for depth).
    new_values : np.ndarray, optional
        The new coordinate values to interpolate to. If None, spacing must
        be provided.
    spacing : float, optional
        Spacing for generating a regular grid. Ignored if new_values is provided.
    dim : str, optional
        The dimension that the coordinate is associated with. Default is 'channel'.
    method : str, optional
        Interpolation method: 'linear', 'nearest'. Default is 'linear'.
        
    Returns
    -------
    xr.DataArray
        Interpolated DataArray. The original dimension is replaced with the
        coordinate name as the new dimension.
        
    Examples
    --------
    >>> # Interpolate to regular 10um spacing along depth
    >>> da_grid = interp_to_grid(da, coord='ch_y', spacing=10)
    """
    validate_coords(da, [coord])
    
    if dim not in da.dims:
        raise ValueError(f"Dimension '{dim}' not found. Available: {list(da.dims)}")
    
    # Get original coordinate values
    orig_values = da.coords[coord].values
    
    # Generate new values if not provided
    if new_values is None:
        if spacing is None:
            raise ValueError("Must provide either new_values or spacing")
        new_values = np.arange(orig_values.min(), orig_values.max() + spacing, spacing)
    
    # Get dimension index and other dimensions
    dim_axis = list(da.dims).index(dim)
    other_dims = [d for d in da.dims if d != dim]
    
    # Move the interpolation dimension to axis 0
    da_transposed = da.transpose(dim, *other_dims)
    
    # Get shapes
    n_orig = da_transposed.sizes[dim]
    other_shapes = tuple(da_transposed.sizes[d] for d in other_dims)
    n_other = int(np.prod(other_shapes)) if other_shapes else 1
    n_new = len(new_values)
    
    # Flatten to 2D for interpolation: (n_channels, n_other)
    data_2d = da_transposed.values.reshape(n_orig, n_other)
    
    # Sort by coordinate for interpolation
    sort_idx = np.argsort(orig_values)
    orig_sorted = orig_values[sort_idx]
    data_sorted = data_2d[sort_idx, :]
    
    # Interpolate
    interp_2d = np.empty((n_new, n_other), dtype=da.dtype)
    for i in range(n_other):
        if method == 'nearest':
            # Nearest neighbor interpolation
            indices = np.searchsorted(orig_sorted, new_values)
            indices = np.clip(indices, 0, len(orig_sorted) - 1)
            interp_2d[:, i] = data_sorted[indices, i]
        else:
            # Linear interpolation
            interp_2d[:, i] = np.interp(new_values, orig_sorted, data_sorted[:, i])
    
    # Reshape back
    if other_shapes:
        interp_data = interp_2d.reshape((n_new,) + other_shapes)
    else:
        interp_data = interp_2d.flatten()
    
    # Build new coordinates
    new_coords = {coord: new_values}
    for c in da.coords:
        if dim not in da.coords[c].dims:
            new_coords[c] = da.coords[c]
    
    # Build new dimensions (replace dim with coord)
    new_dims = [coord if d == dim else d for d in da_transposed.dims]
    
    result = xr.DataArray(
        interp_data,
        dims=new_dims,
        coords=new_coords,
        attrs=da.attrs.copy()
    )
    
    # Update description attribute for provenance
    desc_base = da.attrs.get('description', 'data')
    result.attrs['description'] = f"{desc_base} -> interpolated to grid (spacing={spacing if spacing else 'custom'})"
    
    return result


# ==========================================
# Windowing Functions
# ==========================================

def window(data: xr.DataArray, dim: str, centers: np.ndarray, 
              pre: float|int, post: float|int, indices: bool = False) -> xr.DataArray:
    """
    Windows an xarray DataArray along a specified dimension.

    This function takes an xarray DataArray and extracts windows of data
    centered around specified points along a given dimension. The window
    dimension should be a continuous variable linearly spaced (e.g., time).

    Parameters
    ----------
    data : xr.DataArray
        The xarray DataArray to window.
    dim : str
        The dimension along which to window (e.g., 'time').
    centers : np.ndarray
        An array of center points for the windows.
    pre : float
        A float specifying the length before each center.
    post : float
        A float specifying the length after each center.
    indices : bool, optional
        If True, the centers, pre, and post are treated as indices rather 
        than coordinate values. If false, indices are calculated. Default is False.

    Returns
    -------
    xr.DataArray
        A new xarray DataArray containing the windowed data, concatenated
        along a new ``DIM_EPOCH`` dimension. The original windowed
        dimension name is preserved; its coordinate values are replaced
        with offsets relative to each window centre.

    Raises
    ------
    ValueError
        If ``data`` already contains a ``DIM_EPOCH`` dimension.
    """

    if DIM_EPOCH in data.dims:
        raise ValueError(
            f"Input DataArray already has a '{DIM_EPOCH}' dimension. "
            f"window() adds '{DIM_EPOCH}' as a new dimension and cannot "
            f"be applied to data that is already epoched. To re-window, "
            f"first collapse the existing epoch dimension "
            f"(e.g., data.mean('{DIM_EPOCH}') or data.isel({{'{DIM_EPOCH}': 0}}))."
        )

    dim_values = data[dim].values
    dim_step = np.mean(np.diff(dim_values))

    # Convert window parameters to sample indices
    if not indices:
        centers = np.asarray(centers, dtype=float)
        # Use searchsorted for O(N log M) nearest-index lookup
        center_idx = np.searchsorted(dim_values, centers, side='left')
        center_idx = np.clip(center_idx, 0, len(dim_values) - 1)
        # Snap to nearest: check if left neighbor is closer
        left_idx = np.maximum(center_idx - 1, 0)
        use_left = (np.abs(dim_values[left_idx] - centers)
                    < np.abs(dim_values[center_idx] - centers))
        center_idx[use_left] = left_idx[use_left]

        pre_samp = int(np.round(pre / dim_step))
        post_samp = int(np.round(post / dim_step))
    else:
        center_idx = np.asarray(centers, dtype=int)
        pre_samp = int(pre)
        post_samp = int(post)

    # Vectorized bounds check
    starts = center_idx - pre_samp
    ends = center_idx + post_samp
    oob = np.where((starts < 0) | (ends > len(dim_values)))[0]
    if len(oob) > 0:
        raise ValueError(
            f"Window(s) {oob.tolist()} are out of bounds. "
            f"Dimension '{dim}' has {len(dim_values)} samples."
        )

    # Build 2D index array: (n_epochs, window_length)
    offsets = np.arange(-pre_samp, post_samp)
    idx_2d = center_idx[:, None] + offsets[None, :]  # (N, W)

    # Vectorized extraction via np.take along the windowed axis
    dim_axis = list(data.dims).index(dim)
    raw = data.values
    windowed_values = np.take(raw, idx_2d.ravel(), axis=dim_axis)

    # np.take flattens the index, so reshape to restore (N, W)
    # Output shape: (..., N*W, ...) → (..., N, W, ...)
    shape = list(windowed_values.shape)
    shape[dim_axis:dim_axis + 1] = [len(center_idx), len(offsets)]
    windowed_values = windowed_values.reshape(shape)

    # Build output dims: insert DIM_EPOCH before the windowed dim
    # This matches the original xr.concat(..., dim=DIM_EPOCH) ordering
    new_dims = list(data.dims)
    new_dims.insert(dim_axis, DIM_EPOCH)

    # Build output coordinates
    relative_coords = offsets * dim_step
    new_coords = {
        DIM_EPOCH: np.arange(len(center_idx)),
        dim: relative_coords,
    }

    # Carry forward non-windowed coordinates (spatial coords, etc.)
    # Coordinates indexed along the windowed dimension are re-mapped:
    # their values at each center index become coordinates on DIM_EPOCH.
    for coord_name, coord_var in data.coords.items():
        if coord_name == dim or coord_name in new_coords:
            continue
        if dim in coord_var.dims:
            # Retain the coordinate value at each window centre
            center_values = coord_var.values[center_idx]
            new_coords[coord_name] = (DIM_EPOCH, center_values)
        else:
            new_coords[coord_name] = coord_var

    # Build output DataArray in one shot — no concat needed
    out_xr = xr.DataArray(
        windowed_values,
        dims=new_dims,
        coords=new_coords,
        attrs=data.attrs.copy(),
    )

    # Update description attribute for provenance
    desc_base = data.attrs.get('description', 'data')
    out_xr.attrs['description'] = (
        f"{desc_base} -> windowed along '{dim}' "
        f"({len(center_idx)} epochs, {pre_samp} pre, {post_samp} post)"
    )

    return out_xr

def window_df(data: xr.DataArray, df: pd.DataFrame|pd.Series, 
                 center_cols: str,  dim: str, pre: float|int, post: float|int, 
                 coord_cols: str|list = None, **kwargs) -> xr.DataArray:
    """
    Apply a windowing function to an xarray DataArray based on a pandas DataFrame or Series.

    Parameters
    ----------
    data : xr.DataArray
        The xarray DataArray to window.
    df : pd.DataFrame|pd.Series
        The pandas DataFrame or Series containing the windowing information.
    center_cols : str
        The column(s) in df that contain the center points for the windows.
    coord_cols : str|list, optional
        The column(s) in df that contain the coordinates for the windows.
    dim : str
        The dimension along which to window (e.g., 'time').
    pre : float|int
        A float specifying the length before each center.
    post : float|int
        A float specifying the length after each center.
    **kwargs
        Additional keyword arguments to pass to the windowing function.

    Returns
    -------
    out_xr : xr.DataArray
        A new xarray DataArray containing the windowed data.
    """

    centers = df[center_cols].values
    if coord_cols is not None:
        coords = df[coord_cols].values
    else:
        coords = None

    out_xr = window(data, dim=dim, centers=centers, pre=pre, post=post, **kwargs)

    if coords is not None:
        for i, col in enumerate(coord_cols):
            out_xr = out_xr.assign_coords({col: ((DIM_EPOCH,), coords[:, i])})
    
    # Update description attribute for provenance
    desc_base = data.attrs.get('description', 'data')
    coord_info = f", coords: {coord_cols}" if coord_cols is not None else ''
    out_xr.attrs['description'] = (
        f"{desc_base} -> windowed along '{dim}' by '{center_cols}' "
        f"({len(centers)} epochs, {pre} pre, {post} post{coord_info})"
    )

    return out_xr

def zscore(x: xr.DataArray, dim: str|list = None, bg: dict = None,
              skipna = True) -> xr.DataArray:
    """
    Z-score the data based on background mean and std.

    Parameters
    ----------
    x : xr.DataArray
        The input data array to be z-scored.
    dim : str|list, optional
        The dimensions along which to z-score. If None, z-scores
        along all dimensions. Default is None.
    bg : dict
        Dictionary containing data range used for calculating
        background mean and standard deviation. Should have keys
        for xarray dimensions and specify the slice for each dimension.
        E.g. {'time': slice(-1, 0)}.
        Default is all data.
    skipna : bool, optional
        Whether to skip NaN values when calculating the z-score.
        Default is True.

    Returns
    -------
    xz :xr.DataArray
        The z-scored data array.
    """

    if bg is None:
        bg = x
    else:
        bg = x.sel(bg)
    
    mean = bg.mean(dim=dim, skipna=skipna)
    std = bg.std(dim=dim, skipna=skipna)

    xz = (x - mean) / std

    # Propagate all attrs from input before overwriting specific keys
    xz.attrs = x.attrs.copy()

    # Update description attribute for provenance
    desc_base = x.attrs.get('description', 'data')
    if bg is not x:
        xz.attrs['description'] = f"{desc_base} -> z-scored (bg region)"
    else:
        dim_str = f"along {dim}" if dim else "globally"
        xz.attrs['description'] = f"{desc_base} -> z-scored ({dim_str})"
    
    return xz


def rolling_zscore(data: xr.DataArray, window: float, center: bool = True,
                    min_periods: int = None) -> xr.DataArray:
    """
    Z-score the data using a rolling mean and standard deviation over time.

    Parameters
    ----------
    data : xr.DataArray
        The input data array to be z-scored. Must have a `DIM_TIME`
        dimension and an `ATTR_FS` attribute.
    window : float
        The size of the rolling window in seconds, converted to samples
        using the DataArray's sampling rate.
    center : bool, optional
        Whether to center the rolling window on each sample. Default is True.
    min_periods : int, optional
        Minimum number of observations required in the window to compute
        a value. If None, defaults to the full window size.

    Returns
    -------
    xz : xr.DataArray
        The rolling z-scored data array.

    Raises
    ------
    TypeError
        If `data` is not an xr.DataArray.
    ValueError
        If `data` does not have a `DIM_TIME` dimension.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError(f"Expected xr.DataArray, got {type(data).__name__}.")
    validate_dims(data, [DIM_TIME])

    fs = get_fs(data)
    window_samples = int(round(window * fs))

    rolling = data.rolling({DIM_TIME: window_samples}, center=center,
                           min_periods=min_periods)
    mean = rolling.mean()
    std = rolling.std()
    xz = (data - mean) / std

    # Propagate all attrs from input before overwriting specific keys
    xz.attrs = data.attrs.copy()

    # Update description attribute for provenance
    desc_base = data.attrs.get('description', 'data')
    xz.attrs['description'] = f"{desc_base} -> rolling z-scored ({window}s window)"

    return xz


def interp_stack(da: xr.DataArray, coord_interp: str = COORD_Y, coord_stack: str = COORD_X,
                    spacing: float = 15, coord_min: float = None, coord_max: float = None):
    """
    Interpolate voltages along one axis and stack along the other from an
    xarray.DataArray onto a regular spatial grid.

    Iterates over the stacking coordinate using xarray.groupby, interpolates
    the data within each group onto a regular grid, and stacks the results.
    Preserves coordinates associated with the stacking dimension if they are
    consistent within groups, and preserves all coordinates associated with
    non-interpolated dimensions.

    Parameters
    ----------
    da : xarray.DataArray
        Input data. Must have coordinates specified by coord_interp and 
        coord_stack.
    coord_interp : str, optional
        Name of the coordinate to interpolate along. Default is 'ch_y'.
    coord_stack : str, optional
        Name of the coordinate to stack groups along. Default is 'ch_x'.
    spacing : float, optional
        Spatial spacing for the interpolation grid. Default is 15.
    coord_min : float, optional
        Minimum value for the interpolation grid. If None, uses the 
        minimum of the coord_interp values.
    coord_max : float, optional
        Maximum value for the interpolation grid. If None, uses the 
        maximum of the coord_interp values.

    Returns
    -------
    interp_da : xarray.DataArray
        Interpolated data with new spatial dimensions replacing the 
        original channel dimension.
    """
    validate_coords(da, [coord_interp, coord_stack])

    # Ensure both coord_interp and coord_stack live on the same source dimension
    interp_dims = da.coords[coord_interp].dims
    stack_dims = da.coords[coord_stack].dims
    
    if len(interp_dims) != 1 or len(stack_dims) != 1:
        raise ValueError("coord_interp and coord_stack must be 1-D coordinates.")
    
    channel_dim = interp_dims[0]
    if stack_dims[0] != channel_dim:
        raise ValueError(f"'{coord_interp}' and '{coord_stack}' must refer to the same dimension.")

    # Define interpolation grid
    amin = float(da.coords[coord_interp].min().values) if coord_min is None else float(coord_min)
    amax = float(da.coords[coord_interp].max().values) if coord_max is None else float(coord_max)
    interp_grid = np.arange(amin, amax + spacing, spacing)

    # Identify candidate coordinates to preserve along the stack dimension
    stack_cand_coords = [
        c for c in da.coords 
        if channel_dim in da.coords[c].dims 
        and c not in [coord_interp, coord_stack]
    ]
    stack_coords_vals = {c: [] for c in stack_cand_coords}
    valid_stack_coords = set(stack_cand_coords)

    # Interpolate each group and collect results
    interp_groups = []
    group_keys = []

    for key, group in da.groupby(coord_stack):
        # Skip groups with fewer than 2 unique points (cannot interpolate)
        if np.unique(group[coord_interp].values).size < 2:
            continue

        # Track which coordinates are constant within this group
        for c in list(valid_stack_coords):
            unique_vals = np.unique(group[c].values)
            if unique_vals.size == 1:
                stack_coords_vals[c].append(unique_vals[0])
            else:
                valid_stack_coords.discard(c)
                stack_coords_vals.pop(c, None)

        # Use interp_to_grid for each group
        group_interp = interp_to_grid(
            group, 
            coord=coord_interp, 
            new_values=interp_grid,
            dim=channel_dim,
            method='linear'
        )
        
        interp_groups.append(group_interp)
        group_keys.append(key)

    if not interp_groups:
        raise ValueError("No valid groups found for interpolation.")

    # Stack all interpolated groups along the stack coordinate
    stacked = xr.concat(interp_groups, dim=coord_stack)
    stacked = stacked.assign_coords({coord_stack: group_keys})

    # Add preserved stack coordinates
    for c in valid_stack_coords:
        stacked = stacked.assign_coords({c: (coord_stack, np.array(stack_coords_vals[c]))})

    # Preserve original attrs
    stacked.attrs = da.attrs.copy()

    # Update description attribute for provenance
    desc_base = da.attrs.get('description', 'data')
    stacked.attrs['description'] = f"{desc_base} -> interpolated along {coord_interp} and stacked by {coord_stack}"

    return stacked


# ==========================================
# DataFrame ↔ DataArray utilities
# ==========================================

import os
import warnings
from pathlib import Path


def save_dataarray_df(df: pd.DataFrame, col: str, save_dir: str,
                      name_cols: list[str], label: str) -> list[str]:
    """Save DataArrays stored in a DataFrame column as individual NetCDF files.

    For each row, a filename is constructed by joining the values of
    *name_cols* with underscores and appending ``_{label}.nc``.  Files are
    written with ``engine='netcdf4'``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where column *col* contains ``xr.DataArray`` objects.
    col : str
        Name of the column holding DataArrays.
    save_dir : str
        Directory to save files in.  Created if it does not exist.
    name_cols : list of str
        Column names whose values are joined to form each filename.
    label : str
        Descriptive suffix appended before ``.nc`` (e.g. ``'lfp'``).

    Returns
    -------
    list of str
        Absolute paths of the saved files.

    Raises
    ------
    KeyError
        If *col* or any of *name_cols* is missing from *df*.

    Examples
    --------
    >>> paths = save_dataarray_df(
    ...     results_df, col='rec', save_dir='/data/processed',
    ...     name_cols=['subject', 'session'], label='lfp',
    ... )
    """
    os.makedirs(save_dir, exist_ok=True)

    saved = []
    for idx, row in df.iterrows():
        parts = [str(row[c]) for c in name_cols]
        fname = '_'.join(parts) + f'_{label}.nc'
        fpath = os.path.join(save_dir, fname)
        row[col].to_netcdf(fpath, engine='netcdf4')
        saved.append(fpath)
    return saved


def load_dataarray_df(save_dir: str, pattern: str = '*.nc',
                      name_fields: list[str] | None = None,
                      sep: str = '_',
                      da_col: str = 'data') -> pd.DataFrame:
    """Load a directory of NetCDF files into a DataFrame with a DataArray column.

    Optionally parses filename components (before the extension) into
    metadata columns, splitting on *sep*.  This is the inverse of
    :func:`save_dataarray_df`.

    Parameters
    ----------
    save_dir : str
        Directory containing ``.nc`` files.
    pattern : str, optional
        Glob pattern for file discovery.  Default ``'*.nc'``.
    name_fields : list of str, optional
        Column names to assign to successive parts of each filename
        (after splitting on *sep* and stripping the extension).  Extra
        parts beyond ``len(name_fields)`` are ignored; missing parts
        become ``None``.  If *None* no parsing is done.
    sep : str, optional
        Separator for splitting filenames.  Default ``'_'``.
    da_col : str, optional
        Column name for the loaded DataArray.  Default ``'data'``.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per file.  Contains a *da_col* column
        holding ``xr.DataArray`` objects plus any parsed name columns.

    Examples
    --------
    >>> df = load_dataarray_df(
    ...     '/data/processed',
    ...     name_fields=['subject', 'session', 'label'],
    ... )
    """
    dirpath = Path(save_dir)
    files = sorted(dirpath.glob(pattern))

    rows = []
    for fp in files:
        da = xr.open_dataarray(fp, engine='netcdf4')
        # Load into memory so the file handle is released
        da = da.load()

        row = {da_col: da, '_file': str(fp)}

        if name_fields is not None:
            stem = fp.stem
            parts = stem.split(sep)
            for i, field in enumerate(name_fields):
                row[field] = parts[i] if i < len(parts) else None

        rows.append(row)

    return pd.DataFrame(rows)


# ==========================================
# Main: Test Functions
# ==========================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mimo_pack.testing.data import synthetic_lfp, synthetic_lfp_trials
    
    print("=" * 60)
    print("Testing xarray utility functions")
    print("=" * 60)
    
    # Generate test data
    print("\n1. Generating synthetic LFP data...")
    lfp = synthetic_lfp(duration=2.0, fs=1000, n_chans=32, seed=42)
    print(f"   Shape: {lfp.shape}")
    print(f"   Dims: {lfp.dims}")
    print(f"   Coords: {list(lfp.coords.keys())}")
    print(f"   Attrs: {lfp.attrs}")
    
    # Test get_fs
    print("\n2. Testing get_fs()...")
    fs = get_fs(lfp)
    print(f"   Sample rate: {fs} Hz")
    
    # Test validate_dims
    print("\n3. Testing validate_dims()...")
    try:
        validate_dims(lfp, [DIM_TIME, DIM_CHAN])
        print("   ✓ Passed: Required dims present")
    except ValueError as e:
        print(f"   ✗ Failed: {e}")
    
    try:
        validate_dims(lfp, ['nonexistent_dim'])
        print("   ✗ Failed: Should have raised error")
    except ValueError:
        print("   ✓ Passed: Correctly raised error for missing dim")
    
    # Test validate_coords
    print("\n4. Testing validate_coords()...")
    try:
        validate_coords(lfp, [COORD_X, COORD_Y, COORD_SHANK])
        print("   ✓ Passed: Required coords present")
    except ValueError as e:
        print(f"   ✗ Failed: {e}")
    
    # Test select_channels
    print("\n5. Testing select_channels()...")
    # Select channels on shank 0
    lfp_shank0 = select_channels(lfp, shank=0)
    print(f"   Original channels: {lfp.sizes[DIM_CHAN]}")
    print(f"   Shank 0 channels: {lfp_shank0.sizes[DIM_CHAN]}")
    
    # Select by y coordinate with tolerance
    lfp_y = select_channels(lfp, y=100, tolerance=50)
    print(f"   Channels near y=100 (±50): {lfp_y.sizes[DIM_CHAN]}")
    
    # Test interp
    print("\n6. Testing interp()...")
    new_times = np.linspace(0, 1, 500)  # 500 points over first second
    lfp_interp = interp(lfp, dim=DIM_TIME, new_coords=new_times)
    print(f"   Original time samples: {lfp.sizes[DIM_TIME]}")
    print(f"   Interpolated time samples: {lfp_interp.sizes[DIM_TIME]}")
    
    # Test resample
    print("\n7. Testing resample()...")
    lfp_downsampled = resample(lfp, new_fs=250)
    print(f"   Original fs: {get_fs(lfp)} Hz, samples: {lfp.sizes[DIM_TIME]}")
    print(f"   Downsampled fs: {get_fs(lfp_downsampled)} Hz, samples: {lfp_downsampled.sizes[DIM_TIME]}")
    
    # Test interp_to_grid
    print("\n8. Testing interp_to_grid()...")
    lfp_grid = interp_to_grid(lfp, coord=COORD_Y, spacing=50, dim=DIM_CHAN)
    print(f"   Original channel dim: {lfp.sizes[DIM_CHAN]}")
    print(f"   Grid interpolated: {lfp_grid.sizes[COORD_Y]} y-positions")
    
    # Test window
    print("\n9. Testing window()...")
    centers = np.array([0.5, 1.0, 1.5])  # Window centers in seconds
    lfp_windowed = window(lfp, dim=DIM_TIME, centers=centers, pre=0.1, post=0.1)
    print(f"   Window centers: {centers}")
    print(f"   Windowed shape: {lfp_windowed.shape}")
    print(f"   Windowed dims: {lfp_windowed.dims}")
    
    # Test zscore
    print("\n10. Testing zscore()...")
    lfp_z = zscore(lfp, dim=DIM_TIME)
    print(f"   Original mean: {float(lfp.mean()):.3f}, std: {float(lfp.std()):.3f}")
    print(f"   Z-scored mean: {float(lfp_z.mean()):.6f}, std: {float(lfp_z.std()):.3f}")
    
    # Test concat_by_attr
    print("\n11. Testing concat_by_attr()...")
    # Create multiple LFP arrays with different session attributes
    sessions = []
    for i, sess_name in enumerate(['A', 'B', 'C']):
        sess_lfp = synthetic_lfp(duration=0.5, fs=1000, n_chans=8, seed=i)
        sess_lfp.attrs['session'] = sess_name
        sess_lfp.attrs['subject'] = 'mouse1'  # Same across all
        sess_lfp.attrs['date'] = f'2025-01-0{i+1}'  # Different
        sessions.append(sess_lfp)
    
    combined = concat_by_attr(sessions, attr='session')
    print(f"   Combined shape: {combined.shape}")
    print(f"   Session coords: {combined.coords['session'].values}")
    print(f"   Preserved attrs (identical only): {combined.attrs}")
    
    # Test with keep_attrs='first'
    combined_first = concat_by_attr(sessions, attr='session', keep_attrs='first')
    print(f"   Attrs with keep_attrs='first': {combined_first.attrs}")
    
    # Test interp_stack
    print("\n12. Testing interp_stack()...")
    lfp_stacked = interp_stack(lfp, coord_interp=COORD_Y, coord_stack=COORD_X, spacing=25)
    print(f"   Original shape: {lfp.shape}")
    print(f"   Stacked shape: {lfp_stacked.shape}")
    print(f"   Stacked dims: {lfp_stacked.dims}")
    
    # Generate epoch data for additional tests
    print("\n13. Testing with epoch data...")
    trials = synthetic_lfp_trials(n_trials=10, fs=500, n_chans=16, 
                                   pre=0.2, post=0.5, seed=42)
    print(f"   Epoch data shape: {trials.shape}")
    print(f"   Epoch data dims: {trials.dims}")
    
    # Z-score with baseline
    trials_z = zscore(trials, dim=DIM_TIME, bg={DIM_TIME: slice(-0.1, 0)})
    print(f"   Z-scored epochs shape: {trials_z.shape}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
    
    # Optional: Create visualization
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    
    # Plot 1: Original LFP
    ax = axes[0, 0]
    lfp.isel(channel=0).plot(ax=ax)
    ax.set_title('Original LFP (channel 0)')
    
    # Plot 2: Downsampled LFP
    ax = axes[0, 1]
    lfp_downsampled.isel(channel=0).plot(ax=ax)
    ax.set_title(f'Downsampled to {get_fs(lfp_downsampled)} Hz')
    
    # Plot 3: Windowed data
    ax = axes[1, 0]
    lfp_windowed.isel(channel=0).T.plot(ax=ax)
    ax.set_title('Windowed LFP (3 windows)')
    
    # Plot 4: Z-scored data
    ax = axes[1, 1]
    lfp_z.isel(channel=0).plot(ax=ax)
    ax.set_title('Z-scored LFP')
    
    # Plot 5: Concatenated sessions
    ax = axes[2, 0]
    for i, sess in enumerate(combined.coords['session'].values):
        combined.sel(session=sess).isel(channel=0).plot(ax=ax, label=f'Session {sess}')
    ax.legend()
    ax.set_title('Concatenated sessions')
    
    # Plot 6: Interpolated grid
    ax = axes[2, 1]
    lfp_stacked.isel(time=0).plot(ax=ax)
    ax.set_title('Interpolated spatial grid (t=0)')
    
    plt.tight_layout()
    plt.savefig('xarray_util_tests.pdf')
    print("\nPlot saved to: xarray_util_tests.pdf")
    plt.show()
