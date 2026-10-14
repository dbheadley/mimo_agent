"""Generic DataFrame utilities for row-wise function application and pipelines.

These functions operate on arbitrary object columns in a pandas DataFrame,
providing per-row error handling, multi-column input/output, and processing
tree semantics.  They have no dependency on xarray or any other mimo_pack
module.
"""

import warnings

import pandas as pd


def _apply_row_wise(df, cols_list, fn, result_cols_list, errors,
                    args=(), kwargs=None, skip_indices=None, label=''):
    """Apply *fn* row-wise and collect results.

    Returns ``(result_lists, failed)`` where *result_lists* maps each
    output column name to a list of per-row values, and *failed* is the
    set of row indices where *fn* raised an exception.

    Rows in *skip_indices* are silently set to ``None`` without calling
    *fn* and are **not** included in *failed*.
    """
    if kwargs is None:
        kwargs = {}
    if skip_indices is None:
        skip_indices = set()

    single_output = len(result_cols_list) == 1
    result_lists = {rc: [] for rc in result_cols_list}
    failed = set()

    for idx, row in df.iterrows():
        if idx in skip_indices:
            for rc in result_cols_list:
                result_lists[rc].append(None)
            continue

        try:
            input_vals = [row[c] for c in cols_list]
            result = fn(*input_vals, *args, **kwargs)

            if single_output:
                result_lists[result_cols_list[0]].append(result)
            else:
                if not isinstance(result, tuple) or len(result) != len(result_cols_list):
                    raise ValueError(
                        f"fn must return a tuple of length {len(result_cols_list)} "
                        f"for result_cols={result_cols_list}, got {type(result).__name__}"
                        + (f" of length {len(result)}" if isinstance(result, tuple) else "")
                    )
                for rc, val in zip(result_cols_list, result):
                    result_lists[rc].append(val)
        except Exception as exc:
            if errors == 'raise':
                raise
            if errors == 'warn':
                warnings.warn(f"Row {idx}{label}: {exc}")
            for rc in result_cols_list:
                result_lists[rc].append(None)
            failed.add(idx)

    return result_lists, failed


def apply_to_columns(df: pd.DataFrame, cols: 'str | list[str]', fn,
                     *args, result_cols: 'str | list[str] | None' = None,
                     errors: str = 'warn',
                     **kwargs) -> pd.DataFrame:
    """Apply a function row-wise to one or more DataFrame columns.

    For each row, values are read from *cols* and passed as positional
    arguments to *fn* (before any extra ``*args``).  The return value is
    stored in *result_cols*.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cols : str or list of str
        Column name(s) whose values are passed to *fn*.  A single string
        is treated as one column; a list provides multiple positional
        arguments to *fn*.
    fn : callable
        Function applied per row.

        * Single input column:  ``fn(value, *args, **kwargs) -> result``
        * Multiple input columns: ``fn(val1, val2, ..., *args, **kwargs) -> result``
        * Single output column: *fn* returns one value.
        * Multiple output columns: *fn* returns a **tuple** whose
          elements map positionally to *result_cols*.
    *args
        Extra positional arguments forwarded to *fn* (after column values).
    result_cols : str or list of str, optional
        Column name(s) for the output.  Defaults to *cols*.  When a list
        is provided, *fn* must return a tuple of the same length.
    errors : {'warn', 'raise', 'skip'}
        How to handle per-row exceptions.

        * ``'warn'`` — emit a warning, set result(s) to ``None``.
        * ``'raise'`` — re-raise immediately.
        * ``'skip'`` — silently set result(s) to ``None``.
    **kwargs
        Extra keyword arguments forwarded to *fn*.

    Returns
    -------
    pd.DataFrame
        A **copy** of *df* with the result column(s) added or overwritten.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': [1, 2, 3]})
    >>> apply_to_columns(df, 'x', lambda v: v * 2, result_cols='doubled')
       x  doubled
    0  1        2
    1  2        4
    2  3        6

    Multi-input / multi-output:

    >>> df = pd.DataFrame({'a': [1, 2], 'b': [10, 20]})
    >>> apply_to_columns(df, ['a', 'b'],
    ...                  lambda a, b: (a + b, a * b),
    ...                  result_cols=['sum', 'prod'])
       a   b  sum  prod
    0  1  10   11    10
    1  2  20   22    40
    """
    if errors not in ('warn', 'raise', 'skip'):
        raise ValueError(f"errors must be 'warn', 'raise', or 'skip', got '{errors}'")

    # Normalise cols / result_cols to lists
    cols_list = [cols] if isinstance(cols, str) else list(cols)

    if result_cols is None:
        result_cols_list = cols_list
    elif isinstance(result_cols, str):
        result_cols_list = [result_cols]
    else:
        result_cols_list = list(result_cols)

    result_lists, _ = _apply_row_wise(
        df, cols_list, fn, result_cols_list, errors,
        args=args, kwargs=kwargs,
    )

    out = df.copy()
    for rc in result_cols_list:
        out[rc] = result_lists[rc]
    return out


def pipe_columns(df: pd.DataFrame,
                 steps: list,
                 errors: str = 'warn') -> pd.DataFrame:
    """Apply a sequence of row-wise functions, building a processing tree.

    Each step reads from one or more columns (original or produced by an
    earlier step), applies a function, and stores the result in one or
    more new columns.  Steps execute in order; any step may read from
    any column available at that point.

    When a step fails for a given row, only **dependent** downstream
    steps (those whose inputs include the failed output) are
    automatically skipped.  Independent branches continue executing.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    steps : list of list
        Each inner list is ``[input_cols, fn, output_cols]``:

        * **input_cols** : ``str`` or ``list[str]`` — column(s) to read.
        * **fn** : callable — receives values from *input_cols* as
          positional arguments and returns a single value (if
          *output_cols* is a ``str`` or length-1 list) or a tuple (if
          *output_cols* is a multi-element list).
        * **output_cols** : ``str`` or ``list[str]`` — where to store
          results.
    errors : {'warn', 'raise', 'skip'}
        How to handle per-row exceptions.

        * ``'warn'`` — emit a warning and mark outputs as failed (``None``).
        * ``'raise'`` — re-raise immediately.
        * ``'skip'`` — silently mark outputs as failed (``None``).

    Returns
    -------
    pd.DataFrame
        A **copy** of *df* with one new column per unique output name.

    Raises
    ------
    ValueError
        If *steps* is empty, a step has the wrong length, *errors* is
        invalid, or an input column is not reachable.
    TypeError
        If *steps* is not a list.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'x': [1, 2, 3]})

    Linear pipeline:

    >>> pipe_columns(df, [
    ...     ['x',       lambda v: v * 2,  'doubled'],
    ...     ['doubled', lambda v: v + 1,  'plus_one'],
    ... ])
       x  doubled  plus_one
    0  1        2         3
    1  2        4         5
    2  3        6         7

    Branching (both read from 'x'):

    >>> pipe_columns(df, [
    ...     ['x', lambda v: v * 2,  'branch_a'],
    ...     ['x', lambda v: v + 10, 'branch_b'],
    ... ])
       x  branch_a  branch_b
    0  1         2        11
    1  2         4        12
    2  3         6        13

    Multi-input / multi-output:

    >>> df = pd.DataFrame({'a': [1, 2], 'b': [10, 20]})
    >>> pipe_columns(df, [
    ...     [['a', 'b'], lambda a, b: (a + b, a * b), ['sum', 'prod']],
    ... ])
       a   b  sum  prod
    0  1  10   11    10
    1  2  20   22    40
    """
    if errors not in ('warn', 'raise', 'skip'):
        raise ValueError(
            f"errors must be 'warn', 'raise', or 'skip', got '{errors}'"
        )
    if not isinstance(steps, list):
        raise TypeError(f"steps must be a list, got {type(steps).__name__}")
    if not steps:
        raise ValueError("steps list must not be empty")

    # --- Normalise steps to (input_list, fn, output_list) tuples -------
    normalised = []
    for i, entry in enumerate(steps):
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise ValueError(
                f"Each step must be a list/tuple of length 3 "
                f"[input_cols, fn, output_cols], got length {len(entry) if isinstance(entry, (list, tuple)) else '?'} at index {i}"
            )
        in_cols, fn, out_cols = entry
        in_list = [in_cols] if isinstance(in_cols, str) else list(in_cols)
        out_list = [out_cols] if isinstance(out_cols, str) else list(out_cols)
        normalised.append((in_list, fn, out_list))

    # Validate that source columns are reachable
    available = set(df.columns)
    for in_list, _fn, out_list in normalised:
        for c in in_list:
            if c not in available:
                raise ValueError(
                    f"Step producing {out_list} references input column '{c}' "
                    f"which is not in the DataFrame and is not produced by a "
                    f"previous step."
                )
        available.update(out_list)

    # --- Execute step by step, tracking failed rows for cascade --------
    out = df.copy()
    failed_rows = {}  # col_name -> set of row indices that failed

    for in_list, fn, out_list in normalised:
        # Collect row indices that must be skipped (upstream failure)
        skip = set()
        for c in in_list:
            if c in failed_rows:
                skip |= failed_rows[c]

        result_lists, step_failed = _apply_row_wise(
            out, in_list, fn, out_list, errors,
            skip_indices=skip,
            label=f", step producing {out_list}",
        )

        all_failed = skip | step_failed
        for c in out_list:
            out[c] = result_lists[c]
            if all_failed:
                failed_rows[c] = all_failed

    return out


def pipe_groupby_reduce(
    df: pd.DataFrame,
    by: str | list[str],
    steps: list,
    errors: str = 'warn',
) -> pd.DataFrame:
    """Group a DataFrame and reduce each group with a sequence of steps.

    For each group defined by *by*, values from one or more columns are
    collected into a **list** and passed to *fn*, which must return a
    single reduced value.  The result is a DataFrame with one row per
    group.

    Non-step columns whose values are identical within a group are
    automatically carried forward.  Columns that vary within a group
    are dropped (unless they appear as a step output).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    by : str or list of str
        Column name(s) to group by.  These become the index of the
        output DataFrame.
    steps : list of list
        Each inner list is ``[input_cols, fn, output_cols]``:

        * **input_cols** : ``str`` or ``list[str]`` — column(s) whose
          values are collected into lists and passed to *fn*.  A single
          string provides one list; multiple strings provide multiple
          positional list arguments.
        * **fn** : callable — receives one list per input column and
          returns a single value (if *output_cols* is a ``str`` or
          length-1 list) or a tuple (if *output_cols* is a
          multi-element list).
        * **output_cols** : ``str`` or ``list[str]`` — where to store
          results.
    errors : {'warn', 'raise', 'skip'}
        How to handle per-group exceptions.

        * ``'warn'`` — emit a warning, set outputs to ``None``.
        * ``'raise'`` — re-raise immediately.
        * ``'skip'`` — silently set outputs to ``None``.

    Returns
    -------
    pd.DataFrame
        One row per group, indexed by *by*, with one column per unique
        output name plus any constant non-step columns.

    Raises
    ------
    ValueError
        If *steps* is empty, a step has the wrong length, or *errors*
        is invalid.
    TypeError
        If *steps* is not a list.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'grp': ['a', 'a', 'b', 'b'],
    ...     'x': [1, 2, 3, 4],
    ...     'label': ['A', 'A', 'B', 'B'],
    ... })
    >>> pipe_groupby_reduce(df, 'grp', [
    ...     ['x', lambda vals: sum(vals), 'total'],
    ... ])
         total label
    grp
    a        3     A
    b        7     B
    """
    if errors not in ('warn', 'raise', 'skip'):
        raise ValueError(
            f"errors must be 'warn', 'raise', or 'skip', got '{errors}'"
        )
    if not isinstance(steps, list):
        raise TypeError(f"steps must be a list, got {type(steps).__name__}")
    if not steps:
        raise ValueError("steps list must not be empty")

    by_list = [by] if isinstance(by, str) else list(by)

    # Normalise steps
    normalised = []
    for i, entry in enumerate(steps):
        if not isinstance(entry, (list, tuple)) or len(entry) != 3:
            raise ValueError(
                f"Each step must be [input_cols, fn, output_cols], "
                f"got length {len(entry) if isinstance(entry, (list, tuple)) else '?'} "
                f"at index {i}"
            )
        in_cols, fn, out_cols = entry
        in_list = [in_cols] if isinstance(in_cols, str) else list(in_cols)
        out_list = [out_cols] if isinstance(out_cols, str) else list(out_cols)
        normalised.append((in_list, fn, out_list))

    # Identify all step input/output columns
    step_in_cols = set()
    step_out_cols = set()
    for in_list, _, out_list in normalised:
        step_in_cols.update(in_list)
        step_out_cols.update(out_list)

    # Identify non-step columns that are constant within each group
    non_step_cols = [c for c in df.columns
                     if c not in by_list
                     and c not in step_in_cols
                     and c not in step_out_cols]

    result_rows = []
    for key, grp in df.groupby(by if isinstance(by, str) else by_list, sort=False):
        if isinstance(by, str):
            row = {by: key}
        else:
            row = dict(zip(by_list, key))

        # Carry forward constant columns
        for c in non_step_cols:
            try:
                vals = grp[c].unique()
            except TypeError:
                # unhashable types (e.g. DataArray) — skip column
                continue
            if len(vals) == 1:
                row[c] = vals[0]

        # Apply each step
        for in_list, fn, out_list in normalised:
            single_output = len(out_list) == 1
            try:
                input_lists = [grp[c].tolist() for c in in_list]
                result = fn(*input_lists)
                if single_output:
                    row[out_list[0]] = result
                else:
                    if not isinstance(result, tuple) or len(result) != len(out_list):
                        raise ValueError(
                            f"fn must return a tuple of length {len(out_list)}, "
                            f"got {type(result).__name__}"
                        )
                    for oc, val in zip(out_list, result):
                        row[oc] = val
            except Exception as exc:
                if errors == 'raise':
                    raise
                if errors == 'warn':
                    warnings.warn(f"Group {key}: {exc}")
                for oc in out_list:
                    row[oc] = None

        result_rows.append(row)

    out = pd.DataFrame(result_rows)
    if by_list[0] in out.columns:
        out = out.set_index(by_list if len(by_list) > 1 else by_list[0])
    return out
