---
description: "Use when tasks involve xarray and pynapple interop, fs lookup, epoch handling, channel selection, resampling, windowing, z-scoring, interpolation, or deciding whether a mimo_pack utility already exists."
---

# Interop And Utility Discovery

This instruction intentionally has no `applyTo` field.
It is a reference-only surface that should load on demand by topic keywords rather than auto-apply by file path.

## Decision Tree

1. Need the sampling rate?
   - Prefer `data.attrs[ATTR_FS]` or `mimo_pack.util.xarray.get_fs()`.
2. Need epoch/window conversion?
   - Use `window()`, `window_df()`, or `normalize_to_epoch()` instead of manual looping.
3. Need channel or spatial selection?
   - Use xarray `.sel()` / `.isel()` or `select_channels()` instead of positional numpy indexing.
4. Need alignment between continuous data and spikes?
   - Keep one side rich and use xarray or pynapple methods. Do not strip both to numpy arrays.
5. Need resampling, interpolation, z-scoring, or concatenation?
   - Check `resample()`, `interp()`, `interp_stack()`, `zscore()`, and `concat_by_attr()` before writing a helper.

## Directives

- Continuous data stays in `xarray.DataArray`.
- Spike collections stay in `pynapple.TsGroup`.
- Epochs stay in `pynapple.IntervalSet`.
- Never manually align rich objects by converting both sides to raw numpy arrays.
- Read the relevant utility docstring before adding a new helper.

## Utility Quick Picks

- `get_fs()` for sampling-rate lookup.
- `validate_dims()` and `validate_coords()` for defensive validation.
- `normalize_to_epoch()` for stimulus-dimension normalization.
- `select_channels()` for x/y/shank filtering.
- `window()` and `window_df()` for epoch extraction.
- `zscore()` for standardized normalization.
- `resample()` and `interp()` for axis-aware resampling/interpolation.
- `concat_by_attr()` for combining repeated outputs.