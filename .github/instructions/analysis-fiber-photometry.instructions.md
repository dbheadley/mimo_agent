---
description: "Use when tasks involve fiber photometry analysis: loading Doric recordings, isosbestic or exponential-decay baseline correction, or computing dF/F."
---

# Fiber Photometry Analysis Reference

This instruction intentionally has no `applyTo` field.
It is a reference-only surface that should load on demand by topic keywords rather than auto-apply by file path.

## Decision Tree

1. Loading a `.doric` recording?
   - Use `load_doric(doric_file_path)`; it returns the fluorescence channel(s) as `xarray.DataArray`. For very large files or sync-aligned access, use `doric_to_binary_dclut()` first.
2. Choosing a baseline/bleaching-correction method?
   - An isosbestic (405 nm) control channel was recorded: use `isosbestic_baseline(signal, isosbestic)` — this is the preferred method when available.
   - No isosbestic channel, but the signal shows smooth photobleaching decay: use `exponential_decay_baseline(signal, model='biexponential')`.
   - No isosbestic channel and bleaching is not a smooth exponential (e.g. drift, slow modulation): use `estimate_baseline(signal, method='median', window=...)`.
   - If unsure which applies, ask the user rather than guessing, since the choice changes the scientific interpretation of dF/F.
3. Computing the normalized signal?
   - Use `df_over_f(signal, baseline)` once a baseline has been chosen.

## Pipeline Anchors

- `load_doric(doric_file_path)` -> choose baseline method (`isosbestic_baseline()` / `exponential_decay_baseline()` / `estimate_baseline()`) -> `df_over_f(signal, baseline)` -> `stacked_lines()` or `plot_mean_shaded()` for visualization.

## Directives

- Keep both the raw signal and the isosbestic/control channel (when present) as `xarray.DataArray` through baseline fitting; do not drop one to a raw numpy array before `isosbestic_baseline()`.
- State which baseline method was used and why in notebook prose; this is a scientific choice, not an implementation detail.
- Do not hardcode sampling rate or channel names; use `data.attrs[ATTR_FS]` and the channel coordinates returned by `load_doric()`.
