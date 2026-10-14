---
description: "Use when creating a new analysis notebook, extending an existing notebook, scaling notebook workflows from single session to batch, or editing files under project/, templates/, or *.ipynb. Covers notebook decision trees, Session-based loading, save/export rules, and when to load shared data-loading or utility references."
applyTo:
  - "**/*.ipynb"
  - "project/**"
  - "templates/**"
---

# Notebook Analysis Workflow

## Decision Tree

1. Is this a new analysis notebook?
   - Confirm scope, data type, preprocessing, data selection, combining strategy, visualization, and statistics.
   - If any item is unclear, ask once with every unresolved point grouped together.
2. Is this an edit to an existing notebook?
   - Decide whether the change is scaling to batch, a local analysis/parameter/visualization edit, or the introduction of a new data type.
3. Is the task processing data collected within the lab or public datasets.
   - If public, confirm the data source and load `.github/instructions/public-datasets.instructions.md` for Allen Institute Neuropixels or IBL data.
   - If local, confirm the data type and recording system.
4. Does the task require loading multiple files from the same session or mention SpikeGLX, CatGT, dclut, imec NIDQ, Phy, Kilosort, or sync pulses?
   - Also consult `.github/instructions/data-loading-time-alignment.instructions.md`.
5. Does the task need xarray/pynapple interop, fs lookup, epoch handling, or helper discovery?
   - Also consult `.github/instructions/interop-utilities.instructions.md`.
6. Does the task involve a specific data type or experiment? Also consult the matching reference for which functions to use and how to sequence them:
   - Spike/single-unit (Phy, Kilosort, cross-correlograms, antidromic ID): `.github/instructions/analysis-unit.instructions.md`.
   - LFP (spectral, CSD, evoked, behavioral state): `.github/instructions/analysis-lfp.instructions.md`.
   - Fiber photometry (Doric, dF/F): `.github/instructions/analysis-fiber-photometry.instructions.md`.
   - In-vitro electrophysiology (ABF, IV/IC, opsin kinetics): `.github/instructions/analysis-invitro-ephys.instructions.md`.
   - Behavioral task events / video (NIDQ digital-analog events, ROI tracking, alignment): `.github/instructions/analysis-behavior-task-events.instructions.md`.
7. Does the request actually change code inside `mimo_pack/`? Confirm with the user whether they want a library change.
   - Switch to `.github/instructions/library-core.instructions.md` and the library agent path instead of treating it as notebook-only work.

## Directives

- Structure notebooks as Setup -> Data Loading -> Preprocessing -> Analysis -> Summary Visualization -> Save.
- Incorporate visualizations and data checks throughout the notebook to verify each step before moving on.
- When using datasets with `mimo_pack.core.Session`, use it for config and path discovery. Do not pass `Session` objects into analysis functions.
- Load through `mimo_pack.fileio.*`, analyze through `mimo_pack.analysis.*`, and plot through `mimo_pack.plot.*` or focused matplotlib calls.
- New notebooks and ad hoc scripts belong in `project/`, never inside `mimo_pack/`.
- Do not guess unresolved analysis design choices; ask for them together before writing code.

## Recommendations

- Anchor new work to the closest notebook in `templates/`.
- Verify loaded data early by printing shapes, attrs, and time bounds before downstream analysis.
- Prefer `mimo_pack.util.xarray.window`, `window_df`, `zscore`, `resample`, and `concat_by_attr` over bespoke helpers.
- Keep notebook prose, comments, and variable names in American English.

## Actions

- Keep a dedicated Save cell at the end of the notebook.
- Save figures as PDF only; do not also export PNG.
- Save `xarray.DataArray` outputs as `.nc` with `engine='netcdf4'`.
- If the notebook uses SpikeGLX or spike-sorting data, confirm that inputs are dclut-backed and time-aligned before analysis.

## Pipeline Anchors

Condensed anchors; each matching reference above has the full sequence, QC steps, and decision points:

- LFP spectral: `load_lfp_full() -> stft() -> fit_spectrum_aperiodic() -> wavelet() -> amp_map()`. See `.github/instructions/analysis-lfp.instructions.md`.
- Fiber photometry: `load_doric() -> isosbestic_baseline() / estimate_baseline() -> df_over_f() -> stacked_lines()`. See `.github/instructions/analysis-fiber-photometry.instructions.md`.
- In-vitro IV curve: `load_abf() -> format_ivic() -> extract_iv_properties()`. See `.github/instructions/analysis-invitro-ephys.instructions.md`.
- Cross-correlation: `load_phy_spikes() -> nap.compute_crosscorrelogram() -> jitter/statistical follow-up`. See `.github/instructions/analysis-unit.instructions.md`.
- Evoked response and CSD: load -> epoch -> `evoked_lfp()` / `evoked_ap()` -> `csd()`. See `.github/instructions/analysis-unit.instructions.md` and `.github/instructions/analysis-lfp.instructions.md`.
- Behavioral task events: `load_nidq() -> digital_to_events()/analog_to_events() -> pulse_stats()`, or video ROI tracking + alignment. See `.github/instructions/analysis-behavior-task-events.instructions.md`.