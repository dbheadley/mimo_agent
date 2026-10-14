---
description: "Use when tasks involve LFP analysis: loading SpikeGLX LFP, filtering, line-noise removal, spectral analysis (STFT/wavelet/aperiodic fit), CSD, evoked LFP, spike-phase coupling, or LFP-derived behavioral states."
---

# LFP Analysis Reference

This instruction intentionally has no `applyTo` field.
It is a reference-only surface that should load on demand by topic keywords rather than auto-apply by file path.

## Decision Tree

1. Loading LFP from a SpikeGLX session?
   - Use `load_lfp_full()` / `load_lfp_intervals()` / `load_lfp_windows()` with dclut paths. Also consult `.github/instructions/data-loading-time-alignment.instructions.md`.
2. No LFP file exists yet, only a raw AP binary?
   - Derive it first: `make_lfp_file_spikeglx(bin_path)` (writes the LFP binary via `calc_lfp()`), then `make_lfp_dclut()` / `make_lfp_file_dclut()` / `dclut_from_meta_lfp()` to create the lookup table the loaders expect.
3. Seeing 50/60 Hz line noise or other narrowband contamination?
   - Use `remove_line_noise()` (preprocess) or `notch_filter()` (analysis) depending on whether you are building the LFP file or filtering an already-loaded `DataArray`.
4. Need band power or time-frequency structure?
   - Use the spectral pipeline below, not a hand-written FFT loop.
5. Need current source density across channels?
   - Use `csd()`; requires LFP with `COORD_Y` channel-depth coordinates intact.
6. Need an evoked/stimulus-triggered LFP response?
   - Epoch first, then `evoked_lfp()`. For a joint AP+LFP evoked view (e.g. antidromic stimulation), also consult `.github/instructions/analysis-unit.instructions.md`.
7. Classifying behavioral state (e.g. sleep/wake, movement) from LFP bands?
   - Use `states_lfp_only()` or `states_lfp_mot()` (if motion/video data is available), then `add_states()` to attach the result to other data.
8. Matching a unit or probe channel to the nearest LFP channel?
   - Use `find_lfp_channel()`.

## Pipeline Anchors

- **Load:** `load_lfp_full()` / `load_lfp_intervals()` / `load_lfp_windows()` -> `xarray.DataArray` with `DIM_TIME`, `DIM_CHAN`, `COORD_X`/`COORD_Y`/`COORD_SHANK`.
- **Build (if missing):** `calc_lfp(raw_data, fs)` / `make_lfp_file_spikeglx()` -> `make_lfp_dclut()` / `dclut_from_meta_lfp()`.
- **Filtering:** `lp_filter()`, `hp_filter()`, `notch_filter()`, `remove_line_noise()`.
- **Spectral:** `stft()` -> `fit_spectrum_aperiodic()` (band power minus the aperiodic 1/f component); `wavelet()` -> `wavelet_phamp()` (phase/amplitude) -> `compute_spike_phamp_hist(spks, ...)` for spike-phase coupling against `mimo_pack.analysis.ap` unit times.
- **CSD:** `csd(lfp, coord=COORD_Y)`.
- **Evoked:** epoch -> `evoked_lfp(lfp_trials)`.
- **Behavioral states:** `states_lfp_only()` / `states_lfp_mot()` -> `add_states()` -> `plot_band_power_states()`.
- **Plotting:** `amp_map()` / `depth_lines()` for channel-depth views, `stacked_lines()` for raw traces, `unit_lfp_mapping()` to overlay units.

## Directives

- Keep LFP as `xarray.DataArray`; preserve `COORD_X`/`COORD_Y`/`COORD_SHANK` through every transform (required by `csd()` and the depth-aware plots).
- Prefer loading pre-existing LFP/dclut files over regenerating them; only build new LFP files when the user's session genuinely lacks one.
- Extract `fs` from `data.attrs[ATTR_FS]`; never hardcode a sampling rate. See `.github/instructions/interop-utilities.instructions.md`.
