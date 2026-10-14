---
description: "Use when tasks involve single-unit or multi-unit spike analysis: Kilosort/Phy spike loading, waveform QC, unit classification, cross-correlograms, burst/point-process structure, antidromic identification, or evoked spiking."
---

# Unit (Spike) Analysis Reference

This instruction intentionally has no `applyTo` field.
It is a reference-only surface that should load on demand by topic keywords rather than auto-apply by file path.

## Decision Tree

1. Loading spikes from a Kilosort/Phy session?
   - Use `mimo_pack.fileio.loadphy.load_phy_spikes(phy_dir, dcl_file=AP_DCLUT)` or `as_pynapple(...)` so times are synchronized seconds, not sample indices. Also consult `.github/instructions/data-loading-time-alignment.instructions.md`.
2. Checking unit quality before analysis?
   - Use the waveform-metrics and QC pipeline below. Do not analyze units that have not passed QC unless the user asks for raw/uncurated units.
3. Looking for functional connectivity between units?
   - Use the cross-correlogram pipeline below, not a hand-written histogram.
4. Looking for rhythmic or bursty firing structure?
   - Use `fit_point_process_gmm()`.
5. Identifying antidromically activated units from stimulation?
   - Use the antidromic pipeline below; it needs both AP and LFP evoked data, so also consult `.github/instructions/analysis-lfp.instructions.md` for the LFP side.
6. Relating a unit to nearby LFP channels or probe geometry?
   - Use `nearest_grid()` / `find_lfp_channel()` from `mimo_pack.analysis.probe`.

## Pipeline Anchors

- **Load:** `load_phy_spikes(phy_dir, dcl_file=AP_DCLUT)` / `as_pynapple(...)` -> `pynapple.TsGroup`.
- **Waveform metrics and QC:** `sample_waveforms()` / `mean_waveform()` -> `waveform_peak()`, `waveform_halfwidth()`, `waveform_peak_to_trough()`, `waveform_snr()`, `waveform_spread()` -> `add_waveform_metrics(spks, dcl_file)` -> `classify_unit(frate, halfwidth, region)` (e.g. regular-spiking vs. fast-spiking) -> `run_qc_report(spks, report_path)` to emit the per-unit PDFs under `kilosort4/qc_reports/`.
- **Spike-train quality:** `unit_occupancy()`, `unit_refractory_violations()`, `amp_cutoff()` — use these to flag or exclude unstable/contaminated units.
- **Cross-correlation / connectivity:** `nap.compute_crosscorrelogram()` -> `cc_to_prob()` -> `cc_correct_poly()` -> `ccg_convolution_significance()` -> `find_significant_connections()`.
- **Burst / point-process structure:** `fit_point_process_gmm()`.
- **Antidromic identification:** `get_antistim_times()` (preprocess, from NIDQ stim channel) -> `create_stimulation_summary()` -> `fit_antidromic_gaussians()` (spike-latency fit) and/or `get_antidromic_evoked_summary(ap_dclut, lfp_dclut, stim_times)` (joint AP+LFP evoked view) -> `plot_antidromic_summary()`.
- **Evoked spiking:** epoch spikes around an event, then `evoked_ap(ap_trials)`.
- **Plotting:** `probe_units()` for a probe-wide raster/map, `unit_lfp_mapping()` to relate units to LFP channels, `wave_map()`/`amp_map()` for waveform topography.

## Directives

- Keep spikes as `pynapple.TsGroup`; do not convert to raw sample-index arrays for analysis.
- Run waveform QC (`add_waveform_metrics` + `run_qc_report`) before trusting unit classification or connectivity results on a new session.
- Treat `_sas`-suffixed Phy outputs (e.g. `spike_times_sas.npy`) as the manually curated source of truth when present; see `.github/instructions/data-neuropixel-directory.instructions.md`.
