---
description: "Use when tasks involve in-vitro electrophysiology analysis: loading ABF recordings, formatting by stimulation protocol (IV/IC, focal stim, pattern stim, ramp, tonic LED), cleanup of spikes/offsets/line noise, or extracting passive/active cell properties and opsin kinetics."
---

# In-Vitro Electrophysiology Analysis Reference

This instruction intentionally has no `applyTo` field.
It is a reference-only surface that should load on demand by topic keywords rather than auto-apply by file path.

## Decision Tree

1. Working with a directory of `.abf` files from one or more cells?
   - Use `invitro_files_df(rootdir)` to enumerate files, `extract_protocol_files()` to group by protocol, and `summarize_invitro_protocols()` for an overview before analyzing individual recordings.
2. Loading a single recording?
   - Use `load_abf(abf_file_path)` for the trace data and `load_abf_settings(abf_file_path)` / `get_clamp_mode()` / `summarise_clamp_mode()` for protocol metadata (current-clamp vs. voltage-clamp, step structure).
3. Which `format_*` function matches the protocol?
   - IV/IC step protocol: `format_ivic(rec)`.
   - Focal/spot optogenetic stimulation: `format_focal_stimulation()`.
   - Patterned/spatial stimulation: `format_pattern_stim(patt_xr)`.
   - Variable-amplitude ramp: `format_var_ramp(vr_xr)`.
   - Blue-light tonic LED stimulation: `format_blue_tonic_led()`.
   - If the protocol name/shape is ambiguous, ask the user which `format_*` applies rather than guessing.
4. Need to isolate subthreshold membrane dynamics?
   - Use `remove_ap(abf_xr)` to strip detected action potentials before further analysis.
5. Seeing a DC offset or line noise in the trace?
   - Use `remove_current_offset()` / `remove_line_noise()`; use `get_holding_potential()` to confirm the expected baseline first.
6. Extracting cell properties?
   - Passive and active IV/IC properties: `extract_iv_properties(iv_xr)` (resting potential, input resistance, AP waveform/rate properties, sag ratio, etc.).
   - Optogenetic current kinetics: `fit_opsin_response()` -> `opsin_iv_conductance()`.

## Pipeline Anchors

- **Survey:** `invitro_files_df()` -> `extract_protocol_files()` -> `summarize_invitro_protocols()`.
- **Load:** `load_abf()` + `load_abf_settings()` / `get_clamp_mode()`.
- **Format (protocol-specific):** `format_ivic()` / `format_focal_stimulation()` / `format_pattern_stim()` / `format_var_ramp()` / `format_blue_tonic_led()`.
- **Clean:** `remove_ap()`, `remove_current_offset()`, `remove_line_noise()`, `get_holding_potential()`.
- **Extract properties:** `extract_iv_properties()` or `fit_opsin_response()` -> `opsin_iv_conductance()`.
- **Plot:** `stim_pattern_map()`, `focal_stim_map()`, `violinplot_log()` for property distributions.

## Directives

- Keep recordings as `xarray.DataArray`; do not convert to raw numpy before the `format_*` step, since those functions rely on protocol-specific coordinates.
- Confirm clamp mode (`get_clamp_mode()`) before choosing a `format_*` or property-extraction function; current-clamp and voltage-clamp protocols are not interchangeable.
- In-vitro datasets are organized subject -> slice -> cell -> protocol files; see `.github/instructions/data-loading-time-alignment.instructions.md` for the layout convention.
