---
description: "Use when tasks ask about Neuropixels/SpikeGLX session directory layout, file naming conventions, probe subfolder structure, CatGT/tcat outputs, Kilosort/kilosort4 folders, curated spike files, or which files live at session root vs. per-probe."
---

This instruction intentionally has no `applyTo` field.
It is a reference-only surface that should load on demand by topic keywords rather than auto-apply by file path.

# Neuropixels/SpikeGLX Session Directory Instructions

## Overview
This file describes the directory layout conventions used for Neuropixels/SpikeGLX recording sessions (including CatGT processing and Kilosort outputs). Use these rules when parsing or analyzing neuropixel data paths in the `mimo_pack` library or analysis notebooks.

## 1. General Naming Convention
- **Root Directory:** Uses the SpikeGLX convention `<Date>_<Subject>_<Task>_g0`.
- **Behavior/Video Directory:** Typically exists alongside probe folders, matching the root name but without the `_g0` suffix (e.g., `<Date>_<Subject>_<Task>`). Contains the originally acquired `.tif` images (in an `images/` subfolder), an assembled video file (`.avi` and/or `.mp4`), a `.csv` file with frame-by-frame information, and a `qc_reports/` subfolder (e.g., `aligned_video_data.pdf`, `alignment_log.txt`, `barcode_signals.pdf`).

## 2. Root Level Files (Global Session Scope)
Located directly in the session root (e.g., `..._g0/`):
- **NIDQ & Synchronization:**
  - `*_t0.nidq.bin` / `*_t0.nidq.meta`: Raw SpikeGLX NIDQ DAQ streams.
  - `*_t0.nidq_dclut.json`: Global time-alignment lookup tables for the NIDQ hardware time.
  - `*_t0.nidq_timealign_qc.pdf`: QC report for the time alignment offsets.
  - `*_ct_offsets.txt` / `*_fyi.txt`: CatGT processing offset tracking and logs.
- **Mimo-Specific Artifacts:**
  - `session_config.json`: Master mapping of channels and session metadata.
  - `states.nc`: Xarray dataset containing processed behavioral states, epochs, and other session-wide variables.
  - Standard analysis notebook copies (e.g., `PreprocessReachSession.ipynb`).
- **Directories:**
  - `qc_reports/`: Folder for session-wide, multi-modal QC graphs (e.g., `behavioral_states_lfp.pdf`, `video_alignment_rois.pdf`).

## 3. Probe-Specific Folders
Subdirectories named with `_imec` index suffixes (e.g., `*_imec0`, `*_imec1`).

### Core Hardware Data
- **`probe.json`**: Physical and channel configuration for that specific probe.
- **Raw DAQ (`t0`) files:** 
  - `*.imecX.ap.bin` / `.meta`: The raw Action Potential stream.
  - `*.imecX.lfp.bin`: The raw Local Field Potential stream (LFP `.meta` may be omitted if a lookup table is used).
  - `*.imecX.ap_siteCoords.txt`: Physical layout of the recorded channels on the shank.
- **Time-Aligned/CatGT Data (`tcat`) files — scratch-only, not archived:**
  - CatGT writes `*_tcat.imecX.ap.bin` / `.meta` (the unified, drift-corrected, time-aligned AP binary) and `*_tcat.imecX.ap.xd_384_6_500.txt` (extracted pulse/sync channels) to a local scratch processing directory (see the path recorded in `*_fyi.txt`, e.g. `d:/SortingTemp/...`).
  - These `tcat` files are **not** copied back into the archived session folder. Only the derived `*ap_dclut.json` / `*lfp_dclut.json`, the raw `*_t0.imecX.ap.bin` / `.meta`, and the root-level offset/log files (`*_ct_offsets.txt`, `*_fyi.txt`, `CatGT.log`) persist there.
  - Do not expect `tcat`-prefixed or `xd_*.txt` files to exist in an archived session directory; treat `*_t0.imecX.ap.bin` as the canonical persisted AP binary, with time alignment carried entirely by the dclut files.
- **Probe DCLUTs:** Time-alignment lookup hashes specific to the probe clock (`*ap_dclut.json`, `*lfp_dclut.json`).

### Derived Unit & Spike Data
- **Mimo Extracted Arrays:** `*.ap.su.npz` (single-unit) and `*.ap.mu.npz` (multi-unit) numpy caches used for fast loading.
- **UI Caches:** `.probe_labeler_cache/` (Used by internal plotting GUIs).
- **`kilosort4/` Subdirectory:**
  - Standard Phy/Kilosort output files (e.g., `spike_times.npy`, `cluster_group.tsv`, `channel_map.npy`, `templates.npy`).
  - **Manually Curated Data:** Contains files/columns suffixed `_sas` (e.g., `spike_times_sas.npy`, `spike_clusters_sas.npy`, `cluster_group_sas.tsv`) that reflect manually curated spike data, as well as a `.mat` file reflecting the sorting data used by the manual curation program.
  - `qc_reports/` subfolder: Unit-specific QC PDFs, auto-generated summaries for every single unit (e.g., `unit_121_qc_report.pdf`, `unit_122_qc_report.pdf`).