---
description: "Use when tasks involve extracting behavioral task events from NIDQ digital/analog channels, video ROI tracking, barcode-based video alignment, or converting raw signal channels into pynapple Ts/IntervalSet events."
---

# Behavioral Task Events And Video Reference

This instruction intentionally has no `applyTo` field.
It is a reference-only surface that should load on demand by topic keywords rather than auto-apply by file path.

## Decision Tree

1. Converting a raw NIDQ digital or analog channel into discrete events/intervals?
   - Digital (TTL) channel: `digital_to_events()` for edges, `digital_to_intervals()` for on/off spans.
   - Analog channel with a threshold crossing: `analog_to_events()` / `analog_to_intervals()`.
   - Two channels that together define a pulse (start on one, stop on another): `cross_channel_intervals()`.
   - A whole session's standard event set at once: `extract_nidq_events()`.
2. Need rate, duty-cycle, or bout-level summaries of pulses rather than individual events?
   - Use `pulse_stats()` for per-pulse statistics, `track_pulse_frequency()` for a time-varying rate, or `detect_bouts()` to group events separated by short gaps into bouts.
3. Working with a quadrature encoder or multi-level analog signal?
   - Use `decode_quadrature()` or `quantize()` / `binarize()` respectively, not a hand-written threshold loop.
4. Processing acquired video (reach/behavior camera)?
   - Use the video pipeline below; do not assume a video file is already time-aligned to the session clock until `process_and_align_video()` has run.
5. Need to align video to the SpikeGLX/NIDQ clock?
   - Use `rois_to_barcodes()` plus `process_and_align_video()`, which consumes the dclut and sync-channel info described in `.github/instructions/data-loading-time-alignment.instructions.md`.

## Pipeline Anchors

- **NIDQ -> events:** `load_nidq()` -> `digital_to_events()` / `digital_to_intervals()` / `analog_to_events()` / `analog_to_intervals()` / `cross_channel_intervals()` -> `pulse_stats()` / `track_pulse_frequency()` / `detect_bouts()`, or the convenience wrapper `extract_nidq_events()`.
- **Video assembly:** `tif_to_video(folder_path)` -> `video_frame_table(video_path)`.
- **Video ROI tracking:** `create_video_rois(video_path)` -> `track_video_rois()` / `track_video_diff()`.
- **Video-to-clock alignment:** `rois_to_barcodes(table_path, roi_cols)` -> `process_and_align_video(vid_path, tbl_path, rois, dcl_path, sync_channel_info)`.
- **Plot:** `plot_video_data()` for frame-table/ROI traces.

## Directives

- The outputs of this pipeline (`pynapple.Ts` / `pynapple.IntervalSet`) are the standard way to epoch other data types (LFP, spikes, photometry) around task events; see `.github/instructions/interop-utilities.instructions.md` for epoch/window helpers.
- Do not hand-roll threshold/edge-detection logic; `mimo_pack.preprocess.signal` already covers digital, analog, quadrature, and dual-channel cases.
- Video QC artifacts (`aligned_video_data.pdf`, `alignment_log.txt`, `barcode_signals.pdf`) live in the behavior directory's `qc_reports/`; see `.github/instructions/data-neuropixel-directory.instructions.md`.
