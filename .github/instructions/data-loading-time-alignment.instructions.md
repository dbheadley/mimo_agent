---
description: "Use when tasks mention SpikeGLX, CatGT, imec, NIDQ, dclut, Phy, Kilosort, loader selection, session folder layout, or cross-probe time alignment. Covers what CatGT does, how mimo_pack aligns clocks, and which loaders to use."
---

# Data Loading And Time Alignment Reference

This instruction intentionally has no `applyTo` field.
It is a reference-only surface that should load on demand by topic keywords rather than auto-apply by file path.

## Decision Tree

1. Is the question about CatGT?
   - Clarify that CatGT does CAR, AP-band filtering, and trigger concatenation only.
   - Do not treat CatGT output as cross-probe or NIDQ time-aligned.
2. Is the question about cross-probe or NIDQ alignment?
   - Use `mimo_pack.preprocess.session.preprocess_session()` or `preprocess_spikeglx()`.
   - Alignment is stored in the generated dclut files.
3. Is the task loading SpikeGLX LFP, AP, or NIDQ data?
   - Use dclut JSON paths, not raw `.bin` paths.
4. Is the task loading Phy spikes?
   - Use `load_phy_spikes(..., dcl_file=AP_DCLUT)` so spike times are synchronized to aligned seconds.
5. Is this part of notebook authoring?
   - Also consult `.github/instructions/notebook-analysis.instructions.md`.
6. Is the question about the full session/probe directory layout or file naming conventions rather than loader selection?
   - Also consult `.github/instructions/data-neuropixel-directory.instructions.md`.

## Directives

- Never assume `tcat.imec*.ap.bin` files are time-aligned across probes.
- Treat dclut JSON files as the canonical inputs for mimo_pack SpikeGLX loaders.
- Keep raw `.bin` handling inside preprocessing/fileio layers.

## Loader Quick Picks

| Data type | Preferred loader |
|---|---|
| SpikeGLX LFP | `mimo_pack.fileio.lfp.load_lfp_full()` |
| SpikeGLX AP | `mimo_pack.fileio.ap.load_ap_full()` |
| SpikeGLX NIDQ | `mimo_pack.fileio.nidq.load_nidq()` |
| Phy spikes | `mimo_pack.fileio.loadphy.load_phy_spikes()` |
| Doric photometry | `mimo_pack.fileio.doric.load_doric()` |
| ABF metadata/settings | `mimo_pack.fileio.abf.load_abf_settings()` |
| Intan header | `mimo_pack.fileio.intan.read_rhd_header()` |
| Stimulation images | `mimo_pack.fileio.image.load_images()` |

## Layout Notes

- SpikeGLX sessions contain raw/meta files plus generated `*_dclut.json` files used by loaders.
- Phy outputs live under a probe-specific subdirectory such as `kilosort4/`.
- Doric recordings are single `.doric` HDF5 files.
- In-vitro ABF datasets are typically organized by subject -> slice -> cell -> protocol files.