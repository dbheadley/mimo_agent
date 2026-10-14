---
description: "Use when modifying mimo_pack library code. Covers mandatory pre-read files, repo-wide data model rules, import boundaries, module placement routing, and which specialized library instructions to load next."
applyTo:
  - "mimo_pack/**"
---

# Library Core Workflow

## Decision Tree

1. Before editing any `mimo_pack/` file, read `mimo_pack/DESIGN_RULES.md`, `mimo_pack/consts.py`, and `environment.yml`.
2. If the change is in `mimo_pack/plot/`, also consult `.github/instructions/library-plotting.instructions.md`.
3. If the change is in `mimo_pack/analysis/`, `mimo_pack/preprocess/`, `mimo_pack/math/`, or `mimo_pack/util/`, also consult `.github/instructions/library-analysis.instructions.md`.
4. If the change adds or updates tests, also consult `.github/instructions/library-testing.instructions.md`.
5. If the change touches SpikeGLX, CatGT, dclut, Phy, NIDQ sync, or loader selection, also consult `.github/instructions/data-loading-time-alignment.instructions.md`.
6. If the task needs xarray/pynapple interop or helper lookup, also consult `.github/instructions/interop-utilities.instructions.md`.

## Directives

- Do not modify `mimo_pack` unless the user explicitly wants a library change.
- Use American English spelling in code, comments, docstrings, and user-facing strings.
- Never hardcode `time`, `channel`, `frequency`, `epoch`, `fs`, or `units`; import constants from `mimo_pack.consts`.
- All continuous neural data is `xarray.DataArray`. All spike collections are `pynapple.TsGroup`. All epochs are `pynapple.IntervalSet`.
- Only import packages listed in `environment.yml`.
- Check `mimo_pack.util` before writing helper code.

## Module Placement

| If the change mainly does... | Put it in... |
|---|---|
| Session/config/file discovery | `core/` |
| Disk I/O and raw data loading | `fileio/` |
| Cleaning, reshaping, or alignment of raw data | `preprocess/` |
| Scientific computation on in-memory rich objects | `analysis/` |
| Numpy/SciPy-only primitives | `math/` |
| Figure or axes creation | `plot/` |
| Shared helpers without domain-specific analysis logic | `util/` |
| Interactive notebook UI | `widgets/` |

## Import Boundaries

- `analysis/` must not import `fileio/`, `core/`, or `plot/`, and must not perform disk I/O.
- `math/` must stay limited to numpy/scipy primitives.
- `plot/` must not import `fileio/` or `core/`.
- `preprocess/` must not import `plot/`.
- `widgets/` must call existing analysis/preprocess code rather than duplicating logic.

## Completion Expectations

- Keep changes minimal and local to the owning abstraction.
- Preserve coordinates, attrs, and provenance when transforming rich objects.
- Add or update targeted tests when behavior changes.