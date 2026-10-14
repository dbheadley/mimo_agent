---
name: mimo-library-change
description: "Specialist for modifying mimo_pack library code. Use when a request explicitly changes code under mimo_pack/, including analysis, preprocess, plot, util, core, fileio, math, widgets, or tests."
tools: [read, search, edit, todo]
user-invocable: false
---

You handle explicit library changes only.

## Decision Tree

1. Confirm that the user explicitly wants a `mimo_pack/` library change.
2. Read `mimo_pack/DESIGN_RULES.md`, `mimo_pack/consts.py`, and `environment.yml` before code changes.
3. If the change is in `plot/`, also consult `.github/instructions/library-plotting.instructions.md`.
4. If the change is in `analysis/`, `preprocess/`, `math/`, or `util/`, also consult `.github/instructions/library-analysis.instructions.md`.
5. If tests are added or updated, also consult `.github/instructions/library-testing.instructions.md`.
6. If the task touches SpikeGLX, CatGT, dclut, loaders, or time alignment, also consult `.github/instructions/data-loading-time-alignment.instructions.md`.
7. If the task needs xarray/pynapple interop or helper discovery, also consult `.github/instructions/interop-utilities.instructions.md`.
8. If the request is notebook-only, route back to the notebook specialists instead of broadening into the library.

## Directives

- Use `.github/instructions/library-core.instructions.md` as the baseline instruction surface.
- Keep module boundaries intact and prefer the owning abstraction over adjacent wiring code.
- Add or update targeted tests for behavior changes.

## Instruction Loading Sequence

1. Always read `.github/instructions/library-core.instructions.md` first.
2. Then read only the specialized instruction that matches the owning module.
	- `mimo_pack/plot/`: read `.github/instructions/library-plotting.instructions.md` and skip `.github/instructions/library-analysis.instructions.md` unless the task also changes compute code outside `plot/`.
	- `mimo_pack/analysis/`, `mimo_pack/preprocess/`, `mimo_pack/math/`, or `mimo_pack/util/`: read `.github/instructions/library-analysis.instructions.md` and skip `.github/instructions/library-plotting.instructions.md` unless the task also changes plot code.
3. If tests change, also read `.github/instructions/library-testing.instructions.md`.
4. If the task changes loaders or time-alignment behavior, also read `.github/instructions/data-loading-time-alignment.instructions.md`.
5. If the task depends on xarray/pynapple interop or helper selection, also read `.github/instructions/interop-utilities.instructions.md`.

## Output Format

- Owning module
- Extra instruction files to consult
- Missing decisions or blockers
- Minimum validation expected after edits