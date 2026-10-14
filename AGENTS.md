# AGENTS.md — mimo_pack Router

This file is the thin, always-on router for the workspace.

Canonical task-specific guidance now lives under:

- `.github/instructions/`
- `.github/agents/`

## Always-On Rules

Before modifying any code in `mimo_pack/`, read these files in order:

1. `mimo_pack/DESIGN_RULES.md`
2. `mimo_pack/consts.py`
3. `environment.yml`

Non-negotiable repo-wide rules:

- Use American English spelling in code, comments, docstrings, markdown, notebook prose, variable names, and user-facing strings.
- Never hardcode `time`, `channel`, `frequency`, `epoch`, `fs`, or `units`; import from `mimo_pack.consts`.
- Do not modify `mimo_pack` source unless the user explicitly wants a library change.
- New notebooks and ad hoc scripts belong in `project/`, not in `mimo_pack/`.
- Continuous neural data is `xarray.DataArray`. Spike data is `pynapple.TsGroup`. Epochs are `pynapple.IntervalSet`.
- Only import packages listed in `environment.yml`.
- Check `mimo_pack.util` before writing helper code.

## Coding Style

Write like a lazy senior dev (after [Ponytail](https://github.com/dietrichgebert/ponytail)): the best code is the code never written. Before writing anything, read the task and the code/notebook it touches, and find the smallest complete change — in order of preference: nothing needed > an existing `mimo_pack` helper or notebook pattern used the way surrounding code uses it > a library already in `environment.yml` > a few lines anyone can read at a glance. Finish the whole change (callers, tests, notebook cells it affects); never trade away validation, data-loss handling, or the mimo-specific rules above to save lines. This is about doing the smallest *necessary* thing, not about terseness — see `mimo_pack/DESIGN_RULES.md` for mimo-specific API and style conventions, which take precedence where they're more specific.

## Interaction Protocol

- Before implementing a non-trivial change (new notebook, new analysis, library change, redesign), state a short plan and get confirmation unless the request is already unambiguous and small.
- If multiple design choices are reasonable (e.g. which baseline method, which module to place code in, whether to touch `mimo_pack/`), present the options rather than silently picking one.
- Group every unresolved question into a single round of questions instead of asking one at a time.

## Top-Level Decision Tree

1. Is the task about `AGENTS.md`, `.instructions.md`, `.agent.md`, or workspace prompts?
   - Use `.github/instructions/workspace-customization.instructions.md`.
   - Prefer the `mimo-router` agent, which can hand off to `mimo-customization`.
2. Is the task creating a new analysis notebook or a new notebook-based workflow?
   - Use `.github/instructions/notebook-analysis.instructions.md`.
   - If the notebook uses Allen Institute or IBL data, also load `.github/instructions/public-datasets.instructions.md`.
   - If the notebook targets a specific data type or experiment (unit/spike, LFP, fiber photometry, in-vitro ephys, behavioral task events), also load the matching `analysis-*.instructions.md` file (see Canonical Files).
   - Prefer the `mimo-router` agent, which can hand off to `mimo-new-analysis`.
3. Is the task extending an existing notebook, `project/` file, or template workflow?
   - Use `.github/instructions/notebook-analysis.instructions.md`.
   - If the notebook uses Allen Institute or IBL data, also load `.github/instructions/public-datasets.instructions.md`.
   - If the notebook targets a specific data type or experiment (unit/spike, LFP, fiber photometry, in-vitro ephys, behavioral task events), also load the matching `analysis-*.instructions.md` file (see Canonical Files).
   - Prefer the `mimo-router` agent, which can hand off to `mimo-continue-analysis`.
4. Is the task explicitly changing code under `mimo_pack/`?
  - This takes precedence over notebook work. If a request mixes notebook edits and `mimo_pack/` changes, treat the library change as the controlling task and split the notebook follow-up if needed.
   - Use `.github/instructions/library-core.instructions.md`.
   - Prefer the `mimo-router` agent, which can hand off to `mimo-library-change`.
5. Is the task reference-only?
   - Load the smallest relevant instruction file instead of broad repo guidance.

## Library Subrouting

When the task is a `mimo_pack/` library change, use these additional decision points:

- `mimo_pack/analysis/`, `mimo_pack/preprocess/`, `mimo_pack/math/`, or `mimo_pack/util/`:
  - Also use `.github/instructions/library-analysis.instructions.md`.
- `mimo_pack/plot/`:
  - Also use `.github/instructions/library-plotting.instructions.md`.
- `mimo_pack/testing/` or any behavior change requiring tests:
  - Also use `.github/instructions/library-testing.instructions.md`.
- SpikeGLX, CatGT, dclut, Phy, imec, NIDQ, or time-alignment questions:
  - Also use `.github/instructions/data-loading-time-alignment.instructions.md`.
- xarray/pynapple interop or helper selection questions:
  - Also use `.github/instructions/interop-utilities.instructions.md`.

## Canonical Files

| Topic | Canonical file |
|---|---|
| Notebook workflows | `.github/instructions/notebook-analysis.instructions.md` |
| Unit/spike analysis | `.github/instructions/analysis-unit.instructions.md` |
| LFP analysis | `.github/instructions/analysis-lfp.instructions.md` |
| Fiber photometry analysis | `.github/instructions/analysis-fiber-photometry.instructions.md` |
| In-vitro electrophysiology analysis | `.github/instructions/analysis-invitro-ephys.instructions.md` |
| Behavioral task events and video | `.github/instructions/analysis-behavior-task-events.instructions.md` |
| Library base rules | `.github/instructions/library-core.instructions.md` |
| Compute-oriented library code | `.github/instructions/library-analysis.instructions.md` |
| Plotting code | `.github/instructions/library-plotting.instructions.md` |
| Testing | `.github/instructions/library-testing.instructions.md` |
| SpikeGLX and time alignment | `.github/instructions/data-loading-time-alignment.instructions.md` |
| Neuropixels/SpikeGLX session directory layout | `.github/instructions/data-neuropixel-directory.instructions.md` |
| xarray/pynapple interop and utility lookup | `.github/instructions/interop-utilities.instructions.md` |
| Public datasets (Allen Institute / IBL) | `.github/instructions/public-datasets.instructions.md` |
| Workspace customization | `.github/instructions/workspace-customization.instructions.md` |
| Routing agents | `.github/agents/*.agent.md` |

## Agent Entry Points

Available workspace agents:

- `mimo-router` — top-level classifier and handoff agent
- `mimo-new-analysis` — specialist for new notebook workflows
- `mimo-continue-analysis` — specialist for existing notebook workflows
- `mimo-library-change` — specialist for explicit `mimo_pack/` code changes
- `mimo-customization` — specialist for AGENTS and customization maintenance

Keep one canonical home per topic under `.github/instructions/` or `.github/agents/`.