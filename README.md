# mimo2

**An agentic-coding-first library for systems neuroscience analysis.**

`mimo2` is the working repository for **`mimo_pack`**, the base Python library of the MIMO lab (author: Drew B. Headley). It is designed to be used **together with a coding agent** (GitHub Copilot in VS Code), not as a library you read through and script by hand. You describe the analysis you want in plain language, and the agent plans it, writes the notebook, calls the right library functions, and saves the outputs.

Supported data types include:

- In vivo electrophysiology: Neuropixels / SpikeGLX recordings (AP, LFP, NIDQ), CatGT output, Kilosort/Phy spike sorting, and Intan recordings
- Fiber photometry (Doric)
- In vitro electrophysiology (ABF)
- Behavioral video and stimulation images

## Why agent-first?

Both the library and the repository are structured so an agent can use them reliably:

- **Strict, predictable conventions.** Every module has one responsibility, and data is always a rich object (`xarray.DataArray`, `pynapple.TsGroup`, `pynapple.IntervalSet`) that carries its own metadata. The agent never has to guess sampling rates, units, or dimension names.
- **Written rules the agent reads.** [`AGENTS.md`](AGENTS.md), [`mimo_pack/DESIGN_RULES.md`](mimo_pack/DESIGN_RULES.md), and the files under [`.github/`](.github) encode the lab's conventions, so the agent follows them without you restating them.
- **Specialist agents.** Requests are routed to agents for new analyses, existing notebooks, library changes, and workspace customization.
- **Templates as anchors.** The agent starts from the closest notebook in [`templates/`](templates) rather than inventing a workflow.
- **Guardrails.** The agent leaves `mimo_pack/` alone unless you explicitly ask for a library change.

You can still write code against `mimo_pack` directly, but the conventions are enforced by the agent. If you work by hand, read the design rules first.

## Quick start

1. **Install the environment**

   ```powershell
   conda env create -f environment.yml
   conda activate mimo_env2
   pip install -e .
   ```

2. **Open the `mimo2` folder as the VS Code workspace root.** This is required so Copilot picks up `AGENTS.md` and the `.github/` customizations.
3. **Open Chat (`Ctrl+Alt+I`), switch to Agent mode, and select `mimo-router`.**
4. **Describe what you want.** Include the target file and data modality if you know them. For example:
   - "Create a notebook that analyzes LFP spectral power during reaching for session X."
   - "Add a z-scoring step to `project/my_analysis.ipynb`."
   - "Scale this notebook from one session to all sessions in this folder."
   - "Add a `window` option to `mimo_pack.analysis.evoked`, with tests."
5. **Answer the agent's questions.** It asks about all unresolved decisions together before editing: scope, data type, preprocessing, data selection, combining strategy, visualization, and statistics.

Plain Chat also works. `AGENTS.md` is always loaded, and the instruction files attach by file pattern or topic keywords such as SpikeGLX, CatGT, dclut, NIDQ, or Phy.

## How the agent behaves

`mimo-router` classifies each request and hands it to the narrowest specialist. It reports the classification, the specialist, the instruction files to consult, and any open questions.

| Request | Specialist | Behavior |
|---|---|---|
| New notebook or analysis workflow | `mimo-new-analysis` | Confirms scope and decisions, starts from the closest template, and writes the notebook in `project/` |
| Change to an existing notebook, `project/` file, or template | `mimo-continue-analysis` | Edits locally, preserves the notebook's structure, and can scale a single session to a batch |
| Explicit change under `mimo_pack/` | `mimo-library-change` | Reads `DESIGN_RULES.md`, `consts.py`, and `environment.yml` first, keeps module boundaries, and adds or updates tests |
| Edit to `AGENTS.md`, instructions, agents, or prompts | `mimo-customization` | Keeps `AGENTS.md` thin and puts details in `.github/` |
| Reference-only question | `mimo-router` itself | Answers directly and points to the smallest relevant instruction file |

Rules the agent follows:

- **Library changes need an explicit request.** If a request mixes notebook work and library changes, the library change comes first and the notebook work becomes a follow-up.
- **New notebooks and scripts go in `project/`**, not in `mimo_pack/`.
- **Data model:** continuous data is `xarray.DataArray`, spikes are `pynapple.TsGroup`, and epochs are `pynapple.IntervalSet`.
- **No hardcoded names:** `time`, `channel`, `frequency`, `epoch`, `fs`, and `units` are imported from `mimo_pack.consts`.
- **Reuse before writing:** it checks `mimo_pack.util` and existing analysis functions before creating helpers, and only imports packages listed in `environment.yml`.
- **SpikeGLX data:** it loads from dclut JSON files and never assumes CatGT output is time-aligned.
- **Public datasets (Allen, IBL):** it loads the dedicated dataset guidance under `project/`.
- **Style:** American English everywhere.

## Design philosophy

The library follows a "rich object" approach: data never travels without its metadata.

| Data | Representation |
|---|---|
| Continuous signals (LFP, AP, photometry, ...) | `xarray.DataArray` |
| Spike data | `pynapple.TsGroup` |
| Epochs / intervals | `pynapple.IntervalSet` |

Constants are defined in [`mimo_pack/consts.py`](mimo_pack/consts.py). The full architectural rules are in [`mimo_pack/DESIGN_RULES.md`](mimo_pack/DESIGN_RULES.md).

## Typical workflow

Ask the agent for each step, or for the whole pipeline:

1. **Preprocess** a raw session with `mimo_pack.preprocess.session.preprocess_session()` (or `preprocess_spikeglx()`). This generates `*_dclut.json` files that hold cross-probe and NIDQ time alignment.
2. **Load** data through the `fileio` loaders, using dclut JSON paths rather than raw `.bin` files. For example, use `load_phy_spikes(..., dcl_file=AP_DCLUT)` so spike times are in aligned seconds.
3. **Analyze and plot** with `analysis` and `plot` functions, then save results alongside the session.

CatGT only performs CAR, AP-band filtering, and trigger concatenation. Its output is **not** time-aligned across probes or to NIDQ, so rely on the dclut files for alignment.

### Template notebooks

Starting points live in [`templates/`](templates):

- `PreprocessReachSession`
- `NidqChannelConfig`, `ProcessNidqData`
- `AnalyzeLFPSpectral`, `UnderstandingLFPProbeLayout`
- `AnalyzeMonosynapticCC`
- `AnalyzeFiberPhotometry`
- `AnalyzeIVCurve`, `AnalyzePatternedOptoStimInVitro`, `CharacterizeOpsinCells`

## Repository layout

```
mimo2/
├── mimo_pack/       # The library (see below)
├── templates/       # Starter analysis notebooks
├── project/         # Project-specific notebooks and ad hoc scripts
├── pilots/          # Exploratory notebooks
├── test_data/       # Small datasets used by tests (invitro, lfp, phy, reach)
├── environment.yml  # Conda environment definition
├── setup.py         # Package install script
├── AGENTS.md        # Always-on agent rules and top-level routing
└── .github/         # Agent definitions, instructions, and prompts
```

### `mimo_pack` modules

| Module | Responsibility |
|---|---|
| `core` | `Session` class, which reads a per-dataset `session_config.json`, plus file discovery and parameter lookup |
| `fileio` | Load data from disk into rich objects (`abf`, `ap`, `lfp`, `nidq`, `loadphy`, `doric`, `intan`, `image`, `spikeglx`, `invitro`) |
| `preprocess` | Clean and reshape data: time alignment, NIDQ, LFP, units, video, fiber photometry, antidromic, in vitro, and whole-session preprocessing |
| `analysis` | Pure compute functions: spectral, evoked, CSD, cross-correlation, point process, filtering, antidromic, behavioral state, in vitro properties, and more |
| `math` | Primitive NumPy/SciPy helpers (curve fitting, regression, circular statistics) |
| `plot` | Plotting for timeseries, probes, units, histograms, images, video, connectivity, and more |
| `widgets` | Interactive notebook widgets (cell selector, behavioral state, NIDQ, SpikeGLX, video, opsin, ...) |
| `util` | Shared helpers for xarray, arrays, dataframes, files, and HDF5 |
| `testing` | Unit tests and synthetic data factories |

Module boundaries are strict: `fileio` loads, `preprocess` prepares, `analysis` computes (no disk access), `math` stays free of xarray/pynapple, and `plot` draws.

## Customizing the agent

| Location | Purpose |
|---|---|
| [`AGENTS.md`](AGENTS.md) | Always-on rules and top-level routing |
| [`.github/agents/`](.github/agents) | Agent definitions (`*.agent.md`) |
| [`.github/instructions/`](.github/instructions) | Topic and file-scoped guidance (`*.instructions.md`) |
| [`.github/prompts/`](.github/prompts) | Prompt files |

To change agent behavior, edit these files or ask `mimo-router` to do so. Keep one canonical home per topic and avoid duplicating rules.

## Testing

Tests live in `mimo_pack/testing/` and use synthetic data factories.

```powershell
pytest mimo_pack/testing
```

## Contributing

- Use American English spelling throughout.
- Only import packages listed in `environment.yml`.
- Check `mimo_pack.util` before writing a new helper.
- Read [`mimo_pack/DESIGN_RULES.md`](mimo_pack/DESIGN_RULES.md) before changing library code.
- When a convention changes, update the matching file under `.github/` or `AGENTS.md` so the agent stays in sync with the library.
