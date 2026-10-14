# mimo_pack Design Rules and Coding Standards

**Purpose:** This document defines the architectural philosophy and coding standards for the `mimo_pack` library. All new code—whether written by a human or an AI agent—must adhere to these rules. When in doubt, consult this document first, then `refactor_plan.md` for implementation-level detail.

---

## 1. Core Philosophy

The library follows a **"Rich Object"** philosophy. Data never travels without its metadata.

- **Continuous signals** (LFP, AP, fiber photometry, etc.) must be represented as `xarray.DataArray`.
- **Discrete event data** (spike times) must be represented as `pynapple.TsGroup` or `pynapple.Ts`.
- **Epoch / interval data** must be represented as `pynapple.IntervalSet`.
- **Raw Numpy arrays and separate `fs` arguments are forbidden in public APIs.** Low-level math helpers in `mimo_pack/math/` are the sole exception.

---

## 2. Module Responsibilities

Each top-level directory has a single, well-defined responsibility. Functions must not cross these boundaries.

### `core/` — Session & Configuration
- Houses the `Session` class, which reads a per-dataset `session_config.json`.
- Provides file discovery (`find_files`) and parameter lookup (`get_params`).
- **No analysis or plotting logic belongs here.**
- The `Session` object must never be passed into analysis functions (see "God Object" anti-pattern in `refactor_plan.md`). Instead, extract parameters from the session and pass them as explicit keyword arguments.

### `fileio/` — IO Layer
- Sole responsibility: load data from disk and return **rich objects** (`xarray.DataArray`, `pynapple.TsGroup`).
- Every returned `DataArray` must have populated `.attrs` (see §3).
- Never performs complex analysis—only minimal transformations required for loading (e.g., unit conversion, notch filtering at load time).

### `preprocess/` — Data Preparation
- Prepares raw data for analysis: artifact rejection, resampling, re-referencing, behavioral event alignment.
- Accepts and returns rich objects (`xarray.DataArray`, `pynapple.TsGroup`).
- May call `fileio` functions to load intermediate files, but must not contain low-level file parsing.
- Distinct from `analysis/`: preprocessing **cleans or reshapes** data; analysis **extracts scientific results**.

### `analysis/` — Compute Layer
- **Pure functions only.** Accepts data in memory → returns data in memory.
- **No file loading, no disk access.** If a function needs data, the caller must load it first.
- Must raise `TypeError` if input is not the expected rich type.
- Must preserve spatial coordinates (`ch_x`, `ch_y`) and other metadata on output `DataArray` objects.
- Discrete event functions must accept `pynapple.TsGroup` and return `pynapple.TsGroup`.

### `math/` — Mathematical Primitives
- **Stay primitive.** Pure Numpy / SciPy / Python implementations only.
- **Do not import `xarray` or `pynapple` here.**
- Examples: curve fitting, regression, signal decomposition kernels.
- These are building blocks consumed by `analysis/` functions.

### `plot/` — Visualization Layer
- **Public / Private split:**
  - **Public:** `plot_foo(data: xr.DataArray, ...)` — validates input, extracts coordinates, calls private function.
  - **Private:** `_plot_foo(arr, ...)` — handles raw Matplotlib logic.
- Must use metadata from input objects (`.attrs`, coordinate labels) to auto-label axes.
- Discrete event plots must accept `pynapple.TsGroup`.
- Use constants from `consts.py` for axis lookups—never hardcode `"time"` or `"channel"`.

### `util/` — Shared Helpers
- General-purpose utilities: array manipulation, file path helpers, HDF5 convenience wrappers, scaffold/config generation.
- Must not contain domain-specific analysis logic.
- The `scaffold` module provides CLI helpers for generating default `session_config.json` files.

### `widgets/` — Interactive Tools
- `ipywidgets`-based components for manual data inspection and preprocessing in Jupyter notebooks.
- Widgets consume `xarray.DataArray` or `pynapple` objects for display.
- Widget code must not duplicate analysis logic—it should call functions from `analysis/` or `preprocess/`.

### `testing/` — Test Suite
- Contains unit and integration tests for all other modules.
- Provides synthetic data factories in `testing/data.py`:
  - `synthetic_lfp(duration, fs, n_chans)` → `xarray.DataArray`
  - `synthetic_spikes()` → `pynapple.TsGroup`
- Tests must use these factories rather than loading real data files, except for explicit integration tests in `test_data/`.

---

## 3. Data Structure Rules

### 3.1 `xarray.DataArray` (Continuous Signals)

#### Dimensions
Use the constants defined in `mimo_pack/consts.py`. Never hardcode dimension name strings.

| Constant     | Value         | Usage                          |
|--------------|---------------|--------------------------------|
| `DIM_TIME`   | `'time'`      | Time axis (always present)     |
| `DIM_FREQ`   | `'frequency'` | Frequency axis (spectral data) |
| `DIM_CHAN`   | `'channel'`   | Channel axis                   |
| `DIM_EPOCH`  | `'epoch'`     | Trial / epoch / sweep axis |

#### Stimulus Properties: Always Use DIM_EPOCH + Coordinates

**The Standard:** All experimental trials are indexed by `DIM_EPOCH`, regardless of whether stimuli vary across trials. Stimulus properties (e.g., orientation, current amplitude, contrast) are stored as **coordinates** along `DIM_EPOCH`, not as separate dimensions.

**Why this standard:**
1. **Handles incomplete designs:** Not all stimulus combinations need to be present (e.g., 15 of 20 possible orientation × contrast pairs).
2. **Handles repeated values:** Blocked designs with repeated stimulus values (e.g., `[0, 0, 0, 45, 45, 45, 90, 90, 90]`) work naturally.
3. **Single code path:** All library functions can assume `DIM_EPOCH` format—no branching logic for "is this a dimension or coordinate?"
4. **Clear semantics:** Coordinates on `DIM_EPOCH` represent experimental variables; other coordinates represent derived quantities.

**Identifying stimulus properties:**
List all stimulus coordinate names in `attrs['stimulus_properties']` for programmatic discovery:

```python
data = xr.DataArray(
    responses,  # shape: (n_trials, n_timepoints)
    dims=[DIM_EPOCH, DIM_TIME],
    coords={
        DIM_EPOCH: np.arange(n_trials),
        DIM_TIME: time_vec,
        'orientation': (DIM_EPOCH, [0, 45, 90, 0, 45, 90, ...]),  # stimulus property
        'contrast': (DIM_EPOCH, [0.5, 0.5, 0.5, 1.0, 1.0, 1.0, ...]),  # stimulus property
        'trial_start_time': (DIM_EPOCH, [...]),  # NOT a stimulus property
    },
    attrs={
        'stimulus_properties': ['orientation', 'contrast'],
        'description': 'Visual responses',
    }
)

# Aggregation uses groupby:
ori_tuning = data.groupby('orientation').mean(DIM_EPOCH)
contrast_tuning = data.groupby('contrast').mean(DIM_EPOCH)

# Selection uses boolean indexing or where:
ori_45 = data.sel({DIM_EPOCH: data.orientation == 45})
high_contrast = data.where(data.contrast == 1.0, drop=True)
```

**Benefits:**
- Works for any experimental design (factorial, randomized, adaptive, incomplete)
- No need to decide "is this a dimension or coordinate?"
- Library functions have consistent input expectations
- User code can define helper functions to reduce `.groupby()` verbosity

**Tradeoffs:**
- Cannot use xarray's dimensional operations directly (e.g., `data.mean('orientation')`)
- Must use `.groupby()` for stimulus-based aggregation
- Less efficient storage for complete factorial designs (repeated coordinate values)

**External data (e.g., pynapple output):**
Some external libraries (e.g., pynapple's `compute_tuning_curves`) return `xarray.DataArray` objects with stimulus features as named dimensions. Use `mimo_pack.util.xarray.normalize_to_epoch()` to convert these to the standard format before passing to mimo_pack functions.

#### Coordinates
Spatial coordinates must be attached as non-dimension coordinates:

| Constant      | Value        | Usage                              |
|---------------|--------------|------------------------------------|
| `COORD_X`     | `'ch_x'`    | Channel x-position (µm)           |
| `COORD_Y`     | `'ch_y'`    | Channel y-position (µm)           |
| `COORD_SHANK` | `'ch_shank'` | Shank index for multi-shank probes |

#### Required Attributes (`.attrs`)

| Key             | Constant      | Type    | Example                             |
|-----------------|---------------|---------|-------------------------------------|
| `'fs'`          | `ATTR_FS`     | `float` | `30000.0`                           |
| `'units'`       | `ATTR_UNITS`  | `str`   | `'µV'`, `'mV'`, `'a.u.'`           |
| `'description'` | —             | `str`   | `"LFP data (full): /path/to/file"`  |

#### The `description` Attribute
Every `DataArray` created by `fileio` **must** include a `description` attribute for provenance tracking. It must follow this format:

```
"{Data Type} ({Variant}): {Source Path}"
```

Examples from the codebase:
- `"NIDQ data: /data/mouse1/t0.nidq.bin"`
- `"LFP data (full): /data/mouse1/t0.imec0.lf.bin"`
- `"LFP data (epoch-windowed): /data/mouse1/t0.imec0.lf.bin"`
- `"AP data (intervals): /data/mouse1/t0.imec0.ap.bin"`
- `"Doric fiber photometry: /data/mouse1/doric_file.doric"`
- `"Spike waveforms from Phy output: cluster 42"`
- `"Images from: /data/mouse1/frames/"`
- `"ABF file: /data/invitro/21330003.abf"`

When an `analysis/` or `preprocess/` function transforms a `DataArray`, it should **append** to the description to maintain the provenance chain, or create a new description that references the transformation:
```
"CSD of: LFP data (full): /data/mouse1/t0.imec0.lf.bin"
```

#### Sampling Rate
Never pass `fs` as a separate function argument in public APIs. Extract it from the data:
```python
fs = float(data.attrs[ATTR_FS])
# or derive from the time coordinate:
fs = 1.0 / float(data[DIM_TIME].diff(DIM_TIME).median())
```

### 3.2 `pynapple` Objects (Discrete / Event Data)

| Type                    | Use Case                              |
|-------------------------|---------------------------------------|
| `pynapple.TsGroup`     | Collection of spike trains (units)    |
| `pynapple.Ts`          | Single event series (timestamps only) |
| `pynapple.Tsd`         | Time series with single values        |
| `pynapple.TsdFrame`    | Time series with multiple columns     |
| `pynapple.IntervalSet` | Epochs / valid time intervals         |

#### `time_support`
Every `pynapple` object carries a `time_support` attribute (an `IntervalSet`). This defines the valid time range of the data. When combining pynapple with xarray, always respect `time_support`—do not assume the entire xarray time axis is valid.

### 3.3 Combining Xarray and Pynapple

Many neuroscience analyses require both continuous signals and spike data. Follow these rules:

#### Rule 1: Use Pynapple for Time Alignment
Use pynapple's built-in methods to align continuous and discrete data. **Do not** manually search timestamps with `np.searchsorted` or boolean indexing.

```python
# ✅ CORRECT: Use value_from to sample LFP at spike times
lfp_tsd = nap.Tsd(t=lfp_xr[DIM_TIME].values, d=lfp_xr.values)
lfp_at_spikes = lfp_tsd.value_from(spike_ts)

# ❌ WRONG: Manual index searching
idx = np.searchsorted(lfp_xr[DIM_TIME].values, spike_times)
lfp_at_spikes = lfp_xr.values[idx]
```

#### Rule 2: Use `IntervalSet` for Epoch Restriction
When restricting analysis to specific time windows, use `pynapple.IntervalSet` rather than manual boolean masks on the time coordinate.

```python
# ✅ CORRECT: Restrict spikes to valid epochs
valid_epochs = nap.IntervalSet(start=[0, 30], end=[20, 50])
restricted_spikes = spikes.restrict(valid_epochs)

# ❌ WRONG: Manual masking
mask = (spike_times > 0) & (spike_times < 20)
restricted = spike_times[mask]
```

#### Rule 3: Let Pynapple Compute, Receive Xarray
As documented in [pynapple's tuning curve guide](https://pynapple.org/user_guide/06_tuning_curves.html), pynapple's analysis functions (e.g., `nap.compute_tuning_curves`, `nap.compute_response_per_epoch`) **return `xarray.DataArray` objects** with labeled coordinates. Prefer this pattern:

1. Let pynapple handle binning, counting, and time alignment.
2. Receive the result as an `xarray.DataArray`.
3. Use xarray for downstream operations (slicing, plotting, statistics).

```python
# Pynapple computes and returns xarray
tuning = nap.compute_tuning_curves(
    data=tsgroup,
    features=position_tsd,
    bins=50,
    feature_names=["position"]
)
# tuning is an xarray.DataArray — use .sel(), .plot(), etc.
tuning.sel(unit=0).plot()
```

#### Rule 4: Conversion Direction
- **Xarray → Pynapple:** When you need pynapple time-alignment features on continuous data. Ensure the signal is 1D (time) or 2D (time × feature).
  ```python
  lfp_tsd = nap.Tsd(t=lfp_xr[DIM_TIME].values, d=lfp_xr.values)
  ```
- **Pynapple → Xarray:** Use pynapple's built-in return types (many analysis functions already return `xarray.DataArray`). For manual conversion:
  ```python
  xr.DataArray(nap_obj.values, coords={DIM_TIME: nap_obj.index})
  ```

#### Rule 5: No Raw Array Bridges
Never convert both objects to raw Numpy arrays to perform alignment. The whole point of rich objects is to prevent misalignment bugs.

```python
# ❌ WRONG: Stripping both to numpy for alignment
spike_arr = spikes.times()
lfp_arr = lfp_xr.values
time_arr = lfp_xr[DIM_TIME].values
# ... manual alignment code ...

# ✅ CORRECT: Use the object APIs
lfp_tsd = nap.Tsd(t=lfp_xr[DIM_TIME].values, d=lfp_xr.values)
result = lfp_tsd.value_from(spikes[unit_id])
```

---

## 4. Coding Standards

### 4.1 Type Hints
- All public functions must have type hints in their signatures.
- Use the actual types: `xr.DataArray`, `nap.TsGroup`, `nap.IntervalSet`.
- For internal array work, use `npt.NDArray[np.float64]`.
- Runtime type checks are required in addition to static hints:
  ```python
  if not isinstance(data, xr.DataArray):
      raise TypeError(f"Expected xr.DataArray, got {type(data).__name__}")
  ```

### 4.2 Docstrings
All public functions must have **Numpy-style** docstrings with the following sections:

```python
def evoked_lfp(lfp_trials: xr.DataArray, baseline: float = 0.0) -> xr.DataArray:
    """Compute the trial-averaged evoked LFP.

    Parameters
    ----------
    lfp_trials : xr.DataArray
        Trial-windowed LFP with dimensions (epoch, time, channel).
        Must contain 'ch_y' coordinate for CSD computation.
    baseline : float, optional
        Baseline period in seconds to subtract, by default 0.0.

    Returns
    -------
    xr.DataArray
        Trial-averaged LFP with dimensions (time, channel).
        Retains all spatial coordinates from input.

    Raises
    ------
    TypeError
        If `lfp_trials` is not an xr.DataArray.
    ValueError
        If required dimension 'time' is missing.

    Examples
    --------
    >>> from mimo_pack.testing.data import synthetic_lfp
    >>> lfp = synthetic_lfp(duration=1.0, fs=1000, n_chans=32)
    >>> result = evoked_lfp(lfp)
    """
```

### 4.3 Constants
- Import dimension and attribute names from `mimo_pack.consts`.
- **Never hardcode** `"time"`, `"channel"`, `"frequency"`, or `"fs"` as bare strings.
  ```python
  # ✅ CORRECT
  from mimo_pack.consts import DIM_TIME, DIM_CHAN, ATTR_FS
  fs = data.attrs[ATTR_FS]

  # ❌ WRONG
  fs = data.attrs['fs']
  ```

### 4.4 Naming Conventions
- **Analysis functions:** Use simple, descriptive names. No `_xr` or `_numpy` suffixes on public APIs.
  - `stft`, not `stft_xr`.
  - `evoked_lfp`, not `evoked_lfp_xarray`.
- **Plot functions:** Public `plot_foo` / Private `_plot_foo`.
- **IO functions:** Named after data type: `load_lfp`, `load_nidq`, `load_spikes`.

---

## 5. Error Handling

### 5.1 Input Validation
Every public function must validate inputs immediately, before any computation:

```python
def compute_csd(lfp: xr.DataArray, spacing: float = 100.0) -> xr.DataArray:
    # 1. Type check
    if not isinstance(lfp, xr.DataArray):
        raise TypeError(
            f"Expected xr.DataArray, got {type(lfp).__name__}. "
            "Use mimo_pack.fileio functions to load data first."
        )
    # 2. Dimension check
    if DIM_TIME not in lfp.dims:
        raise ValueError(
            f"DataArray must have a '{DIM_TIME}' dimension. "
            f"Found dimensions: {lfp.dims}"
        )
    # 3. Coordinate check
    if COORD_Y not in lfp.coords:
        raise ValueError(
            f"DataArray must have '{COORD_Y}' coordinate for CSD computation."
        )
    # ... proceed with computation
```

### 5.2 Exception Types

| Situation                          | Exception      |
|------------------------------------|----------------|
| Wrong input type                   | `TypeError`    |
| Missing dimension or coordinate    | `ValueError`   |
| Missing required `.attrs` key      | `KeyError`     |
| File not found during IO           | `FileNotFoundError` |
| Invalid parameter value            | `ValueError`   |

### 5.3 Error Messages
Error messages must be **actionable**. Tell the user what went wrong and how to fix it:
```python
# ✅ GOOD
raise TypeError(
    f"Expected xr.DataArray, got {type(data).__name__}. "
    "Use mimo_pack.fileio.load_lfp() to load LFP data."
)

# ❌ BAD
raise TypeError("Wrong type")
```

### 5.4 Dimension Agnosticism
Do not assume axis order. Never use positional indexing like `data[0, :]` to select channels. Always use named selection:
```python
# ✅ CORRECT
data.sel({DIM_CHAN: 0})
data.isel({DIM_CHAN: 0})

# ❌ WRONG
data[0, :]
data.values[0]
```

---

## 6. Definition of Done

A module is considered properly implemented when:

- [ ] It imports constants from `mimo_pack.consts`.
- [ ] Public functions accept `xr.DataArray` or `pynapple.TsGroup` inputs (not raw arrays).
- [ ] No function accepts a file path string (except in `fileio/` or `core/`).
- [ ] All public functions have Numpy-style docstrings with Parameters, Returns, Raises, and Examples.
- [ ] All public functions have type hints in their signatures.
- [ ] Input validation raises `TypeError` / `ValueError` with actionable messages.
- [ ] A unit test exists using synthetic data from `mimo_pack.testing`.
- [ ] The `description` attribute is populated for any newly created `DataArray`.
- [ ] Spatial coordinates are preserved through transformations.
