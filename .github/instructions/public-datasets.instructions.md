---
description: "Use when tasks involve Allen Institute Visual Coding Neuropixels data, IBL International Brain Laboratory data, ONE API, public neuroscience datasets, NWB files, loading data from W:\\PublicData\\Allen\\NPVisCode, or reformatting public dataset outputs for mimo_pack analysis and plot functions."
applyTo:
  - "project/public/**"
  - "project/allen/**"
  - "project/ibl/**"
---

# Public Datasets Reference

## Decision Tree

1. Is the data source Allen Institute or IBL?
   - Allen → use the Allen section below.
   - IBL → use the IBL section below.
2. What data type is needed?
   - LFP / continuous voltage → reformat to `xr.DataArray` with dims `[time, channel]`.
   - Sorted spikes → reformat to `pynapple.TsGroup`.
   - Running speed or continuous behavioral signal → reformat to `xr.DataArray` with dim `[time]`.
   - Stimulus trials or epochs → reformat to `pynapple.IntervalSet` or `pd.DataFrame`.
   - Channel / unit metadata → keep as `pd.DataFrame`.
3. Are helper functions needed to load or reformat the data?
   - Write them in the same directory as the notebook, not inside `mimo_pack/`.
   - Name helper files descriptively, e.g., `allen_loaders.py` or `ibl_loaders.py`.

## Non-Negotiable Directives

- Never call `mimo_pack.fileio.*` or `mimo_pack.preprocess.*` on Allen or IBL data.
- Always reformat to the mimo_pack data model before calling any `mimo_pack.analysis.*` or `mimo_pack.plot.*` function.
- Use `DIM_TIME`, `DIM_CHAN`, `ATTR_FS`, and `ATTR_UNITS` from `mimo_pack.consts` when constructing DataArrays — never hardcode those strings.
- Helper functions belong in the same directory as the notebook. Do not add them to `mimo_pack/`.
- If `one.api` is not in `environment.yml`, then run `pip install ONE-api` before starting IBL work.

---

## Allen Institute Visual Coding Neuropixels

**Local root**: `W:\PublicData\Allen\NPVisCode`
If path not found, ask the user to provide it.

### Folder Structure

| Path | Contents |
|---|---|
| `sessions.csv` | Session metadata: `id`, `session_type`, `age_in_days`, `sex`, `genotype`, `has_nwb` |
| `probes.csv` | Probe metadata: `id`, `ecephys_session_id`, `name` (probeA–F), `lfp_sampling_rate` (~2500 Hz), `sampling_rate` (~30000 Hz), `surface_channel_index` |
| `channels.csv` | Channel metadata: `id`, `ecephys_probe_id`, `local_index`, `probe_horizontal_position`, `probe_vertical_position`, CCF coordinates, `ecephys_structure_acronym` |
| `units.csv` | Unit metadata: `id`, `ecephys_channel_id`, `quality`, `firing_rate`, `amplitude`, `snr`, `isi_violations`, `isolation_distance` |
| `session_{id}/session_{id}.nwb` | Spikes, stimulus intervals, running speed, opto stimulation (HDF5) |
| `session_{id}/probe_{probe_id}_lfp.nwb` | LFP for one probe (HDF5) |

- `pynwb` is not in `environment.yml`. Use `h5py` for all NWB file reads.
- Load the root CSVs with `pandas.read_csv()` first to discover session and probe IDs before opening NWB files.

### h5py Paths

**Session NWB** (`session_{id}.nwb`):

| Data | h5py path | Notes |
|---|---|---|
| Spike times (ragged) | `units/spike_times` | Flat array of all spike times; use `units/spike_times_index` to split by unit |
| Spike times index | `units/spike_times_index` | Cumulative end indices for each unit (length = n_units) |
| Unit IDs | `units/id` | Matches row order of all other `units/*` fields |
| Unit quality | `units/quality` | String: `'good'` or `'noise'`; filter to `'good'` before analysis |
| Unit peak channel | `units/peak_channel_id` | Cross-reference to `channels.csv` |
| Running speed | `processing/running/running_speed/data` | m/s; timestamps at `processing/running/running_speed/timestamps` |
| Stimulus intervals | `intervals/{stim_name}_presentations/start_time` | Available stimuli: `drifting_gratings`, `flashes`, `gabors`, `natural_movie_one`, `natural_movie_three`, `natural_scenes`, `spontaneous`, `static_gratings` |
| Stimulus stop times | `intervals/{stim_name}_presentations/stop_time` | Paired with `start_time` above |
| Opto stimulation | `processing/optotagging/optogenetic_stimulation/start_time` | Also: `stop_time`, `level`, `duration` |
| Electrode metadata | `general/extracellular_ephys/electrodes/` | Fields: `id`, `local_index`, `probe_vertical_position`, `probe_horizontal_position`, `probe_id`, `location` (brain region), `x`, `y`, `z` (CCF) |

**LFP NWB** (`probe_{probe_id}_lfp.nwb`):

| Data | h5py path | Notes |
|---|---|---|
| LFP voltage | `acquisition/probe_{probe_id}_lfp/probe_{probe_id}_lfp_data/data` | Shape `(n_times, n_channels)`, float32 |
| LFP timestamps | `acquisition/probe_{probe_id}_lfp/probe_{probe_id}_lfp_data/timestamps` | Shape `(n_times,)` in seconds |
| Electrode indices | `acquisition/probe_{probe_id}_lfp/probe_{probe_id}_lfp_data/electrodes` | Shape `(n_channels,)`; values are electrode IDs from the root `channels.csv` |
| Channel metadata | `general/extracellular_ephys/electrodes/` | Same fields as session NWB |

### Reformatting Patterns

**LFP → `xr.DataArray`**:
```python
import h5py, numpy as np, xarray as xr, pandas as pd
from mimo_pack.consts import DIM_TIME, DIM_CHAN, ATTR_FS, ATTR_UNITS

probe_id = ...  # from probes.csv
lfp_path = rf'W:\PublicData\Allen\NPVisCode\session_{session_id}\probe_{probe_id}_lfp.nwb'

with h5py.File(lfp_path, 'r') as f:
    key = f'acquisition/probe_{probe_id}_lfp/probe_{probe_id}_lfp_data'
    data = f[f'{key}/data'][:]          # (n_times, n_channels) float32
    times = f[f'{key}/timestamps'][:]   # (n_times,) seconds
    elec_ids = f[f'{key}/electrodes'][:] # (n_channels,) electrode IDs

# fs from probes.csv
probes = pd.read_csv(r'W:\PublicData\Allen\NPVisCode\probes.csv')
fs = probes.loc[probes['id'] == probe_id, 'lfp_sampling_rate'].item()

lfp = xr.DataArray(
    data.T,  # transpose to (n_channels, n_times)
    dims=[DIM_CHAN, DIM_TIME],
    coords={DIM_TIME: times, DIM_CHAN: elec_ids},
    attrs={ATTR_FS: fs, ATTR_UNITS: 'uV'},
)
```

**Spikes → `pynapple.TsGroup`**:
```python
import h5py, numpy as np, pynapple as nap

session_path = rf'W:\PublicData\Allen\NPVisCode\session_{session_id}\session_{session_id}.nwb'

with h5py.File(session_path, 'r') as f:
    spike_times_flat = f['units/spike_times'][:]
    spike_times_index = f['units/spike_times_index'][:]
    unit_ids = f['units/id'][:]
    quality = np.array([q.decode() for q in f['units/quality'][:]])

# Reconstruct ragged spike times per unit
ends = spike_times_index.astype(int)
starts = np.concatenate([[0], ends[:-1]])
spikes_dict = {
    uid: nap.Ts(spike_times_flat[s:e])
    for uid, s, e, q in zip(unit_ids, starts, ends, quality)
    if q == 'good'
}
units = nap.TsGroup(spikes_dict)
```

**Running speed → `xr.DataArray`**:
```python
from mimo_pack.consts import DIM_TIME, ATTR_FS, ATTR_UNITS

with h5py.File(session_path, 'r') as f:
    speed = f['processing/running/running_speed/data'][:]
    speed_t = f['processing/running/running_speed/timestamps'][:]

running = xr.DataArray(speed, dims=[DIM_TIME], coords={DIM_TIME: speed_t},
                       attrs={ATTR_UNITS: 'm/s'})
```

**Stimulus intervals → `pynapple.IntervalSet`**:
```python
stim_name = 'drifting_gratings_presentations'  # example

with h5py.File(session_path, 'r') as f:
    starts = f[f'intervals/{stim_name}/start_time'][:]
    stops  = f[f'intervals/{stim_name}/stop_time'][:]

epochs = nap.IntervalSet(start=starts, end=stops)
```

---

## IBL (International Brain Laboratory)

**Access**: ONE API — `pip install ONE-api` (not in `environment.yml`).

### Setup

```python
from one.api import ONE
one = ONE()  # prompts for server URL and credentials on first run; caches locally after that
```

### Discovering Available Data

```python
# List all available sessions
sessions = one.search()

# List all datasets for a session
eid = '...'  # experiment ID (UUID string)
datasets = one.list_datasets(eid)
```

### Standard ONE Dataset Keys

| Category | Dataset key | Description |
|---|---|---|
| Spikes | `spikes.times` | Spike times in seconds (all clusters concatenated) |
| Spikes | `spikes.clusters` | Cluster ID for each spike |
| Spikes | `spikes.amps` | Spike amplitudes |
| Clusters | `clusters.channels` | Peak channel per cluster |
| Clusters | `clusters.brainAcronyms` | Brain region per cluster (Allen CCF) |
| Clusters | `clusters.metrics` | Quality metrics per cluster |
| Trials | `trials.goCue_times` | Go cue onset times |
| Trials | `trials.stimOn_times` | Stimulus onset times |
| Trials | `trials.response_times` | Response times |
| Trials | `trials.feedbackType` | +1 correct / -1 incorrect |
| Wheel | `wheel.timestamps` | Wheel sample timestamps |
| Wheel | `wheel.position` | Wheel position in radians |

### Reformatting Patterns

**Spikes → `pynapple.TsGroup`**:
```python
import numpy as np, pynapple as nap

spike_times = one.load_dataset(eid, 'spikes.times')
spike_clusters = one.load_dataset(eid, 'spikes.clusters')
cluster_ids = np.unique(spike_clusters)

spikes_dict = {
    int(cid): nap.Ts(spike_times[spike_clusters == cid])
    for cid in cluster_ids
}
units = nap.TsGroup(spikes_dict)
```

**Wheel → `xr.DataArray`**:
```python
import numpy as np, xarray as xr
from mimo_pack.consts import DIM_TIME, ATTR_UNITS

wheel_t = one.load_dataset(eid, 'wheel.timestamps')
wheel_pos = one.load_dataset(eid, 'wheel.position')

wheel = xr.DataArray(wheel_pos, dims=[DIM_TIME], coords={DIM_TIME: wheel_t},
                     attrs={ATTR_UNITS: 'rad'})
```

**Trials → `pynapple.IntervalSet`**:
```python
stim_on  = one.load_dataset(eid, 'trials.stimOn_times')
feedback = one.load_dataset(eid, 'trials.response_times')

epochs = nap.IntervalSet(start=stim_on, end=feedback)
```

---

## Data Model Conformance

All outputs passed to `mimo_pack.analysis.*` or `mimo_pack.plot.*` must conform to these rules:

| Object type | Required dims | Required attrs | Source in `mimo_pack.consts` |
|---|---|---|---|
| LFP `xr.DataArray` | `['time', 'channel']` | `fs` (float), `units` (str) | `DIM_TIME`, `DIM_CHAN`, `ATTR_FS`, `ATTR_UNITS` |
| Continuous signal `xr.DataArray` | `['time']` | `units` (str) | `DIM_TIME`, `ATTR_UNITS` |
| Spike collection | `pynapple.TsGroup` | — | — |
| Epochs / trial windows | `pynapple.IntervalSet` | — | — |
| Metadata / trial parameters | `pd.DataFrame` | — | — |

- Import dim/attr names: `from mimo_pack.consts import DIM_TIME, DIM_CHAN, ATTR_FS, ATTR_UNITS`
- Verify loaded data with `print(da.dims, da.attrs, da.shape)` before passing downstream.
