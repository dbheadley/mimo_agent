"""Organize in vitro data
Author: Drew B. Headley
"""

import os
import pandas as pd
from mimo_pack.util.files import find_all_matching_files, split_file_path
from .abf import load_abf_settings, get_clamp_mode, summarise_clamp_mode

def invitro_files_df(rootdir: str, required_ext='.abf') -> pd.DataFrame:
    """
    Scans a directory of in vitro experimental files, extracts metadata from their paths, 
    and returns a pivoted DataFrame.

    Parameters
    ----------
    rootdir : str
        The root directory path to search for files.
    required_ext : str, list of str, or None, optional
        File extension(s) that must be present for a row to be kept in the final dataframe.
        If None, no extension is required. Default is '.abf'.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by 'subject', 'slice', and 'cell', with columns for each file extension.
    
    Notes
    -----
    """

    # Find all files matching the pattern in the specified directory
    files = find_all_matching_files(rootdir, '.*')
    files = pd.DataFrame(files, columns=['file_path'])

    # Create a column for the last, second to last, and third to last directory names.
    files['cell'] = files['file_path'].apply(lambda x: split_file_path(x)[-2])
    files['slice'] = files['file_path'].apply(lambda x: split_file_path(x)[-3])
    files['subject'] = files['file_path'].apply(lambda x: split_file_path(x)[-4])

    # Add column for file name extension
    files['ext'] = files['file_path'].apply(lambda x: os.path.splitext(x)[1])

    # Pivot the dataframe to have one row per cell, slice, and subject
    # and a column for each type of file path
    files = files.pivot_table(index=['subject', 'slice', 'cell'],
                              values='file_path',
                              columns='ext',
                              aggfunc=list).reset_index()

    # Handle required_ext argument
    if required_ext is None:
        pass  # No filtering
    else:
        if isinstance(required_ext, str):
            required_ext = [required_ext]
       
        # Keep only rows where all required extensions are present (not NaN)
        mask = files[required_ext].notna().all(axis=1)
        files = files[mask]

    # if .abf column is present, create a column with the settings
    # of each abf file
    if '.abf' in files.columns:
        files['protocols'] = files['.abf'].apply(lambda x: load_abf_settings_dict(x) if x else {})
    else:
        files['protocols'] = None
        
    return files

# function that accepts a list of abf files, loads the settings for each file, then return
# a dictionary where the keys are the protocols and the value are the lists of files
def load_abf_settings_dict(abf_files: list) -> dict:
    
    """
    Organizes ABF files by protocol type.

    Given a list of ABF file paths, this function loads the settings for each file,
    extracts the protocol name, and groups the files into a dictionary where each key
    is a protocol name and the value is a list of dicts, one per file, with keys:

    - ``'file'`` (str): path to the ABF file.
    - ``'clamp_mode'`` (str): ``'V-Clamp'``, ``'I-Clamp'``, ``'both'``, or
      ``'unknown'``, summarised across all ADC channels in the file.
      ``'I=0'`` is treated as ``'I-Clamp'``.

    Parameters
    ----------
    abf_files : list of str
        List of ABF file paths.

    Returns
    -------
    dict
        Dictionary mapping protocol names (str) to lists of file-info dicts.
    """
    settings_dict = {}
    for abf_file in abf_files:
        settings = load_abf_settings(abf_file)
        protocol = settings.get('protocol', 'unknown')

        # Determine clamp mode for this file from telegraph metadata
        try:
            channel_modes = [ch_info['mode'] for ch_info in get_clamp_mode(abf_file)]
            clamp_mode = summarise_clamp_mode(channel_modes)
        except Exception:
            clamp_mode = 'unknown'

        if protocol not in settings_dict:
            settings_dict[protocol] = []
        settings_dict[protocol].append({'file': abf_file, 'clamp_mode': clamp_mode})
    return settings_dict


def summarize_invitro_protocols(cells_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize protocol usage across all cells in an in-vitro experiment.

    Each unique (protocol, clamp_mode) combination gets its own row, so that
    a protocol run in both V-Clamp and I-Clamp appears as two separate entries.

    Parameters
    ----------
    cells_df : pandas.DataFrame
        DataFrame returned by :func:`invitro_files_df`.  Must contain a
        ``'protocols'`` column whose values are dicts mapping protocol name
        to a list of file-info dicts (as produced by
        :func:`load_abf_settings_dict`).

    Returns
    -------
    pandas.DataFrame
        One row per (protocol, clamp_mode) combination, sorted by ``n_total``
        descending, with columns:

        - ``'protocol'``: protocol name (str).
        - ``'clamp_mode'``: clamp mode (``'V-Clamp'``, ``'I-Clamp'``,
          ``'both'``, or ``'unknown'``).
        - ``'n_cells'``: number of cells that have at least one file for this
          (protocol, clamp_mode) combination (int).
        - ``'n_total'``: total number of files across all cells (int).
        - ``'files_per_cell'``: list of per-cell file counts (list of int).

    Examples
    --------
    >>> cells = invitro_files_df('/data/invitro')
    >>> summarize_invitro_protocols(cells)
    """
    counts: dict[tuple, dict] = {}

    for _, row in cells_df.iterrows():
        proto_dict = row.get('protocols') or {}
        if not isinstance(proto_dict, dict):
            continue
        for proto, files in proto_dict.items():
            if not isinstance(files, list):
                files = []
            # Group files by clamp_mode within this protocol
            mode_files: dict[str, int] = {}
            for f in files:
                mode = f.get('clamp_mode', 'unknown') if isinstance(f, dict) else 'unknown'
                mode_files[mode] = mode_files.get(mode, 0) + 1
            for mode, n in mode_files.items():
                key = (proto, mode)
                if key not in counts:
                    counts[key] = {'n_cells': 0, 'n_total': 0, 'files_per_cell': []}
                counts[key]['n_cells'] += 1
                counts[key]['n_total'] += n
                counts[key]['files_per_cell'].append(n)

    rows = [
        {
            'protocol': proto,
            'clamp_mode': mode,
            'n_cells': v['n_cells'],
            'n_total': v['n_total'],
            'files_per_cell': v['files_per_cell'],
        }
        for (proto, mode), v in counts.items()
    ]
    df = pd.DataFrame(rows, columns=['protocol', 'clamp_mode', 'n_cells', 'n_total', 'files_per_cell'])
    df = df.sort_values('n_total', ascending=False).reset_index(drop=True)
    return df


def extract_protocol_files(
    cells_df: pd.DataFrame,
    protocols: list[str],
    clamp_mode: str | None = None,
) -> pd.DataFrame:
    """Flatten cells × protocols into one row per (cell, file).

    Iterates over rows of *cells_df* (as returned by
    :func:`invitro_files_df`) and expands the ``'protocols'`` dict so
    that each individual ABF file becomes its own row.  This is useful
    for feeding file-level DataFrames into
    :func:`mimo_pack.util.dataframe.pipe_columns` pipelines.

    Parameters
    ----------
    cells_df : pd.DataFrame
        DataFrame returned by :func:`invitro_files_df`.  Must contain a
        ``'protocols'`` column whose values are dicts mapping protocol
        names to lists of file-info dicts (as produced by
        :func:`load_abf_settings_dict`).
    protocols : list of str
        Protocol names to extract.  Only files belonging to these
        protocols are included.
    clamp_mode : str or None, optional
        If given, keep only files whose ``'clamp_mode'`` matches this
        value (e.g. ``'V-Clamp'``, ``'I-Clamp'``).  If ``None`` (default),
        all clamp modes are kept.

    Returns
    -------
    pd.DataFrame
        One row per file with columns ``'cell_idx'``, ``'subject'``,
        ``'slice'``, ``'cell'``, ``'protocol'``, and ``'fpath'``.
        Returns an empty DataFrame (with the same columns) when no
        matching files are found.

    Examples
    --------
    >>> cells = invitro_files_df('/data/invitro')
    >>> vc_files = extract_protocol_files(cells, ['Left_BlueLED_Tonic_IC'],
    ...                                   clamp_mode='V-Clamp')
    >>> vc_files.head()
    """
    rows: list[dict] = []
    for idx, row in cells_df.iterrows():
        proto = row.get('protocols', {})
        if not isinstance(proto, dict):
            continue
        for protocol in protocols:
            files = proto.get(protocol, [])
            if clamp_mode is not None:
                files = [f for f in files if f.get('clamp_mode') == clamp_mode]
            for f in files:
                rows.append({
                    'cell_idx': idx,
                    'subject':  row.get('subject'),
                    'slice':    row.get('slice'),
                    'cell':     row.get('cell'),
                    'protocol': protocol,
                    'fpath':    f['file'],
                })
    columns = ['cell_idx', 'subject', 'slice', 'cell', 'protocol', 'fpath']
    return pd.DataFrame(rows, columns=columns)
