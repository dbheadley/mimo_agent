# Session management for mimo_pack
# Author: Drew Headley
# Date: 2025-06-16

import json
import re
from pathlib import Path

import pandas as pd


# Matches SpikeGLX session root folders: <YY-MM-DD>_<Subject>_<Task>_g<N>
_SESSION_DIR_RE = re.compile(
    r'^(?P<date>\d{2,4}-\d{1,2}-\d{1,2})_(?P<subject>[^_]+)_(?P<task>.+)_g\d+$'
)


def scan_sessions(root_dir, exclude_dirs=['images', 'skip'], verbose=True):
    """
    Recursively scan a directory for SpikeGLX recording session folders.

    A session folder is identified by the naming convention
    ``<YY-MM-DD>_<Subject>_<Task>_g<N>``.

    Parameters
    ----------
    root_dir : str or Path
        Root directory to scan.
    exclude_dirs : str, list of str, or None, optional
        Directory name(s) to skip entirely during the recursive scan.
        Any directory whose name matches will not be entered or searched.
        Defaults to ``'images'``.
    verbose : bool, optional
        If ``True``, display a ``tqdm`` progress bar while scanning directories.
        Defaults to ``False``.

    Returns
    -------
    pandas.DataFrame
        One row per session folder with columns:

        - ``path`` : absolute path to the session folder
        - ``date`` : date string from folder name (``YY-MM-DD``)
        - ``subject`` : subject identifier
        - ``task`` : task name
        - ``n_probes`` : number of imec probe subfolders
        - ``probe_indices`` : list of imec probe indices present (e.g. ``[0, 1]``)
        - ``has_nidq_raw`` : raw NIDQ binary (``*_t0.nidq.bin``) present
        - ``has_nidq_dclut`` : NIDQ time-alignment dclut JSON present
        - ``has_video`` : behavior/video subfolder present
        - ``has_video_tifs`` : ``image/`` subdirectory present in video folder
        - ``has_video_file`` : assembled video file (``.mp4`` or ``.avi``) present in video folder
        - ``has_video_csv`` : per-frame info ``.csv`` file present in video folder
        - ``has_video_qc`` : ``qc/`` subfolder present in video folder
        - ``has_session_config`` : ``session_config.json`` present
        - ``has_states`` : behavioral states ``states.nc`` present
        - ``has_catgt`` : CatGT-aligned ``tcat`` AP binary present in any probe
        - ``has_ap_dclut`` : probe-level AP dclut JSON present in any probe
        - ``has_kilosort`` : ``kilosort4/`` output folder present in any probe
        - ``has_curation`` : manually curated ``_sas.npy`` files present in any probe
        - ``has_mimo_units`` : mimo-extracted unit arrays (``.su.npz`` / ``.mu.npz``) present in any probe
        - ``processing_level`` : highest processing stage reached

        Processing levels in ascending order:
        ``"raw"`` -> ``"aligned"`` -> ``"sorted"`` -> ``"curated"`` -> ``"extracted"`` -> ``"analyzed"``
    """
    root_dir = Path(root_dir)
    records = []

    if exclude_dirs is None:
        _exclude = set()
    elif isinstance(exclude_dirs, str):
        _exclude = {exclude_dirs}
    else:
        _exclude = set(exclude_dirs)

    def _iter_dirs(path):
        for entry in sorted(path.iterdir()):
            if not entry.is_dir() or entry.name in _exclude:
                continue
            yield entry
            yield from _iter_dirs(entry)

    if verbose:
        from tqdm import tqdm
        _scan = tqdm(_iter_dirs(root_dir), desc='Scanning directories', unit='dir')
    else:
        _scan = _iter_dirs(root_dir)

    for candidate in _scan:
        match = _SESSION_DIR_RE.match(candidate.name)
        if match is None:
            continue

        date = match.group('date')
        subject = match.group('subject')
        task = match.group('task')

        # --- root-level checks ---
        has_nidq_raw = any(candidate.glob('*_t0.nidq.bin'))
        has_nidq_dclut = any(candidate.glob('*_t0.nidq_dclut.json'))
        has_session_config = (candidate / 'session_config.json').exists()
        has_states = (candidate / 'states.nc').exists()

        # Video folder has the same name but without the trailing _gN suffix
        video_folder_name = re.sub(r'_g\d+$', '', candidate.name)
        video_dir = candidate / video_folder_name
        has_video = video_dir.is_dir()
        has_video_tifs = has_video and (video_dir / 'image').is_dir()
        has_video_file = has_video and (
            any(video_dir.glob('*.mp4')) or any(video_dir.glob('*.avi'))
        )
        has_video_csv = has_video and any(video_dir.glob('*.csv'))
        has_video_qc = has_video and (video_dir / 'qc').is_dir()

        # --- probe-level checks ---
        probe_dirs = sorted(candidate.glob('*_imec[0-9]*'))
        probe_indices = []
        for probe_dir in probe_dirs:
            m = re.search(r'_imec(\d+)$', probe_dir.name)
            if m:
                probe_indices.append(int(m.group(1)))

        has_catgt = False
        has_ap_dclut = False
        has_kilosort = False
        has_curation = False
        has_mimo_units = False

        for probe_dir in probe_dirs:
            if any(probe_dir.glob('*_tcat.imec*.ap.bin')):
                has_catgt = True
            if any(probe_dir.glob('*_tcat.imec*.ap_dclut.json')):
                has_ap_dclut = True
            ks_dir = probe_dir / 'kilosort4'
            if ks_dir.is_dir():
                has_kilosort = True
                if any(ks_dir.glob('*_sas.npy')):
                    has_curation = True
            if any(probe_dir.glob('*.ap.su.npz')) or any(probe_dir.glob('*.ap.mu.npz')):
                has_mimo_units = True

        # --- highest processing stage ---
        if has_states:
            processing_level = 'analyzed'
        elif has_mimo_units:
            processing_level = 'extracted'
        elif has_curation:
            processing_level = 'curated'
        elif has_kilosort:
            processing_level = 'sorted'
        elif has_catgt or has_nidq_dclut:
            processing_level = 'aligned'
        else:
            processing_level = 'raw'

        records.append({
            'path': candidate,
            'date': date,
            'subject': subject,
            'task': task,
            'n_probes': len(probe_indices),
            'probe_indices': probe_indices,
            'has_nidq_raw': has_nidq_raw,
            'has_nidq_dclut': has_nidq_dclut,
            'has_video': has_video,
            'has_video_tifs': has_video_tifs,
            'has_video_file': has_video_file,
            'has_video_csv': has_video_csv,
            'has_video_qc': has_video_qc,
            'has_session_config': has_session_config,
            'has_states': has_states,
            'has_catgt': has_catgt,
            'has_ap_dclut': has_ap_dclut,
            'has_kilosort': has_kilosort,
            'has_curation': has_curation,
            'has_mimo_units': has_mimo_units,
            'processing_level': processing_level,
        })

    return pd.DataFrame(records)


class Session:
    """
    Session manager that reads configuration from session_config.json.
    
    This class provides a standardized way to access file paths and analysis
    parameters for a recording session, decoupling the codebase from
    hardcoded file naming conventions.
    
    Parameters
    ----------
    session_dir : str or Path
        Path to the session directory containing session_config.json.
        
    Attributes
    ----------
    dir : Path
        The session directory path.
    config_path : Path
        Path to the session_config.json file.
    config : dict
        The loaded configuration dictionary.
        
    Examples
    --------
    >>> sess = Session("/data/mouse1")
    >>> ap_files = sess.find_files('ap_pattern')
    >>> lfp_params = sess.get_params('lfp')
    """
    
    def __init__(self, session_dir):
        """Initialize a Session by loading the session_config.json from the given directory.

        Parameters
        ----------
        session_dir : str or Path
            Path to the directory containing a ``session_config.json`` file.

        Raises
        ------
        FileNotFoundError
            If no ``session_config.json`` exists in *session_dir*.
        """
        self.dir = Path(session_dir)
        self.config_path = self.dir / 'session_config.json'
        
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"No session_config.json found in {session_dir}. "
                "Please create a configuration file or use Session.create_default_config()."
            )
            
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
    
    def find_files(self, key):
        """
        Find files matching a pattern defined in the config.
        
        Parameters
        ----------
        key : str
            The key in config['paths'] to look up the glob pattern.
            
        Returns
        -------
        list of Path
            List of full paths to files matching the pattern.
            
        Raises
        ------
        ValueError
            If the key is not found in the config paths.
            
        Examples
        --------
        >>> sess.find_files('ap_pattern')  # Returns all AP files
        >>> sess.find_files('lfp_pattern')  # Returns all LFP files
        """
        pattern = self.config.get('paths', {}).get(key)
        if not pattern:
            raise ValueError(
                f"Path key '{key}' not found in config. "
                f"Available keys: {list(self.config.get('paths', {}).keys())}"
            )
        return list(self.dir.glob(f"**/{pattern}"))
    
    def get_params(self, analysis_type):
        """
        Get parameters for a specific analysis type.
        
        Parameters
        ----------
        analysis_type : str
            The analysis type key to look up in config['parameters'].
            
        Returns
        -------
        dict
            Parameter dictionary for the specified analysis type.
            Returns empty dict if not found.
            
        Examples
        --------
        >>> lfp_params = sess.get_params('lfp')
        >>> # Returns {'resample_rate': 1000, 'filter_opts': {...}}
        """
        return self.config.get('parameters', {}).get(analysis_type, {})
    
    def get_metadata(self, key=None):
        """
        Get session metadata.
        
        Parameters
        ----------
        key : str, optional
            Specific metadata key to retrieve. If None, returns all metadata.
            
        Returns
        -------
        dict or any
            The metadata dictionary or specific value.
        """
        metadata = self.config.get('metadata', {})
        if key is None:
            return metadata
        return metadata.get(key)
    
    def get_dclut_path(self, data_file):
        """
        Get the corresponding dclut file path for a given AP, LFP, or NIDQ file.
        
        Parameters
        ----------
        data_file : str or Path
            Path to the data file (.bin) for which to find the dclut file.
            
        Returns
        -------
        Path
            Path to the corresponding dclut file.
            
        Raises
        ------
        FileNotFoundError
            If the dclut file doesn't exist.
        ValueError
            If the input file doesn't have a .bin extension.
            
        Examples
        --------
        >>> ap_file = "/data/session/imec0/recording_t0.imec0.ap.bin"
        >>> dclut_path = sess.get_dclut_path(ap_file)
        >>> # Returns: /data/session/imec0/recording_t0.imec0.ap_dclut.json
        
        >>> nidq_file = "/data/session/recording_t0.nidq.bin" 
        >>> dclut_path = sess.get_dclut_path(nidq_file)
        >>> # Returns: /data/session/recording_t0.nidq_dclut.json
        """
        data_file = Path(data_file)
        
        if data_file.suffix != '.bin':
            raise ValueError(f"Input file must have .bin extension, got: {data_file.suffix}")
        
        # Get dclut suffix from config, default to '_dclut.json'
        dclut_suffix = self.config.get('paths', {}).get('dclut_suffix', '_dclut.json')
        
        # Replace .bin with dclut suffix (use string replacement instead of with_suffix)
        dclut_path = Path(str(data_file).replace('.bin', dclut_suffix))
        
        if not dclut_path.exists():
            raise FileNotFoundError(f"Dclut file not found: {dclut_path}")
        
        return dclut_path
    
    def save(self, backup=True):
        """
        Save the current configuration to the session_config.json file.
        
        Parameters
        ----------
        backup : bool, optional
            Whether to create a backup of the existing file before saving.
            Default is True. The backup will be saved as session_config.json.backup.
            
        Returns
        -------
        Path
            Path to the saved config file.
            
        Examples
        --------
        >>> sess.config['parameters']['new_param'] = {'value': 42}
        >>> sess.save()  # Saves with backup
        >>> sess.save(backup=False)  # Saves without backup
        """
        if backup and self.config_path.exists():
            backup_path = self.config_path.with_suffix('.json.backup')
            backup_path.write_text(self.config_path.read_text())
        
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        
        return self.config_path
    
    @staticmethod
    def create_default_config(session_dir, overwrite=False):
        """
        Create a default session_config.json file in the specified directory.
        
        Parameters
        ----------
        session_dir : str or Path
            Path to the session directory.
        overwrite : bool, optional
            Whether to overwrite an existing config file. Default is False.
            
        Returns
        -------
        Path
            Path to the created config file.
            
        Raises
        ------
        FileExistsError
            If config file exists and overwrite is False.
        """
        session_dir = Path(session_dir)
        config_path = session_dir / 'session_config.json'
        
        if config_path.exists() and not overwrite:
            raise FileExistsError(
                f"Config file already exists at {config_path}. "
                "Set overwrite=True to replace it."
            )
        
        default_config = {
            "paths": {
                "ap_pattern": "*t0.imec*.ap.bin",
                "lfp_pattern": "*t0.imec*.lfp.bin",
                "nidq_pattern": "*t0.nidq.bin",
                "dclut_suffix": "_dclut.json"
            },
            "parameters": {
                "lfp": {
                    "lfp_cutoff": 200, 
                    "lfp_fs": 1000,
                },
                "csd": {
                    "spacing": 100,
                    "method": "standard"
                },
                "evoked_lfp": {
                    "pre": 0.01,
                    "post": 0.03,
                    "spacing": 100
                }
            },
            "metadata": {
                "subject": "",
                "probe_type": "NP2.0",
                "date": "",
                "notes": ""
            }
        }
        
        session_dir.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        return config_path
    
    def __repr__(self):
        """Return a concise string representation of the Session."""
        return f"Session('{self.dir}')"
