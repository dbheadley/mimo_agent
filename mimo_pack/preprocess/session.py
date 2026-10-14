# Functions for preprocessing entire sessions
# Author: Drew Headley, 2024

import os
import pdb
import pandas as pd
from mimo_pack.util.files import find_all_matching_files
from mimo_pack.preprocess.lfp import make_lfp_file_dclut, make_lfp_dclut
from mimo_pack.preprocess.time import align_sync_dclut
from mimo_pack.fileio.spikeglx import dclut_from_meta
from mimo_pack.core.session import Session


def preprocess_session(sess_dir, sync_ap=None, sync_nidq=None, force_dclut=False, force_lfp=False):
    """
    Preprocess a session using a session_config.json configuration file.
    
    This is the recommended method for preprocessing as it uses the Session
    configuration system, avoiding hardcoded file patterns.

    Parameters
    ----------
    sess_dir : str
        The session directory containing session_config.json.
    sync_ap : dict, optional
        The sync channel for the AP files as specified for dclut.
        If None, uses default {'channel': [384]}.
    sync_nidq : dict, optional
        The sync channel for the NIDQ file as specified for dclut.
        If None, uses default {'channel': [13]}.
    force_dclut : bool, optional
        If True, recreate AP and NIDQ dclut files even when they already
        exist and re-run sync alignment.  Default False (skip existing).
    force_lfp : bool, optional
        If True, recreate LFP bin and dclut files even when they already
        exist.  Default False (skip existing).

    Returns
    -------
    df : pd.DataFrame
        A dataframe with file names, dclut files, and types.

    See Also
    --------
    mimo_pack.core.session.Session : Session configuration manager.
    preprocess_spikeglx : Legacy preprocessing function with hardcoded patterns.
    """

    if sync_ap is None:
        sync_ap = {'channel': [384]}
    if sync_nidq is None:
        sync_nidq = {'channel': [13]}

    print(f"Starting preprocessing for session: {sess_dir}")
    print(f"Run started at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load session configuration
    sess = Session(sess_dir)

    # Get files using the config's patterns
    print(f"Identifying files in session directory: {sess_dir}")
    ap_files = sess.find_files('ap_pattern')
    nidq_files = sess.find_files('nidq_pattern')
    
    if not ap_files:
        raise FileNotFoundError(f"No AP files found matching pattern in {sess_dir}")
    if not nidq_files:
        raise FileNotFoundError(f"No NIDQ files found matching pattern in {sess_dir}")
    
    nidq_file = nidq_files[0]  # there should be only one nidq file

    # Convert Path objects to strings for compatibility
    ap_files = [str(f) for f in ap_files]
    nidq_file = str(nidq_file)

    print("ap files:")
    for f in ap_files:
        print(f)

    print("nidq file:")
    print(nidq_file)

    # Create dclut json files, skipping existing outputs when force_dclut=False
    ap_dclut_files = []
    ap_dclut_fresh = []  # True if the dclut was (re)created this run
    for f in ap_files:
        expected_dclut = f.replace('.bin', '_dclut.json')
        if not force_dclut and os.path.exists(expected_dclut):
            print(f"\u23ed Skipping AP dclut (already exists): {os.path.basename(expected_dclut)}")
            ap_dclut_files.append(expected_dclut)
            ap_dclut_fresh.append(False)
        else:
            ap_dclut_files.append(dclut_from_meta(f))
            ap_dclut_fresh.append(True)

    expected_nidq_dclut = nidq_file.replace('.bin', '_dclut.json')
    if not force_dclut and os.path.exists(expected_nidq_dclut):
        print(f"\u23ed Skipping NIDQ dclut (already exists): {os.path.basename(expected_nidq_dclut)}")
        nidq_dclut_file = expected_nidq_dclut
        nidq_dclut_fresh = False
    else:
        nidq_dclut_file = dclut_from_meta(nidq_file)
        nidq_dclut_fresh = True

    # Align the times across all the dclut files to the first ap file
    # Only align files that were freshly (re)created this run
    ap_r = ap_dclut_files[0]
    sync_scale_name = 'time'
    print(f"Aligning all files to {os.path.basename(ap_r)}")
    for ap_t in range(1, len(ap_dclut_files)):
        file_t = ap_dclut_files[ap_t]
        if ap_dclut_fresh[ap_t]:
            print(f'Aligning {os.path.basename(file_t)} to {os.path.basename(ap_r)}')
            align_sync_dclut(file_t, ap_r, sync_ap, sync_ap, sync_scale_name, verbose=True)
        else:
            print(f'\u23ed Skipping alignment (dclut exists): {os.path.basename(file_t)}')

    # Align the NIDQ file to the first AP file
    if sync_nidq is not None:
        if nidq_dclut_fresh:
            print(f'Aligning {os.path.basename(nidq_dclut_file)} to {os.path.basename(ap_r)}')
            align_sync_dclut(nidq_dclut_file, ap_r, sync_nidq, sync_ap, sync_scale_name, verbose=True)
        else:
            print(f'\u23ed Skipping NIDQ alignment (dclut exists): {os.path.basename(nidq_dclut_file)}')

    # Get LFP parameters from config (if available)
    lfp_params = sess.get_params('lfp')

    # Make LFP files from the imec AP files with dclut.
    # Binary creation (slow) and dclut creation (fast) are checked independently
    # so a missing dclut is regenerated even when the binary already exists.
    lfp_files = []
    lfp_dclut_files = []
    print("Making LFP files from AP files:")
    for file in ap_dclut_files:
        lfp_bin = file.replace('ap_dclut.json', 'lfp.bin')
        lfp_dclut = lfp_bin.replace('.bin', '_dclut.json')
        lfp_bin_exists = os.path.exists(lfp_bin)
        lfp_dclut_exists = os.path.exists(lfp_dclut)

        if not force_lfp and lfp_bin_exists and lfp_dclut_exists:
            # Both outputs present — skip entirely
            print(f'\u23ed Skipping LFP (already exists): {os.path.basename(lfp_bin)}')
        elif not force_lfp and lfp_bin_exists and not lfp_dclut_exists:
            # Binary present but dclut missing — regenerate just the dclut (fast)
            print(f'\u23ed LFP binary exists, regenerating missing dclut: {os.path.basename(lfp_dclut)}')
            dclut_params = {k: lfp_params[k] for k in ('lfp_fs', 'sync', 'time_scale') if k in lfp_params}
            make_lfp_dclut(file, lfp_bin, **dclut_params)
        else:
            # Create (or force-recreate) both binary and dclut
            print(f'Processing {os.path.basename(file)}')
            make_lfp_file_dclut(file, lfp_bin, verbose=True, **lfp_params)

        lfp_files.append(lfp_bin)
        lfp_dclut_files.append(lfp_dclut)

    # Create a dataframe with the file names
    files = []
    print("Creating dataframe with file names, dclut files, and types")
    for f in range(len(ap_files)):
        files.append([sess_dir, ap_files[f], ap_dclut_files[f], 'ap'])
    for f in range(len(lfp_files)):
        files.append([sess_dir, lfp_files[f], lfp_dclut_files[f], 'lfp'])
    files.append([sess_dir, nidq_file, nidq_dclut_file, 'nidq'])

    df = pd.DataFrame(files, columns=['session_dir', 'file', 'dclut_file', 'type'])
    return df


def preprocess_spikeglx(sess_dir, sync_ap={'channel': [384]}, sync_nidq={'channel': [13]}, 
                        use_catgt_version=False):
    """
    DEPRECATED: Use preprocess_session() instead for new workflows.
    
    Legacy preprocessing function that has been updated to use the Session
    configuration system internally while maintaining backwards compatibility.
    
    Takes a session directory and preprocesses all the imec files in the directory. This 
    includes creating LFP files from the AP files, creating dclut json files for all the 
    files, and aligning the times across all the files to the first AP file. The function 
    returns a dataframe with the file names, the dclut json files, and the type of file.

    Parameters
    ----------
    sess_dir : str
        The session directory

    Optional
    --------
    sync_ap : dict
        The sync channel for the AP files as specified for dclut. 
        Default is {'channel': [384]}
    sync_nidq : dict
        The sync channel for the NIDQ file as specified for dclut. If None, no alignment is done.
        Default is {'channel': [13]}
    use_catgt_version : bool
        Whether to use the CatGT version of the files. 
        Default is False

    Returns
    -------
    df : pd.DataFrame
        A dataframe with the file names, the dclut json files, and the type of file
    
    Notes
    -----
    This function now internally attempts to use the Session configuration system.
    If session_config.json exists, it will use that. Otherwise, it falls back to
    hardcoded patterns for backwards compatibility.
    """
    
    # print date and time that run started
    print("Starting preprocessing for session: {}".format(sess_dir))
    print("Run started at: {}".format(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')))
    
    # Try to use Session configuration first
    try:
        sess = Session(sess_dir)
        print("Found session_config.json, using Session configuration")
        
        # Use the session configuration
        return preprocess_session(sess_dir, sync_ap, sync_nidq)
        
    except FileNotFoundError:
        print("No session_config.json found, falling back to hardcoded patterns")
        print("Consider creating a session_config.json file for better flexibility")
        
        # Fall back to legacy hardcoded patterns
        print("Identifying files in session directory: {}".format(sess_dir))
        if use_catgt_version:
            ap_files = find_all_matching_files(sess_dir, r'tcat\.imec([0-9]+)\.ap\.bin')
        else:
            ap_files = find_all_matching_files(sess_dir, r't0\.imec([0-9]+)\.ap\.bin')
        nidq_file = find_all_matching_files(sess_dir, r't0\.nidq\.bin')[0] # there should be only one nidq file

    print("ap files:")
    for f in ap_files:
        print(f)

    print("nidq file:")
    print(nidq_file)

    # create dclut json files
    ap_dclut_files = [dclut_from_meta(f) for f in ap_files]
    nidq_dclut_file = dclut_from_meta(nidq_file)

    # align the times across all the dclut files to the first ap file
    ap_r = ap_dclut_files[0] 
    sync_scale_name = 'time'
    print("Aligning all files to {}".format(os.path.basename(ap_r)))
    for ap_t in range(1, len(ap_dclut_files)):
        file_t = ap_dclut_files[ap_t]
        print('Aligning {} to {}'.format(os.path.basename(file_t), os.path.basename(ap_r)))
        align_sync_dclut(file_t, ap_r, sync_ap, sync_ap, sync_scale_name, verbose=True)
    
    # Align the NIDQ file to the first AP file
    if sync_nidq is not None:
        print('Aligning {} to {}'.format(os.path.basename(nidq_dclut_file), os.path.basename(ap_r)))
        align_sync_dclut(nidq_dclut_file, ap_r, sync_nidq, sync_ap, sync_scale_name, verbose=True)

    # make a LFP file from the imec AP files with dclut 
    lfp_files = []
    lfp_dclut_files = []
    # make a LFP file from the imec files 
    print("Making LFP files from AP files:")
    for file in ap_dclut_files:
        print('Processing {}'.format(os.path.basename(file)))
        lfp_files.append(file.replace('ap_dclut.json','lfp.bin'))
        lfp_dclut_files.append(make_lfp_file_dclut(file, lfp_files[-1], verbose=True))

    # create a dataframe with the file names, first column is the session dir, second is the probe directory, 
    # third is the file, fourth is its dclut json file, and last is the type of file
    files = []
    print("Creating dataframe with file names, dclut files, and types")
    for f in range(len(ap_files)):
        files.append([sess_dir, ap_files[f], ap_dclut_files[f], 'ap'])
    for f in range(len(lfp_files)):
        files.append([sess_dir, lfp_files[f], lfp_dclut_files[f], 'lfp'])
    files.append([sess_dir, nidq_file, nidq_dclut_file, 'nidq'])

    df = pd.DataFrame(files, columns=['session_dir', 'file', 'dclut_file', 'type'])
    return df



def get_session_dict(sess_dir, use_catgt_version=False):
    """
    DEPRECATED: Consider using Session class directly for new workflows.
    
    Takes a session directory and returns a dictionary with fields 'nidq',
    'imec'. The 'nidq' field contains the path to the nidq file and its dclut json file. The
    'imec' field contains a list of dictionaries, each with fields 'ap', 'lfp', 'ap_dclut',
    and 'lfp_dclut'.

    Parameters
    ----------
    sess_dir : str
        The session directory
    use_catgt_version : bool
        Whether to use the CatGT version of the files. 
        Default is False

    Returns
    -------
    session_dict : dict
        A dictionary with fields 'nidq', 'imec'. The 'nidq' field contains the path to the nidq file 
        and its dclut json file. The 'imec' field contains a list of dictionaries, each with fields 
        'ap', 'lfp', 'ap_dclut', and 'lfp_dclut'.
        
    Notes
    -----
    This function now internally attempts to use the Session configuration system.
    If session_config.json exists, it will use that. Otherwise, it falls back to
    hardcoded patterns for backwards compatibility.

    """
    # Try to use Session configuration first
    try:
        sess = Session(sess_dir)
        print("Found session_config.json, using Session configuration")
        
        # Get files using the session configuration
        ap_files = [str(f) for f in sess.find_files('ap_pattern')]
        nidq_files = [str(f) for f in sess.find_files('nidq_pattern')]
        
        if not ap_files:
            raise FileNotFoundError(f"No AP files found matching pattern in {sess_dir}")
        if not nidq_files:
            raise FileNotFoundError(f"No NIDQ files found matching pattern in {sess_dir}")
            
        nidq_file = nidq_files[0]
        
    except FileNotFoundError as e:
        if "session_config.json" in str(e):
            print("No session_config.json found, falling back to hardcoded patterns")
            print("Consider creating a session_config.json file for better flexibility")
            
            # Fall back to legacy hardcoded patterns
            if use_catgt_version:
                ap_files = find_all_matching_files(sess_dir, r'tcat\.imec([0-9]+)\.ap\.bin')
            else:
                ap_files = find_all_matching_files(sess_dir, r't0\.imec([0-9]+)\.ap\.bin')
            nidq_file = find_all_matching_files(sess_dir, r't0\.nidq\.bin')[0] # there should be only one nidq file
        else:
            # Re-raise file not found errors for actual files
            raise e

    session_dict = {}

    session_dict['nidq'] = {
        'dir': os.path.dirname(nidq_file),
        'file': nidq_file,
        'dclut': nidq_file.replace('.bin', '_dclut.json')
    }

    imec_list = []
    for ap_file in ap_files:
        imec_dict = {}
        imec_dict['dir'] = os.path.dirname(ap_file)
        imec_dict['ap'] = ap_file
        imec_dict['ap_dclut'] = ap_file.replace('.bin', '_dclut.json')
        lfp_file = ap_file.replace('ap.bin', 'lfp.bin')
        imec_dict['lfp'] = lfp_file
        imec_dict['lfp_dclut'] = lfp_file.replace('.bin', '_dclut.json')
        imec_list.append(imec_dict)

    session_dict['imec'] = imec_list

    return session_dict
