# Utility functions for working with SpikeGLX files
# Author: Drew Headley
# Date: 2024-05-19

import os
import numpy as np
import pandas as pd
import re
from dclut import create_dclut

def read_meta(bin_path):
    """
    Read the .meta file associated with a SpikeGLX .bin file
    
    Parameters
    ----------
    meta_path : str
        Path to the .meta file
        
    Returns
    -------
    meta : dict
        Dictionary containing the metadata
    """

    check_bin_path(bin_path)

    meta_path = get_meta_path(bin_path)
    meta = {}
    with open(meta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue

            match = re.search(r"(.*?)=(.*)", line)   #line.split('=')
            key = match.group(1).strip()
            value = match.group(2).strip()

            if not key.startswith('imDat'): # leave version numbers as strings
                # Format the values
                if re.match(r'^\d+$', value): # format an integer
                    value = int(value)
                elif re.match(r'^\d+\.\d+$', value): # format a decimal number
                    value = float(value)
                elif re.match(r'^-?\d+(?:,-?\d+)*$', value): # format a list of integers
                    value = [int(v) for v in value.split(',')]
                elif value.lower() == 'true': # format a boolean
                    value = True
                elif value.lower() == 'false': # format a boolean
                    value = False

            meta[key] = value
    
    return meta



def get_meta_path(bin_path):
    """
    Get the path to the .meta file from a SpikeGLX binary file.
    Tests if the .meta file exists in the same directory as the binary file.
    
    Parameters
    ----------
    bin_path : str
        Path to the binary file
        
    Returns
    -------
    meta_path : str
        Path to the .meta file
    """

    # validate binary file path
    check_bin_path(bin_path)
    
    # remove the .bin extension, add .meta extension
    meta_path = bin_path[:-4] + '.meta'

    # check if meta file exists
    if not os.path.exists(meta_path):
        raise FileNotFoundError('Meta file {} not found'.format(meta_path))
    
    return meta_path

def get_bin_memmap(bin_path):
    """
    Get a memory map of the binary file.
    
    Parameters
    ----------
    bin_path : str
        Path to the binary file
    
    Returns
    -------
    bin_mm : np.memmap
        Memory map of the binary file
    """

    # validate binary file path
    check_bin_path(bin_path)
    
    bin_size = os.path.getsize(bin_path)
    meta = read_meta(bin_path)
    
    fs = meta['imSampRate']
    chan_num = meta['nSavedChans']
    samp_num = bin_size // (2 * chan_num)

    # open binary file, read only
    bin_mm = np.memmap(bin_path, dtype='int16', mode='r', shape=(samp_num, chan_num))

    return bin_mm


def check_bin_path(bin_path):
    """
    Check if the binary file exists and has a .bin extension.
    
    Parameters
    ----------
    bin_path : str
        Path to the binary file

    """

    # check if bin path is a .bin file
    if not bin_path.endswith('.bin'):
        raise ValueError('Binary file must have .bin extension')

    # check if the binary file exists
    if not os.path.exists(bin_path):
        raise FileNotFoundError('Binary file {} not found'.format(bin_path))
    

def get_geommap(bin_path):
    """
    Gets the geometry map of each channel on the probe from the .bin file

    Parameters
    ----------
    bin_path : str
        Path to the binary file

    Returns
    -------
    g_df : pd.DataFrame
        Geometry map of the recorded electrodes. Each row is a channel.
        The columns are: shank, x, y, used (u-flag)
    """
    

    meta = read_meta(bin_path)
    sgm = meta['~snsGeomMap']

    prb_params = probe_params(meta['imDatPrb_pn'])

    # get the entry for each channel
    sgm_split = re.findall(r'\((.*?)\)', sgm)

    # pull out the 
    gmap = np.zeros((len(sgm_split)-1, 4))
    for i, s in enumerate(sgm_split[1:]):
        gmap[i] = [float(x) for x in s.split(':')]
    
    gmap[:, 1] += gmap[:, 0] * prb_params['shank_spacing']

    g_df = pd.DataFrame(gmap, columns=['shank', 'x', 'y', 'used'])
    return g_df

def get_chanmap(bin_path):
    """
    Get the channel map of the binary file.
    
    Parameters
    ----------
    bin_path : str
        Path to the binary file
        
    Returns
    -------
    c_df : pd.DataFrame
        Channel map of the binary file. Each row is a channel.
        The columns are: name, order
    """

    meta = read_meta(bin_path)
    chanmap = meta['~snsChanMap']

    scm_split = re.findall(r'\((.*?)\)', chanmap)
    ch_names = []
    ch_order = []
    for s in scm_split[1:]:
        ch_names.append(s.split(':')[0])
        ch_order.append(int(s.split(':')[1]))
    
    c_df = pd.DataFrame({'name': ch_names, 'order': ch_order})
    return c_df


def probe_params(probe_pn):
    """
    Returns the probe parameters for a given probe part number
    
    Parameters
    ----------
    probe_pn : str
        Probe part number, derived from imDatPrb field in .meta file
    
    Returns
    -------
    params : dict
        Dictionary of probe parameters
    """
    
    params = {}
    if probe_pn == 'NP2014':
        params['shank_num'] = 4
        params['shank_spacing'] = 250
    else:
        raise ValueError('Probe part number not recognized')
    
    return params

def int2uv(bin_path):
    """
    Get the conversion factor from binary data to uV.
    
    Parameters
    ----------
    bin_path : str
        Path to the binary file
        
    Returns
    -------
    conv : float
        Conversion factor from binary data to uV
    """

    meta = read_meta(bin_path)

    if meta['typeThis'] == 'imec':
        if 'imMaxInt' in meta:
            max_int = meta['imMaxInt']
        else:
            max_int = 512
        aiRange = meta['imAiRangeMax']
        gain = meta['imChan0apGain']
    elif meta['typeThis'] == 'nidq':
        max_int = meta['niMaxInt']
        aiRange = meta['niAiRangeMax']
        gain = 1
    else:
        max_int = 1
        aiRange = 1
        gain = 1
    
    conv = ((float(aiRange) / max_int) / gain) * 1e6
    return conv

def dclut_from_meta(bin_path, dcl_path=None):
    """
    Create a dclut json file from the .meta file associated with a SpikeGLX .bin file.

    For NIDQ files a per-channel ``gain`` scale (unit ``V/count``) is appended
    so that :func:`mimo_pack.fileio.nidq.load_nidq` can automatically convert
    raw int16 counts to volts on load.  Gain values are derived from the .meta
    fields ``niMaxInt``, ``niAiRangeMax``, ``niMNGain``, and ``niMAGain``:

    * **XA** channels: ``niAiRangeMax / niMaxInt``
    * **MN** channels: ``niAiRangeMax / (niMaxInt \u00d7 niMNGain)``
    * **MA** channels: ``niAiRangeMax / (niMaxInt \u00d7 niMAGain)``
    * **XD / SY** channels: ``1.0`` (digital / unitless)

    Parameters
    ----------
    bin_path : str
        Path to the binary file

    Optional
    --------
    dcl_path : str
        Path to save the dclut json file. If not provided, the file will be
        saved in the same directory as the binary file with a .dclut extension.

    Returns
    -------
    dcl_path : str
        Path to the dclut json file
    """
    
    meta = read_meta(bin_path)
    chmap = get_chanmap(bin_path)
    
    chan_num = meta['nSavedChans']
    t_num = meta['fileSizeBytes'] // (2 * chan_num)

    if meta['typeThis'] == 'imec':
        gmap = get_geommap(bin_path)
        chan_props = chmap.merge(gmap, left_index=True, right_index=True, how='outer')
        scales = [{'name': 'time', 'dim': 0, 'unit': 'seconds', 
                   'type': 'linear', 'val': [1/meta['imSampRate'], 0]}, 
                  {'name': 'channel', 'dim': 1, 'unit': 'none', 
                   'type': 'index', 'val': None}, 
                  {'name': 'ch_name', 'dim': 1, 'unit': 'none', 
                   'type': 'list', 'val': chan_props['name'].values}, 
                  {'name': 'ch_order', 'dim': 1, 'unit': 'none', 
                   'type': 'list', 'val': chan_props['order'].values}, 
                  {'name': 'ch_x', 'dim': 1, 'unit': 'um', 
                   'type': 'list', 'val': chan_props['x'].values}, 
                  {'name': 'ch_y', 'dim': 1, 'unit': 'um', 
                   'type': 'list', 'val': chan_props['y'].values}, 
                  {'name': 'ch_shank', 'dim': 1, 'unit': 'none', 
                   'type': 'list', 'val': chan_props['shank'].values}]
    elif meta['typeThis'] == 'nidq':
        # Compute per-channel voltage gain (V / count)
        # Formula: niAiRangeMax / (niMaxInt * channel_gain)
        # MN channels have a programmable gain (niMNGain); MA have niMAGain;
        # XA channels have no extra gain; XD/SY are digital (gain = 1.0)
        ni_max_int   = float(meta.get('niMaxInt', 32768))
        ni_ai_range  = float(meta.get('niAiRangeMax', 5.0))
        mn_gain      = float(meta.get('niMNGain', 1.0))
        ma_gain      = float(meta.get('niMAGain', 1.0))

        _PREFIX_RE = re.compile(r'^([A-Za-z]+)')
        def _nidq_ch_gain(name: str) -> float:
            m = _PREFIX_RE.match(name.split(';')[0])
            prefix = m.group(1).upper() if m else ''
            if prefix == 'MN':
                return ni_ai_range / (ni_max_int * mn_gain)
            elif prefix == 'MA':
                return ni_ai_range / (ni_max_int * ma_gain)
            elif prefix == 'XA':
                return ni_ai_range / ni_max_int
            else:          # XD, SY, unknown — digital / unitless
                return 1.0

        gain_vals = [_nidq_ch_gain(n) for n in chmap['name'].values]

        scales = [{'name': 'time', 'dim': 0, 'unit': 'seconds',
                   'type': 'linear', 'val': [1/meta['niSampRate'], 0]},
                  {'name': 'channel', 'dim': 1, 'unit': 'none',
                   'type': 'index', 'val': None},
                  {'name': 'ch_name', 'dim': 1, 'unit': 'none',
                   'type': 'list', 'val': chmap['name'].values},
                  {'name': 'gain', 'dim': 1, 'unit': 'V/count',
                   'type': 'list', 'val': np.array(gain_vals)}]

    dcl_path = create_dclut(bin_path, [t_num, chan_num], dcl_path=dcl_path,
                            dtype='int16', data_name='data', data_unit='au',
                            scales=scales)
    return dcl_path
