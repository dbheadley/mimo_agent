"""Load Intan RHD recording system data (one-file-per-signal-type format).

This module loads data from Intan's "One File Per Signal Type" recording format,
where each signal type is saved as a separate ``.dat`` binary file alongside an
``info.rhd`` header file containing metadata.

Supported files:
    - ``info.rhd`` — binary header (channel info, sample rate, filter settings)
    - ``amplifier.dat`` — wideband neural amplifier data (int16, µV)
    - ``digitalin.dat`` — digital input lines (uint16, one bit per channel)
    - ``time.dat`` — sample timestamps (int32, converted to seconds)

Reference: https://intantech.com/files/Intan_RHD2000_data_file_formats.pdf

Author: Drew B. Headley
"""

import os
import struct

import numpy as np
import xarray as xr

from mimo_pack.consts import ATTR_FS, ATTR_UNITS, DIM_TIME, DIM_CHAN


# ---------------------------------------------------------------------------
# Header parsing (adapted from Intan Technologies reference code, GPL-3.0)
# ---------------------------------------------------------------------------

def _read_qstring(fid):
    """Read a Qt-style QString from an open binary file.

    The first 4 bytes are an unsigned 32-bit integer giving the string length
    in bytes.  A value of ``0xFFFFFFFF`` indicates a null string.
    """
    (length,) = struct.unpack('<I', fid.read(4))
    if length == 0xFFFFFFFF:
        return ""
    length = int(length / 2)  # convert bytes to 16-bit Unicode chars
    data = []
    for _ in range(length):
        (c,) = struct.unpack('<H', fid.read(2))
        data.append(c)
    return ''.join(chr(c) for c in data)


def read_rhd_header(rhd_path: str) -> dict:
    """Parse an Intan RHD header file and return metadata as a dictionary.

    Works with both standalone ``info.rhd`` files (one-file-per-signal-type
    format) and self-contained ``.rhd`` data files.

    Parameters
    ----------
    rhd_path : str
        Path to the ``info.rhd`` (or ``.rhd``) file.

    Returns
    -------
    header : dict
        Dictionary containing at minimum:

        - ``'sample_rate'`` (float): amplifier sampling rate in Hz.
        - ``'num_amplifier_channels'`` (int): number of enabled amplifier
          channels.
        - ``'amplifier_channels'`` (list[dict]): per-channel metadata dicts
          with keys ``'native_channel_name'``, ``'custom_channel_name'``,
          ``'native_order'``, ``'chip_channel'``, ``'board_stream'``,
          ``'electrode_impedance_magnitude'``,
          ``'electrode_impedance_phase'``, ``'port_name'``,
          ``'port_prefix'``, ``'port_number'``.
        - ``'num_board_dig_in_channels'`` (int): number of enabled digital
          input channels.
        - ``'board_dig_in_channels'`` (list[dict]): per-channel metadata.
        - ``'version'`` (dict): ``{'major': int, 'minor': int}``.
        - ``'notch_filter_frequency'`` (int): 0, 50, or 60 Hz.
        - ``'frequency_parameters'`` (dict): filter bandwidth settings.

    Raises
    ------
    FileNotFoundError
        If *rhd_path* does not exist.
    ValueError
        If the file does not begin with the Intan RHD magic number.
    """
    if not os.path.isfile(rhd_path):
        raise FileNotFoundError(
            f"RHD header file not found: {rhd_path}. "
            "Ensure the Intan recording directory contains an 'info.rhd' file."
        )

    header = {}
    with open(rhd_path, 'rb') as fid:
        # -- magic number --
        (magic,) = struct.unpack('<I', fid.read(4))
        if magic != 0xC6912702:
            raise ValueError(
                f"File does not appear to be an Intan RHD header "
                f"(magic=0x{magic:08X}). "
                "Expected magic number 0xC6912702."
            )

        # -- version --
        version = {}
        version['major'], version['minor'] = struct.unpack('<hh', fid.read(4))
        header['version'] = version

        # samples per data block (60 for <v2.0, 128 for >=v2.0)
        header['num_samples_per_data_block'] = (
            128 if version['major'] > 1 else 60
        )

        # -- sample rate --
        (header['sample_rate'],) = struct.unpack('<f', fid.read(4))

        # -- frequency settings --
        freq = {}
        (freq['dsp_enabled'],
         freq['actual_dsp_cutoff_frequency'],
         freq['actual_lower_bandwidth'],
         freq['actual_upper_bandwidth'],
         freq['desired_dsp_cutoff_frequency'],
         freq['desired_lower_bandwidth'],
         freq['desired_upper_bandwidth']) = struct.unpack('<hffffff',
                                                         fid.read(26))

        # -- notch filter --
        (notch_mode,) = struct.unpack('<h', fid.read(2))
        header['notch_filter_frequency'] = {1: 50, 2: 60}.get(notch_mode, 0)
        freq['notch_filter_frequency'] = header['notch_filter_frequency']

        # -- impedance test frequencies --
        (freq['desired_impedance_test_frequency'],
         freq['actual_impedance_test_frequency']) = struct.unpack(
            '<ff', fid.read(8)
        )

        # -- notes --
        header['notes'] = {
            'note1': _read_qstring(fid),
            'note2': _read_qstring(fid),
            'note3': _read_qstring(fid),
        }

        # -- temp sensor channels (v1.1+) --
        header['num_temp_sensor_channels'] = 0
        if ((version['major'] == 1 and version['minor'] >= 1)
                or version['major'] > 1):
            (header['num_temp_sensor_channels'],) = struct.unpack(
                '<h', fid.read(2)
            )

        # -- eval board mode (v1.3+) --
        header['eval_board_mode'] = 0
        if ((version['major'] == 1 and version['minor'] >= 3)
                or version['major'] > 1):
            (header['eval_board_mode'],) = struct.unpack('<h', fid.read(2))

        # -- reference channel (v2.0+) --
        if version['major'] > 1:
            header['reference_channel'] = _read_qstring(fid)

        # -- frequency parameters --
        freq['amplifier_sample_rate'] = header['sample_rate']
        freq['aux_input_sample_rate'] = header['sample_rate'] / 4
        freq['supply_voltage_sample_rate'] = (
            header['sample_rate'] / header['num_samples_per_data_block']
        )
        freq['board_adc_sample_rate'] = header['sample_rate']
        freq['board_dig_in_sample_rate'] = header['sample_rate']
        header['frequency_parameters'] = freq

        # -- initialise channel lists --
        header['amplifier_channels'] = []
        header['aux_input_channels'] = []
        header['supply_voltage_channels'] = []
        header['board_adc_channels'] = []
        header['board_dig_in_channels'] = []
        header['board_dig_out_channels'] = []
        header['spike_triggers'] = []

        # -- signal groups --
        (n_signal_groups,) = struct.unpack('<h', fid.read(2))
        for _ in range(n_signal_groups):
            _sg_name = _read_qstring(fid)
            _sg_prefix = _read_qstring(fid)
            sg_enabled, sg_n_chans, _ = struct.unpack('<hhh', fid.read(6))

            if sg_n_chans > 0 and sg_enabled > 0:
                for _ in range(sg_n_chans):
                    ch = {
                        'port_name': _sg_name,
                        'port_prefix': _sg_prefix,
                    }
                    ch['native_channel_name'] = _read_qstring(fid)
                    ch['custom_channel_name'] = _read_qstring(fid)
                    (ch['native_order'],
                     ch['custom_order'],
                     signal_type,
                     channel_enabled,
                     ch['chip_channel'],
                     ch['board_stream']) = struct.unpack(
                        '<hhhhhh', fid.read(12)
                    )

                    # spike trigger settings
                    trigger = {}
                    (trigger['voltage_trigger_mode'],
                     trigger['voltage_threshold'],
                     trigger['digital_trigger_channel'],
                     trigger['digital_edge_polarity']) = struct.unpack(
                        '<hhhh', fid.read(8)
                    )

                    # impedance
                    (ch['electrode_impedance_magnitude'],
                     ch['electrode_impedance_phase']) = struct.unpack(
                        '<ff', fid.read(8)
                    )

                    if not channel_enabled:
                        continue

                    if signal_type == 0:
                        header['amplifier_channels'].append(ch)
                        header['spike_triggers'].append(trigger)
                    elif signal_type == 1:
                        header['aux_input_channels'].append(ch)
                    elif signal_type == 2:
                        header['supply_voltage_channels'].append(ch)
                    elif signal_type == 3:
                        header['board_adc_channels'].append(ch)
                    elif signal_type == 4:
                        header['board_dig_in_channels'].append(ch)
                    elif signal_type == 5:
                        header['board_dig_out_channels'].append(ch)

        # -- channel counts --
        header['num_amplifier_channels'] = len(header['amplifier_channels'])
        header['num_aux_input_channels'] = len(header['aux_input_channels'])
        header['num_supply_voltage_channels'] = len(
            header['supply_voltage_channels']
        )
        header['num_board_adc_channels'] = len(header['board_adc_channels'])
        header['num_board_dig_in_channels'] = len(
            header['board_dig_in_channels']
        )
        header['num_board_dig_out_channels'] = len(
            header['board_dig_out_channels']
        )

    return header


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_intan_amplifier(intan_dir: str,
                         channels: list = None) -> xr.DataArray:
    """Load amplifier data from an Intan one-file-per-signal-type recording.

    Reads ``info.rhd`` for metadata and ``amplifier.dat`` + ``time.dat`` for
    data, returning a fully-attributed ``xr.DataArray``.

    Parameters
    ----------
    intan_dir : str
        Path to the Intan recording directory containing ``info.rhd``,
        ``amplifier.dat``, and ``time.dat``.
    channels : list of int, optional
        Subset of channel indices (0-based) to load.  If ``None``, all
        channels are loaded.

    Returns
    -------
    xr.DataArray
        Amplifier data with dimensions ``(time, channel)``.

        Coordinates:
            - ``time`` — sample times in seconds.
            - ``channel`` — channel indices.
            - ``ch_names`` — per-channel custom name strings
              (along ``channel``).
            - ``ch_native_names`` — per-channel native name strings
              (along ``channel``).

        Attributes:
            - ``fs`` (float): sampling rate in Hz.
            - ``units`` (str): ``'µV'``.
            - ``description`` (str): provenance string.
            - ``notch_filter_frequency`` (int): 0, 50, or 60.
            - ``intan_version`` (str): e.g. ``'3.0'``.

    Raises
    ------
    FileNotFoundError
        If ``info.rhd``, ``amplifier.dat``, or ``time.dat`` is missing.
    TypeError
        If *intan_dir* is not a string.
    ValueError
        If *channels* contains invalid indices.

    Examples
    --------
    >>> amp = load_intan_amplifier('/data/mouse1/intan_session/')
    >>> amp.dims
    ('time', 'channel')
    >>> float(amp.attrs['fs'])
    30000.0
    """
    if not isinstance(intan_dir, str):
        raise TypeError(
            f"Expected str for intan_dir, got {type(intan_dir).__name__}. "
            "Provide the path to the Intan recording directory."
        )

    # -- Locate required files --
    rhd_path = os.path.join(intan_dir, 'info.rhd')
    amp_path = os.path.join(intan_dir, 'amplifier.dat')
    time_path = os.path.join(intan_dir, 'time.dat')

    for fpath, label in [(rhd_path, 'info.rhd'),
                         (amp_path, 'amplifier.dat'),
                         (time_path, 'time.dat')]:
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"Required file '{label}' not found in {intan_dir}."
            )

    # -- Read header --
    header = read_rhd_header(rhd_path)
    fs = float(header['sample_rate'])
    n_chans = header['num_amplifier_channels']

    if n_chans == 0:
        raise ValueError(
            "No amplifier channels found in info.rhd. "
            "Check that amplifier channels were enabled during recording."
        )

    # -- Read timestamps --
    # time.dat: int32 per sample, divide by sample rate to get seconds
    t_raw = np.fromfile(time_path, dtype=np.int32)
    n_samples = t_raw.size
    time_s = t_raw / fs

    # -- Read amplifier data --
    # amplifier.dat: int16, interleaved across channels.
    # Layout: [s0_ch0, s0_ch1, ..., s0_chN-1, s1_ch0, s1_ch1, ...]
    raw = np.fromfile(amp_path, dtype=np.int16)
    expected_size = n_samples * n_chans
    if raw.size != expected_size:
        raise ValueError(
            f"amplifier.dat size mismatch: expected {expected_size} samples "
            f"({n_samples} timepoints x {n_chans} channels), got {raw.size}. "
            "The file may be truncated or the channel count in info.rhd "
            "may be wrong."
        )

    # Reshape to (n_samples, n_channels) and convert to microvolts.
    # Intan amplifier conversion factor: 0.195 µV per ADC step.
    data = raw.reshape(n_samples, n_chans).astype(np.float32)
    data *= 0.195

    # -- Build channel metadata --
    ch_ids = np.arange(n_chans)
    ch_custom_names = [
        ch['custom_channel_name'] for ch in header['amplifier_channels']
    ]
    ch_native_names = [
        ch['native_channel_name'] for ch in header['amplifier_channels']
    ]

    # -- Optional channel subset --
    if channels is not None:
        channels = np.asarray(channels)
        if np.any(channels < 0) or np.any(channels >= n_chans):
            raise ValueError(
                f"Channel indices must be in [0, {n_chans - 1}]. "
                f"Got: {channels}."
            )
        data = data[:, channels]
        ch_ids = channels
        ch_custom_names = [ch_custom_names[i] for i in channels]
        ch_native_names = [ch_native_names[i] for i in channels]

    # -- Construct DataArray --
    amp_xr = xr.DataArray(
        data,
        dims=(DIM_TIME, DIM_CHAN),
        coords={
            DIM_TIME: time_s,
            DIM_CHAN: ch_ids,
            'ch_names': (DIM_CHAN, ch_custom_names),
            'ch_native_names': (DIM_CHAN, ch_native_names),
        },
    )

    amp_xr.attrs[ATTR_FS] = fs
    amp_xr.attrs[ATTR_UNITS] = 'µV'
    amp_xr.attrs['description'] = f"Intan amplifier data: {intan_dir}"
    amp_xr.attrs['notch_filter_frequency'] = header['notch_filter_frequency']
    amp_xr.attrs['intan_version'] = (
        f"{header['version']['major']}.{header['version']['minor']}"
    )
    amp_xr[DIM_TIME].attrs['units'] = 's'

    return amp_xr


def load_intan_digitalin(intan_dir: str,
                         channels: list = None) -> xr.DataArray:
    """Load digital input data from an Intan one-file-per-signal-type recording.

    Reads ``info.rhd`` for metadata and ``digitalin.dat`` + ``time.dat`` for
    data.  Each bit of the 16-bit samples corresponds to a separate digital
    input channel; these are unpacked into boolean rows.

    Parameters
    ----------
    intan_dir : str
        Path to the Intan recording directory containing ``info.rhd``,
        ``digitalin.dat``, and ``time.dat``.
    channels : list of int, optional
        Subset of digital input channel indices (0-based, matching the
        ``native_order`` field in ``info.rhd``) to load.  If ``None``, all
        enabled digital input channels are loaded.

    Returns
    -------
    xr.DataArray
        Digital input data with dimensions ``(time, channel)``.
        Values are boolean (``True`` = high, ``False`` = low).

        Coordinates:
            - ``time`` — sample times in seconds.
            - ``channel`` — digital input channel indices.
            - ``ch_names`` — per-channel custom name strings
              (along ``channel``).

        Attributes:
            - ``fs`` (float): sampling rate in Hz.
            - ``units`` (str): ``'bool'``.
            - ``description`` (str): provenance string.

    Raises
    ------
    FileNotFoundError
        If ``info.rhd``, ``digitalin.dat``, or ``time.dat`` is missing.
    TypeError
        If *intan_dir* is not a string.
    ValueError
        If no digital input channels are enabled, or *channels* is invalid.

    Examples
    --------
    >>> din = load_intan_digitalin('/data/mouse1/intan_session/')
    >>> din.dims
    ('time', 'channel')
    >>> din.dtype
    dtype('bool')
    """
    if not isinstance(intan_dir, str):
        raise TypeError(
            f"Expected str for intan_dir, got {type(intan_dir).__name__}. "
            "Provide the path to the Intan recording directory."
        )

    # -- Locate required files --
    rhd_path = os.path.join(intan_dir, 'info.rhd')
    din_path = os.path.join(intan_dir, 'digitalin.dat')
    time_path = os.path.join(intan_dir, 'time.dat')

    for fpath, label in [(rhd_path, 'info.rhd'),
                         (din_path, 'digitalin.dat'),
                         (time_path, 'time.dat')]:
        if not os.path.isfile(fpath):
            raise FileNotFoundError(
                f"Required file '{label}' not found in {intan_dir}."
            )

    # -- Read header --
    header = read_rhd_header(rhd_path)
    fs = float(header['sample_rate'])
    n_dig_chans = header['num_board_dig_in_channels']

    if n_dig_chans == 0:
        raise ValueError(
            "No digital input channels found in info.rhd. "
            "Check that digital input channels were enabled during recording."
        )

    # -- Read timestamps --
    t_raw = np.fromfile(time_path, dtype=np.int32)
    n_samples = t_raw.size
    time_s = t_raw / fs

    # -- Read digital input data --
    # digitalin.dat: uint16 per sample.  Each bit corresponds to one
    # digital input channel.
    raw = np.fromfile(din_path, dtype=np.uint16)
    if raw.size != n_samples:
        raise ValueError(
            f"digitalin.dat size mismatch: expected {n_samples} samples "
            f"(matching time.dat), got {raw.size}. "
            "The file may be truncated."
        )

    # -- Determine which channels to extract --
    dig_channels = header['board_dig_in_channels']

    if channels is not None:
        valid_orders = {ch['native_order'] for ch in dig_channels}
        channels_arr = np.asarray(channels)
        invalid = set(channels_arr.tolist()) - valid_orders
        if invalid:
            raise ValueError(
                f"Digital input channel indices {invalid} not found in "
                f"info.rhd. Available channels: {sorted(valid_orders)}."
            )
        dig_channels = [
            ch for ch in dig_channels
            if ch['native_order'] in set(channels_arr.tolist())
        ]

    # -- Extract individual channels from bitmask --
    n_out_chans = len(dig_channels)
    data = np.zeros((n_samples, n_out_chans), dtype=np.bool_)
    ch_ids = np.zeros(n_out_chans, dtype=int)
    ch_names = []

    for i, ch in enumerate(dig_channels):
        bit = ch['native_order']
        data[:, i] = (raw & (1 << bit)) != 0
        ch_ids[i] = bit
        ch_names.append(ch['custom_channel_name'])

    # -- Construct DataArray --
    din_xr = xr.DataArray(
        data,
        dims=(DIM_TIME, DIM_CHAN),
        coords={
            DIM_TIME: time_s,
            DIM_CHAN: ch_ids,
            'ch_names': (DIM_CHAN, ch_names),
        },
    )

    din_xr.attrs[ATTR_FS] = fs
    din_xr.attrs[ATTR_UNITS] = 'bool'
    din_xr.attrs['description'] = f"Intan digital input data: {intan_dir}"
    din_xr[DIM_TIME].attrs['units'] = 's'

    return din_xr


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m mimo_pack.fileio.intan <intan_directory>")
        sys.exit(1)

    test_dir = sys.argv[1]

    print("--- Loading amplifier data ---")
    amp = load_intan_amplifier(test_dir)
    print(amp)
    print(f"  fs = {amp.attrs[ATTR_FS]} Hz")
    print(f"  shape = {amp.shape}")
    print(f"  units = {amp.attrs[ATTR_UNITS]}")
    print()

    try:
        print("--- Loading digital input data ---")
        din = load_intan_digitalin(test_dir)
        print(din)
        print(f"  fs = {din.attrs[ATTR_FS]} Hz")
        print(f"  shape = {din.shape}")
    except (FileNotFoundError, ValueError) as e:
        print(f"  Skipped: {e}")
