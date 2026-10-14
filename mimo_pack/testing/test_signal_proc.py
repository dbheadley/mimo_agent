"""
Unit tests for mimo_pack.analysis.signal_proc module.

Tests the refactored signal processing functions to ensure they:
1. Accept only xarray.DataArray inputs
2. Preserve spatial coordinates and metadata
3. Use constants from mimo_pack.consts
"""

import numpy as np
import xarray as xr
import pytest
from mimo_pack.analysis.signal_proc import envelope
from mimo_pack.testing.data import synthetic_lfp
from mimo_pack.consts import DIM_TIME, DIM_CHAN, COORD_X, COORD_Y


class TestSignalProcessing:
    """Test suite for signal processing functions."""
    
    def test_envelope_basic(self):
        """Test basic envelope calculation."""
        lfp = synthetic_lfp(duration=1.0, fs=1000, n_chans=4, seed=42)
        
        lfp_env = envelope(lfp)
        
        # Check output type and shape
        assert isinstance(lfp_env, xr.DataArray)
        assert lfp_env.shape == lfp.shape
        
        # Check that envelope is non-negative
        assert np.all(lfp_env.values >= 0)
        
        # Envelope should generally bound the absolute signal
        # (within numerical tolerance)
        bound_violations = np.sum(np.abs(lfp.values) > lfp_env.values + 1e-10)
        violation_rate = bound_violations / lfp.values.size
        assert violation_rate < 0.01  # Less than 1% violations
    
    def test_envelope_input_validation(self):
        """Test that envelope rejects non-xarray inputs."""
        data = np.random.randn(1000, 4)
        
        with pytest.raises(TypeError, match="Input must be an xarray.DataArray"):
            envelope(data)
    
    def test_envelope_coordinates_preserved(self):
        """Test that envelope processing preserves coordinate information."""
        lfp = synthetic_lfp(duration=0.5, fs=1000, n_chans=4, seed=42)
        
        result = envelope(lfp)
        
        # Check time coordinate preserved
        np.testing.assert_array_equal(result[DIM_TIME], lfp[DIM_TIME])
        
        # Check spatial coordinates preserved (if they exist)
        for coord in [COORD_X, COORD_Y]:
            if coord in lfp.coords:
                np.testing.assert_array_equal(result[coord], lfp[coord])
    
    def test_envelope_metadata_preservation(self):
        """Test that envelope processing preserves metadata attributes."""
        lfp = synthetic_lfp(duration=0.5, fs=1000, n_chans=2, seed=42)
        lfp.attrs['experiment'] = 'test_session'
        lfp.attrs['custom_param'] = 42
        
        lfp_env = envelope(lfp)
        
        # Should preserve custom attributes
        assert lfp_env.attrs['experiment'] == 'test_session'
        assert lfp_env.attrs['custom_param'] == 42
        
        # Should preserve or add sampling rate info
        assert 'fs' in lfp_env.attrs
    
    def test_envelope_sinusoid(self):
        """Test envelope of pure sinusoid."""
        # Create a pure sinusoid
        fs = 1000
        t = np.linspace(0, 2, 2000)
        amplitude = 3.0
        freq = 10  # Hz
        
        signal = amplitude * np.sin(2 * np.pi * freq * t)
        signal_xr = xr.DataArray(
            signal,
            dims=[DIM_TIME],
            coords={DIM_TIME: t},
            attrs={'fs': fs}
        )
        
        env = envelope(signal_xr)
        
        # Envelope of pure sinusoid should be approximately constant
        # (ignoring edge effects)
        env_middle = env.values[200:-200]  # Avoid edges
        env_mean = np.mean(env_middle)
        env_std = np.std(env_middle)
        
        assert abs(env_mean - amplitude) < 0.1
        assert env_std < 0.1


if __name__ == '__main__':
    pytest.main([__file__])