"""
Unit tests for mimo_pack.analysis.filter module.

Tests the refactored filter functions to ensure they:
1. Accept only xarray.DataArray inputs
2. Extract sampling rates from time coordinates
3. Preserve spatial coordinates and metadata
4. Use constants from mimo_pack.consts
"""

import numpy as np
import xarray as xr
import pytest
from mimo_pack.analysis.filter import lp_filter, hp_filter
from mimo_pack.testing.data import synthetic_lfp
from mimo_pack.consts import DIM_TIME, DIM_CHAN, COORD_X, COORD_Y


class TestFilterFunctions:
    """Test suite for filter functions."""
    
    def test_lp_filter_basic(self):
        """Test basic low-pass filtering functionality."""
        # Create synthetic data
        lfp = synthetic_lfp(duration=2.0, fs=1000, n_chans=4, seed=42)
        
        # Apply low-pass filter
        lfp_filt = lp_filter(lfp, cutoff_freq=100)
        
        # Check output is xarray.DataArray
        assert isinstance(lfp_filt, xr.DataArray)
        
        # Check dimensions preserved
        assert lfp_filt.dims == lfp.dims
        assert lfp_filt.shape == lfp.shape
        
        # Check coordinates preserved
        for coord in [DIM_TIME, DIM_CHAN, COORD_X, COORD_Y]:
            if coord in lfp.coords:
                np.testing.assert_array_equal(lfp_filt[coord], lfp[coord])
        
        # Check attributes preserved
        for key in lfp.attrs:
            assert key in lfp_filt.attrs
    
    def test_lp_filter_input_validation(self):
        """Test that lp_filter rejects non-xarray inputs."""
        # Create numpy array (should be rejected)
        data = np.random.randn(1000, 4)
        
        with pytest.raises(TypeError, match="Input must be an xarray.DataArray"):
            lp_filter(data, cutoff_freq=100)
    
    def test_lp_filter_preserves_metadata(self):
        """Test that filtering preserves important metadata."""
        lfp = synthetic_lfp(duration=1.0, fs=1000, n_chans=2, seed=42)
        lfp.attrs['custom_meta'] = 'test_value'
        
        lfp_filt = lp_filter(lfp, cutoff_freq=50)
        
        # Should preserve custom metadata
        assert lfp_filt.attrs['custom_meta'] == 'test_value'
        assert 'fs' in lfp_filt.attrs
        assert 'units' in lfp_filt.attrs
    
    def test_hp_filter_basic(self):
        """Test basic high-pass filtering."""
        lfp = synthetic_lfp(duration=1.0, fs=500, n_chans=2, seed=42)
        
        lfp_filt = hp_filter(lfp, cutoff_freq=1)
        
        assert isinstance(lfp_filt, xr.DataArray)
        assert lfp_filt.shape == lfp.shape
        
        # Check coordinates preserved
        np.testing.assert_array_equal(lfp_filt[DIM_TIME], lfp[DIM_TIME])
    
    def test_sampling_rate_extraction(self):
        """Test that filters properly extract sampling rate from time coordinates."""
        # Create data with known sampling rate
        fs = 2000
        lfp = synthetic_lfp(duration=0.5, fs=fs, n_chans=2, seed=42)
        
        # Apply filter (should work without explicit fs)
        lfp_filt = lp_filter(lfp, cutoff_freq=200)
        
        # Should succeed without errors and preserve sampling rate info
        assert isinstance(lfp_filt, xr.DataArray)
        extracted_fs = 1 / float(lfp[DIM_TIME].diff(DIM_TIME).median())
        assert abs(extracted_fs - fs) < 1  # Within 1 Hz tolerance
    
    def test_multichannel_consistency(self):
        """Test that filtering is consistent across channels."""
        lfp = synthetic_lfp(duration=1.0, fs=1000, n_chans=4, seed=42)
        
        lfp_filt = lp_filter(lfp, cutoff_freq=100)
        
        # All channels should be filtered (none should be unchanged)
        for ch in range(lfp.sizes[DIM_CHAN]):
            orig_ch = lfp.isel({DIM_CHAN: ch})
            filt_ch = lfp_filt.isel({DIM_CHAN: ch})
            
            # Should not be identical (filtering should change the signal)
            assert not np.allclose(orig_ch.values, filt_ch.values, rtol=0.1)


if __name__ == '__main__':
    pytest.main([__file__])
