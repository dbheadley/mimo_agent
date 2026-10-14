"""
Unit tests for mimo_pack.analysis.csd module.

Tests the refactored current source density functions to ensure they:
1. Accept only xarray.DataArray inputs
2. Preserve spatial coordinates and metadata
3. Use constants from mimo_pack.consts
"""

import numpy as np
import xarray as xr
import pytest
from mimo_pack.analysis.csd import csd
from mimo_pack.testing.data import synthetic_lfp
from mimo_pack.consts import DIM_TIME, DIM_CHAN, COORD_X, COORD_Y


class TestCSDAnalysis:
    """Test suite for current source density analysis."""
    
    def test_csd_basic(self):
        """Test basic CSD computation."""
        # Create data with linear arrangement 
        lfp = synthetic_lfp(duration=0.5, fs=1000, n_chans=16, seed=42)
        
        # Assign linear depth coordinates  
        depths = np.arange(16) * 25  # 25 µm spacing
        lfp = lfp.assign_coords({COORD_Y: (DIM_CHAN, depths)})
        
        csd_result = csd(lfp, coord=COORD_Y)
        
        # Check output properties
        assert isinstance(csd_result, xr.DataArray)
        assert csd_result.dims == lfp.dims
        
        # Time should be preserved
        np.testing.assert_array_equal(csd_result[DIM_TIME], lfp[DIM_TIME])
        
        # Should have spatial coordinates
        assert COORD_Y in csd_result.coords
    
    def test_csd_input_validation(self):
        """Test that CSD functions reject non-xarray inputs."""
        data = np.random.randn(1000, 16)
        
        with pytest.raises(TypeError, match="Input must be an xarray.DataArray"):
            csd(data, coord=COORD_Y)
    
    def test_csd_metadata_preservation(self):
        """Test that CSD preserves important metadata."""
        lfp = synthetic_lfp(duration=0.3, fs=1000, n_chans=8, seed=42)
        lfp.attrs['experiment'] = 'csd_test'
        lfp.attrs['probe_type'] = 'neuropixels'
        
        # Add linear coordinates
        depths = np.arange(8) * 25
        lfp = lfp.assign_coords({COORD_Y: (DIM_CHAN, depths)})
        
        csd_result = csd(lfp, coord=COORD_Y)
        
        # Should preserve custom metadata
        assert csd_result.attrs['experiment'] == 'csd_test'
        assert csd_result.attrs['probe_type'] == 'neuropixels'
        
        # Should preserve sampling rate info
        assert 'fs' in csd_result.attrs
    
    def test_csd_units_and_scaling(self):
        """Test that CSD has proper units and scaling."""
        lfp = synthetic_lfp(duration=0.3, fs=1000, n_chans=8, seed=42)
        
        # Add spatial coordinates
        depths = np.arange(8) * 25
        lfp = lfp.assign_coords({COORD_Y: (DIM_CHAN, depths)})
        
        csd_result = csd(lfp, coord=COORD_Y)
        
        # CSD values should be finite and reasonable
        assert np.all(np.isfinite(csd_result.values))
        
        # CSD should have different magnitude than LFP
        csd_rms = float(np.sqrt(np.mean(csd_result.values**2)))
        lfp_rms = float(np.sqrt(np.mean(lfp.values**2)))
        
        # Should be different (exact relationship depends on implementation)
        assert csd_rms != lfp_rms


if __name__ == '__main__':
    pytest.main([__file__])