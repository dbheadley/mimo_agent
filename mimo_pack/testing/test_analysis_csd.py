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
    
    def test_csd_2d_basic(self):
        """Test basic 2D CSD computation."""
        # Create data with proper spatial arrangement
        lfp = synthetic_lfp(duration=0.5, fs=1000, n_chans=16, seed=42)
        
        # Arrange channels in 4x4 grid
        x_coords = np.tile(np.arange(4), 4) * 25  # 25 µm spacing
        y_coords = np.repeat(np.arange(4), 4) * 25
        
        lfp = lfp.assign_coords({
            COORD_X: (DIM_CHAN, x_coords),
            COORD_Y: (DIM_CHAN, y_coords)
        })
        
        csd = csd_2d(lfp, spacing=25)
        
        # Check output properties
        assert isinstance(csd, xr.DataArray)
        assert csd.dims == lfp.dims
        
        # Time should be preserved
        np.testing.assert_array_equal(csd[DIM_TIME], lfp[DIM_TIME])
        
        # Should have spatial coordinates
        assert COORD_X in csd.coords
        assert COORD_Y in csd.coords
    
    def test_csd_2d_input_validation(self):
        """Test that CSD functions reject non-xarray inputs."""
        data = np.random.randn(1000, 16)
        
        with pytest.raises(TypeError, match="Input must be xarray.DataArray"):
            csd_2d(data, spacing=25)
    
    def test_csd_1d_basic(self):
        """Test basic 1D CSD computation (linear probe)."""
        # Create data for linear probe
        lfp = synthetic_lfp(duration=0.5, fs=1000, n_chans=32, seed=42)
        
        # Assign linear depth coordinates
        depths = np.arange(32) * 25  # 25 µm spacing
        lfp = lfp.assign_coords({COORD_Y: (DIM_CHAN, depths)})
        
        csd = csd_1d(lfp, spacing=25)
        
        # Check output properties
        assert isinstance(csd, xr.DataArray)
        
        # Time should be preserved
        np.testing.assert_array_equal(csd[DIM_TIME], lfp[DIM_TIME])
        
        # Should preserve depth coordinates
        assert COORD_Y in csd.coords
    
    def test_csd_units_and_scaling(self):
        """Test that CSD has proper units and scaling."""
        lfp = synthetic_lfp(duration=0.5, fs=1000, n_chans=16, seed=42)
        
        # Add spatial coordinates
        x_coords = np.tile(np.arange(4), 4) * 25
        y_coords = np.repeat(np.arange(4), 4) * 25
        lfp = lfp.assign_coords({
            COORD_X: (DIM_CHAN, x_coords),
            COORD_Y: (DIM_CHAN, y_coords)
        })
        
        csd = csd_2d(lfp, spacing=25)
        
        # CSD units should be updated
        if 'units' in lfp.attrs:
            # Should have different units than LFP
            assert csd.attrs.get('units') != lfp.attrs.get('units')
        
        # CSD values should be reasonable (not infinite or NaN)
        assert np.all(np.isfinite(csd.values))
        
        # CSD should have different magnitude than LFP
        csd_rms = float(np.sqrt(np.mean(csd.values**2)))
        lfp_rms = float(np.sqrt(np.mean(lfp.values**2)))
        
        # Should be different (exact relationship depends on implementation)
        assert csd_rms != lfp_rms
    
    def test_csd_spatial_derivatives(self):
        """Test that CSD computation produces spatial derivatives."""
        # Create artificial spatial gradient in LFP
        lfp = synthetic_lfp(duration=0.2, fs=1000, n_chans=9, seed=42)
        
        # Arrange in 3x3 grid
        x_coords = np.tile(np.arange(3), 3) * 50
        y_coords = np.repeat(np.arange(3), 3) * 50
        lfp = lfp.assign_coords({
            COORD_X: (DIM_CHAN, x_coords),
            COORD_Y: (DIM_CHAN, y_coords)
        })
        
        # Add artificial gradient (higher values at higher y)
        for i, y in enumerate(y_coords):
            lfp.values[:, i] += y * 0.01  # Small gradient
        
        csd = csd_2d(lfp, spacing=50)
        
        # CSD should detect the gradient
        # (More sophisticated testing would check specific patterns)
        assert not np.allclose(csd.values, 0, atol=1e-10)
    
    def test_metadata_preservation(self):
        """Test that CSD preserves important metadata."""
        lfp = synthetic_lfp(duration=0.3, fs=1000, n_chans=8, seed=42)
        lfp.attrs['experiment'] = 'csd_test'
        lfp.attrs['probe_type'] = 'neuropixels'
        
        # Add linear coordinates
        depths = np.arange(8) * 25
        lfp = lfp.assign_coords({COORD_Y: (DIM_CHAN, depths)})
        
        csd = csd_1d(lfp, spacing=25)
        
        # Should preserve custom metadata
        assert csd.attrs['experiment'] == 'csd_test'
        assert csd.attrs['probe_type'] == 'neuropixels'
        
        # Should preserve sampling rate info
        assert 'fs' in csd.attrs
    
    def test_edge_handling(self):
        """Test that CSD handles edge channels appropriately."""
        lfp = synthetic_lfp(duration=0.2, fs=1000, n_chans=8, seed=42)
        
        # Linear arrangement
        depths = np.arange(8) * 25
        lfp = lfp.assign_coords({COORD_Y: (DIM_CHAN, depths)})
        
        csd = csd_1d(lfp, spacing=25)
        
        # Edge channels might be handled differently
        # (implementation-dependent, but should not crash)
        assert isinstance(csd, xr.DataArray)
        assert csd.shape[0] == lfp.shape[0]  # Time preserved
        
        # Number of output channels depends on edge handling
        # (could be same as input, or fewer)
        assert csd.sizes[DIM_CHAN] <= lfp.sizes[DIM_CHAN]
        assert csd.sizes[DIM_CHAN] >= 2  # Should have at least some channels


if __name__ == '__main__':
    pytest.main([__file__])
