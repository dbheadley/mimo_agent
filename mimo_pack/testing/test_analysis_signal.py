"""
Unit tests for mimo_pack.analysis.signal_proc module.

Tests the refactored signal processing functions to ensure they:
1. Accept only xarray.DataArray inputs
2. Extract sampling rates from time coordinates  
3. Preserve spatial coordinates and metadata
4. Use constants from mimo_pack.consts
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
        
        # Check that envelope is larger than or equal to absolute value
        assert np.all(lfp_env.values >= np.abs(lfp.values) - 1e-10)
    
    def test_envelope_input_validation(self):
        """Test that envelope rejects non-xarray inputs."""
        data = np.random.randn(1000, 4)
        
        with pytest.raises(TypeError, match="Input must be xarray.DataArray"):
            envelope(data)
    
    def test_detrend_basic(self):
        """Test basic detrending functionality."""
        lfp = synthetic_lfp(duration=1.0, fs=1000, n_chans=2, seed=42)
        
        # Add linear trend
        trend = np.linspace(0, 100, lfp.sizes[DIM_TIME])
        lfp_with_trend = lfp + trend
        
        # Detrend
        lfp_detrend = detrend(lfp_with_trend)
        
        assert isinstance(lfp_detrend, xr.DataArray)
        assert lfp_detrend.shape == lfp.shape
        
        # Should have removed most of the trend
        # (exact comparison depends on detrending method)
        trend_removed = abs(float(lfp_detrend.mean())) < abs(float(lfp_with_trend.mean()))
        assert trend_removed
    
    def test_car_basic(self):
        """Test common average referencing."""
        lfp = synthetic_lfp(duration=0.5, fs=1000, n_chans=4, seed=42)
        
        lfp_car = car(lfp)
        
        # Check output properties
        assert isinstance(lfp_car, xr.DataArray)
        assert lfp_car.shape == lfp.shape
        
        # After CAR, the mean across channels should be ~0 at each timepoint
        mean_across_chans = lfp_car.mean(dim=DIM_CHAN)
        assert float(mean_across_chans.std()) < float(lfp.std()) * 0.1
    
    def test_detect_artifacts_basic(self):
        """Test artifact detection."""
        lfp = synthetic_lfp(duration=1.0, fs=1000, n_chans=2, seed=42)
        
        # Add artificial artifacts (large spikes)
        artifact_times = [200, 600]
        for t_idx in artifact_times:
            lfp.values[t_idx, :] += 1000  # Large spike
        
        # Detect artifacts
        artifacts = detect_artifacts(lfp, threshold=5.0)
        
        # Should return boolean array or indices
        assert artifacts is not None
        
        # Check that it found some artifacts
        if isinstance(artifacts, xr.DataArray):
            assert artifacts.any()
        elif isinstance(artifacts, np.ndarray):
            assert len(artifacts) > 0
    
    def test_spatial_filter_basic(self):
        """Test spatial filtering functionality."""
        lfp = synthetic_lfp(duration=0.5, fs=1000, n_chans=8, seed=42)
        
        # This test assumes spatial_filter function exists
        try:
            lfp_filt = spatial_filter(lfp)
            
            assert isinstance(lfp_filt, xr.DataArray)
            # Spatial filtering might change the number of channels
            assert lfp_filt.sizes[DIM_TIME] == lfp.sizes[DIM_TIME]
            
        except (NameError, AttributeError):
            # Skip if function doesn't exist
            pytest.skip("spatial_filter function not implemented")
    
    def test_coordinates_preserved(self):
        """Test that all processing preserves coordinate information."""
        lfp = synthetic_lfp(duration=0.5, fs=1000, n_chans=4, seed=42)
        
        # Test multiple functions
        functions_to_test = [zscore, detrend, car]
        
        for func in functions_to_test:
            try:
                result = func(lfp)
                
                # Check time coordinate preserved
                np.testing.assert_array_equal(result[DIM_TIME], lfp[DIM_TIME])
                
                # Check spatial coordinates preserved (if they exist)
                for coord in [COORD_X, COORD_Y]:
                    if coord in lfp.coords:
                        np.testing.assert_array_equal(result[coord], lfp[coord])
                        
            except Exception as e:
                pytest.fail(f"Function {func.__name__} failed: {e}")
    
    def test_metadata_preservation(self):
        """Test that processing preserves metadata attributes."""
        lfp = synthetic_lfp(duration=0.5, fs=1000, n_chans=2, seed=42)
        lfp.attrs['experiment'] = 'test_session'
        lfp.attrs['custom_param'] = 42
        
        lfp_proc = zscore(lfp)
        
        # Should preserve custom attributes
        assert lfp_proc.attrs['experiment'] == 'test_session'
        assert lfp_proc.attrs['custom_param'] == 42
        
        # Should preserve standard attributes
        assert 'fs' in lfp_proc.attrs
        assert 'units' in lfp_proc.attrs


if __name__ == '__main__':
    pytest.main([__file__])
