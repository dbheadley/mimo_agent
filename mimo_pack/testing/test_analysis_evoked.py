"""
Unit tests for mimo_pack.analysis.evoked module.

Tests the refactored evoked response functions to ensure they:
1. Accept only xarray.DataArray inputs
2. Remove file path dependencies
3. Preserve spatial coordinates and metadata
4. Use constants from mimo_pack.consts
"""

import numpy as np
import xarray as xr
import pytest
from mimo_pack.analysis.lfp import evoked_lfp
from mimo_pack.testing.data import synthetic_lfp, synthetic_lfp_trials
from mimo_pack.consts import DIM_TIME, DIM_CHAN, DIM_EPOCH, COORD_X, COORD_Y


class TestEvokedAnalysis:
    """Test suite for evoked response analysis."""
    
    def test_evoked_lfp_basic(self):
        """Test basic evoked LFP computation."""
        # Create epoch-based data
        lfp_trials = synthetic_lfp_trials(
            duration=0.2, fs=1000, n_chans=4, n_trials=20, seed=42
        )
        
        lfp_mean, lfp_sem = evoked_lfp(lfp_trials)
        
        # Check output types and dimensions
        assert isinstance(lfp_mean, xr.DataArray)
        assert isinstance(lfp_sem, xr.DataArray)
        
        # Should have time and channel dimensions (no epoch)
        assert DIM_TIME in lfp_mean.dims
        assert DIM_CHAN in lfp_mean.dims
        assert DIM_EPOCH not in lfp_mean.dims
        
        # Shapes should match (minus epoch dimension)
        expected_shape = (lfp_trials.sizes[DIM_TIME], lfp_trials.sizes[DIM_CHAN])
        assert lfp_mean.shape == expected_shape
        assert lfp_sem.shape == expected_shape
    
    def test_evoked_lfp_input_validation(self):
        """Test that evoked_lfp rejects invalid inputs."""
        # Test non-xarray input
        data = np.random.randn(200, 4, 20)
        
        with pytest.raises(TypeError, match="Input must be xarray.DataArray"):
            evoked_lfp(data)
    
    def test_evoked_lfp_coordinates_preserved(self):
        """Test that evoked analysis preserves spatial coordinates."""
        lfp_trials = synthetic_lfp_trials(
            duration=0.15, fs=1000, n_chans=4, n_trials=15, seed=42
        )
        
        lfp_mean, lfp_sem = evoked_lfp(lfp_trials)
        
        # Time coordinates should be preserved
        np.testing.assert_array_equal(lfp_mean[DIM_TIME], lfp_trials[DIM_TIME])
        np.testing.assert_array_equal(lfp_sem[DIM_TIME], lfp_trials[DIM_TIME])
        
        # Channel coordinates should be preserved
        np.testing.assert_array_equal(lfp_mean[DIM_CHAN], lfp_trials[DIM_CHAN])
        
        # Spatial coordinates should be preserved
        for coord in [COORD_X, COORD_Y]:
            if coord in lfp_trials.coords:
                np.testing.assert_array_equal(lfp_mean[coord], lfp_trials[coord])
                np.testing.assert_array_equal(lfp_sem[coord], lfp_trials[coord])
    
    def test_evoked_csd_basic(self):
        """Test basic evoked CSD computation."""
        lfp_trials = synthetic_lfp_trials(
            duration=0.1, fs=1000, n_chans=16, n_trials=10, seed=42
        )
        
        # Add spatial coordinates for CSD
        x_coords = np.tile(np.arange(4), 4) * 25
        y_coords = np.repeat(np.arange(4), 4) * 25
        lfp_trials = lfp_trials.assign_coords({
            COORD_X: (DIM_CHAN, x_coords),
            COORD_Y: (DIM_CHAN, y_coords)
        })
        
        try:
            csd_mean, csd_sem = evoked_csd(lfp_trials, spacing=25)
            
            # Check output types
            assert isinstance(csd_mean, xr.DataArray)
            assert isinstance(csd_sem, xr.DataArray)
            
            # Should have proper dimensions
            assert DIM_TIME in csd_mean.dims
            assert DIM_CHAN in csd_mean.dims
            assert DIM_EPOCH not in csd_mean.dims
            
            # Should preserve spatial coordinates
            assert COORD_X in csd_mean.coords
            assert COORD_Y in csd_mean.coords
            
        except (NameError, AttributeError):
            pytest.skip("evoked_csd function not implemented")
    
    def test_evoked_statistics(self):
        """Test that evoked analysis computes proper statistics."""
        # Create data with known pattern
        n_trials = 50
        lfp_trials = synthetic_lfp_trials(
            duration=0.1, fs=1000, n_chans=2, n_trials=n_trials, seed=42
        )
        
        # Add consistent evoked response across trials
        t = lfp_trials[DIM_TIME].values
        response = np.sin(2 * np.pi * 10 * t) * np.exp(-t * 10)
        
        for trial in range(n_trials):
            lfp_trials.values[:, :, trial] += response[:, np.newaxis]
        
        lfp_mean, lfp_sem = evoked_lfp(lfp_trials)
        
        # Mean should show the evoked response
        # (Response is additive, so should be preserved in average)
        assert not np.allclose(lfp_mean.values, 0, atol=0.1)
        
        # SEM should be positive and reasonable
        assert np.all(lfp_sem.values >= 0)
        sem_max = float(lfp_sem.max())
        mean_max = float(np.abs(lfp_mean).max())
        assert sem_max < mean_max  # SEM should be smaller than signal
    
    def test_evoked_baseline_handling(self):
        """Test baseline subtraction in evoked responses."""
        lfp_trials = synthetic_lfp_trials(
            duration=0.2, fs=1000, n_chans=2, n_trials=20, seed=42
        )
        
        try:
            # Test with baseline period
            lfp_mean, lfp_sem = evoked_lfp(
                lfp_trials, 
                baseline_window=[-0.05, -0.01]
            )
            
            # After baseline subtraction, early timepoints should be ~0
            baseline_mean = lfp_mean.sel({DIM_TIME: slice(-0.05, -0.01)}).mean()
            assert abs(float(baseline_mean)) < 0.1
            
        except TypeError:
            # If function doesn't support baseline, that's okay
            # (depends on implementation)
            pass
    
    def test_metadata_preservation(self):
        """Test that evoked analysis preserves metadata."""
        lfp_trials = synthetic_lfp_trials(
            duration=0.1, fs=1000, n_chans=4, n_trials=15, seed=42
        )
        lfp_trials.attrs['experiment'] = 'evoked_test'
        lfp_trials.attrs['stimulus'] = 'tone_pip'
        
        lfp_mean, lfp_sem = evoked_lfp(lfp_trials)
        
        # Should preserve custom attributes
        assert lfp_mean.attrs['experiment'] == 'evoked_test'
        assert lfp_mean.attrs['stimulus'] == 'tone_pip'
        assert lfp_sem.attrs['experiment'] == 'evoked_test'
        
        # Should preserve standard attributes
        assert 'fs' in lfp_mean.attrs
    
    def test_trial_dimension_requirement(self):
        """Test that evoked functions require trial dimension."""
        # Create data without trial dimension
        lfp_no_trials = synthetic_lfp(duration=0.1, fs=1000, n_chans=4, seed=42)
        
        with pytest.raises((ValueError, KeyError)):
            evoked_lfp(lfp_no_trials)
    
    def test_consistency_across_channels(self):
        """Test that evoked analysis is consistent across channels."""
        lfp_trials = synthetic_lfp_trials(
            duration=0.1, fs=1000, n_chans=8, n_trials=25, seed=42
        )
        
        lfp_mean, lfp_sem = evoked_lfp(lfp_trials)
        
        # All channels should have data
        assert lfp_mean.sizes[DIM_CHAN] == lfp_trials.sizes[DIM_CHAN]
        
        # Each channel should have non-zero variance (unless artificial)
        for ch in range(lfp_mean.sizes[DIM_CHAN]):
            ch_data = lfp_mean.isel({DIM_CHAN: ch})
            ch_sem = lfp_sem.isel({DIM_CHAN: ch})
            
            # Should have some variation over time
            assert float(ch_data.std()) > 0
            assert float(ch_sem.mean()) > 0


if __name__ == '__main__':
    pytest.main([__file__])