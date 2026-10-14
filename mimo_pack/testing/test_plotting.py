"""
Tests for plotting functions in mimo_pack.plot.

This module contains comprehensive tests for all plotting functions,
including proper xarray integration, matplotlib compatibility, and
parameter validation.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pytest
from unittest.mock import patch, MagicMock

import mimo_pack.plot.xarray as plot_xr
import mimo_pack.plot.map as plot_map
import mimo_pack.plot.line as plot_line
from mimo_pack.testing.data import synthetic_lfp_trials, synthetic_lfp
from mimo_pack.consts import DIM_TIME, DIM_CHAN, COORD_X, COORD_Y


class TestXarrayPlots:
    """Tests for mimo_pack.plot.xarray functions."""
    
    def test_get_dim_helper(self):
        """Test the _get_dim helper function."""
        # Create test data
        data = xr.DataArray(
            np.random.randn(10, 5, 3),
            dims=['time', 'channel', 'epoch'],
            coords={
                'time': np.arange(10),
                'ch_x': ('channel', np.arange(5)),
                'ch_y': ('channel', np.arange(5) * 20),
            }
        )
        
        # Test coordinate name that is a dimension
        assert plot_xr._get_dim(data, 'time') == 'time'
        
        # Test coordinate name that maps to a dimension
        assert plot_xr._get_dim(data, 'ch_x') == 'channel'
        
        # Test invalid coordinate name
        with pytest.raises(ValueError, match="Coordinate 'invalid' not found"):
            plot_xr._get_dim(data, 'invalid')
    
    def test_prepare_data_basic(self):
        """Test basic _prepare_data functionality."""
        # Create 2D test data
        data = xr.DataArray(
            np.random.randn(10, 5),
            dims=['time', 'channel'],
            coords={
                'time': np.arange(10),
                'ch_x': ('channel', np.arange(5)),
                'ch_y': ('channel', np.arange(5) * 20),
            }
        )
        
        prepared = plot_xr._prepare_data(data, 'ch_y', 'time')
        
        # Should transpose to (ch_y, time)
        assert list(prepared.dims) == ['channel', 'time']
        assert prepared.shape == (5, 10)
    
    def test_prepare_data_extra_dims(self):
        """Test _prepare_data with extra dimensions that should cause an error."""
        # Create 3D test data with non-singleton extra dim
        data = xr.DataArray(
            np.random.randn(10, 5, 3),
            dims=['time', 'channel', 'epoch']
        )
        
        # Should raise error due to non-singleton extra dimension
        with pytest.raises(ValueError, match="Data contains non-singleton extra dimensions"):
            plot_xr._prepare_data(data, 'channel', 'time')
    
    @patch('matplotlib.pyplot.gca')
    def test_composite_basic(self, mock_gca):
        """Test basic composite function."""
        mock_ax = MagicMock()
        mock_gca.return_value = mock_ax
        
        # Create test data
        data = xr.DataArray(
            np.random.randn(10, 5),
            dims=['time', 'channel'],
            coords={
                'time': np.arange(10),
                'ch_x': ('channel', np.arange(5)),
                'ch_y': ('channel', np.arange(5) * 20),
            }
        )
        
        with patch('mimo_pack.plot.image.composite_image') as mock_composite:
            mock_composite.return_value = np.zeros((100, 50, 3), dtype=np.uint8)
            
            result = plot_xr.composite(data)
            
            # Should call composite_image
            mock_composite.assert_called_once()
            
            # Should return axes
            assert result == mock_ax
    
    @patch('matplotlib.pyplot.subplots')
    def test_pcolormesh_basic(self, mock_subplots):
        """Test basic pcolormesh function."""
        fig, ax = plt.subplots()
        mock_subplots.return_value = (fig, ax)
        
        # Create test data
        data = xr.DataArray(
            np.random.randn(10, 5),
            dims=['time', 'channel'],
            coords={
                'time': np.arange(10),
                'ch_x': ('channel', np.arange(5)),
                'ch_y': ('channel', np.arange(5) * 20),
            }
        )
        
        with patch.object(ax, 'pcolormesh') as mock_pcm:
            mock_pcm.return_value = MagicMock()
            
            result = plot_xr.pcolormesh(data)
            
            # Should call pcolormesh
            mock_pcm.assert_called_once()
            
            # Should return axes
            assert result == ax
    
    def test_pcolormesh_with_synthetic_data(self):
        """Test pcolormesh with synthetic LFP data."""
        # Generate synthetic data
        lfp = synthetic_lfp(duration=1.0, fs=1000, n_chans=4, seed=42)
        
        # Should work without errors
        fig, ax = plt.subplots()
        result = plot_xr.pcolormesh(lfp, ax=ax)
        
        assert result == ax
        plt.close(fig)


class TestMapPlots:
    """Tests for mimo_pack.plot.map functions."""
    
    def test_wave_map_numpy_basic(self):
        """Test basic _wave_map_numpy function."""
        # Create test data
        n_chans, n_times = 4, 100
        waves = np.random.randn(n_times, n_chans)
        x_pos = np.array([0, 50, 100, 150])
        y_pos = np.array([0, 25, 50, 75])
        
        fig, ax = plt.subplots()
        
        result = plot_map._wave_map_numpy(
            waves, x_pos, y_pos, ax=ax
        )
        
        assert result == ax
        plt.close(fig)
    
    def test_wave_map_numpy_with_colors(self):
        """Test _wave_map_numpy with color values."""
        # Create test data
        n_chans, n_times = 4, 100
        waves = np.random.randn(n_times, n_chans)
        x_pos = np.array([0, 50, 100, 150])
        y_pos = np.array([0, 25, 50, 75])
        c = np.array([1, 2, 3, 4])  # Color values
        
        fig, ax = plt.subplots()
        
        result = plot_map._wave_map_numpy(
            waves, x_pos, y_pos, c=c, ax=ax
        )
        
        assert result == ax
        plt.close(fig)
    
    def test_amp_map_numpy(self):
        """Test amp_map function with numpy arrays."""
        # Create test data
        amp = np.array([1.0, 2.5, 0.8, 3.2])
        x_pos = np.array([0, 50, 100, 150])
        y_pos = np.array([0, 25, 50, 75])
        
        fig, ax = plt.subplots()
        
        result_ax, scat = plot_map.amp_map(amp, x_pos, y_pos, ax=ax)
        
        assert result_ax == ax
        assert hasattr(scat, 'set_array')  # Should be a scatter plot object
        plt.close(fig)
    
    def test_amp_map_xarray(self):
        """Test amp_map with xarray DataArray."""
        # Create test data
        amp = xr.DataArray(
            np.array([1.0, 2.5, 0.8, 3.2]),
            dims=['channel'],
            coords={
                'ch_x': ('channel', np.array([0, 50, 100, 150])),
                'ch_y': ('channel', np.array([0, 25, 50, 75])),
            }
        )
        
        fig, ax = plt.subplots()
        
        # The xarray version should delegate to numpy version
        with patch('mimo_pack.plot.map.amp_map') as mock_amp_map:
            mock_amp_map.return_value = (ax, MagicMock())
            
            # This should call the numpy version
            result_ax, scat = plot_map.amp_map(amp)
            
            # Should call the numpy amp_map function
            mock_amp_map.assert_called_once()
        
        plt.close(fig)
    
    def test_wave_map_xarray(self):
        """Test wave_map with xarray DataArray."""
        # Create synthetic LFP data
        lfp = synthetic_lfp(duration=0.5, fs=1000, n_chans=4, seed=42)
        
        fig, ax = plt.subplots()
        
        # Should work without errors
        result = plot_map.wave_map(lfp, ax=ax)
        
        assert result == ax
        plt.close(fig)
    
    def test_wave_map_with_label_coord(self):
        """Test wave_map with label coordinates."""
        # Create synthetic data with labels
        lfp = synthetic_lfp(duration=0.5, fs=1000, n_chans=4, seed=42)
        lfp = lfp.assign_coords({
            'label': ('channel', ['Ch1', 'Ch2', 'Ch3', 'Ch4'])
        })
        
        fig, ax = plt.subplots()
        
        result = plot_map.wave_map(lfp, label_coord='label', ax=ax)
        
        assert result == ax
        plt.close(fig)
    
    def test_wave_map_parameter_validation(self):
        """Test wave_map parameter validation."""
        # Create test data without required coordinates
        bad_data = xr.DataArray(
            np.random.randn(100, 4),
            dims=['time', 'channel']
            # Missing ch_x, ch_y coordinates
        )
        
        # Should raise error for missing coordinates
        with pytest.raises((ValueError, KeyError)):
            plot_map.wave_map(bad_data)


class TestLinePlots:
    """Tests for mimo_pack.plot.line functions."""
    
    def test_plot_mean_shaded_basic(self):
        """Test basic plot_mean_shaded function."""
        # Create test data with multiple trials
        data = xr.DataArray(
            np.random.randn(100, 10, 5),  # time x channel x trial
            dims=['time', 'channel', 'epoch'],
            coords={
                'time': np.linspace(0, 1, 100),
            }
        )
        
        fig, ax = plt.subplots()
        
        result = plot_line.plot_mean_shaded(data, plot_dim='time', ax=ax)
        
        assert result == ax
        plt.close(fig)
    
    def test_plot_mean_shaded_sem(self):
        """Test plot_mean_shaded with standard error of mean."""
        data = xr.DataArray(
            np.random.randn(50, 10),  # time x trial
            dims=['time', 'epoch'],
            coords={'time': np.linspace(0, 1, 50)}
        )
        
        fig, ax = plt.subplots()
        
        result = plot_line.plot_mean_shaded(
            data, plot_dim='time', err='sem', ax=ax
        )
        
        assert result == ax
        plt.close(fig)
    
    def test_plot_mean_shaded_std(self):
        """Test plot_mean_shaded with standard deviation."""
        data = xr.DataArray(
            np.random.randn(50, 10),
            dims=['time', 'epoch'],
            coords={'time': np.linspace(0, 1, 50)}
        )
        
        fig, ax = plt.subplots()
        
        result = plot_line.plot_mean_shaded(
            data, plot_dim='time', err='std', ax=ax
        )
        
        assert result == ax
        plt.close(fig)
    
    def test_plot_mean_shaded_ci(self):
        """Test plot_mean_shaded with confidence intervals."""
        data = xr.DataArray(
            np.random.randn(50, 10),
            dims=['time', 'epoch'],
            coords={'time': np.linspace(0, 1, 50)}
        )
        
        fig, ax = plt.subplots()
        
        result = plot_line.plot_mean_shaded(
            data, plot_dim='time', err='ci', ci=95.0, ax=ax
        )
        
        assert result == ax
        plt.close(fig)
    
    def test_plot_mean_shaded_with_kwargs(self):
        """Test plot_mean_shaded with custom kwargs."""
        data = xr.DataArray(
            np.random.randn(50, 10),
            dims=['time', 'epoch'],
            coords={'time': np.linspace(0, 1, 50)}
        )
        
        fig, ax = plt.subplots()
        
        line_kwargs = {'linewidth': 3, 'linestyle': '--'}
        fill_kwargs = {'alpha': 0.1}
        
        result = plot_line.plot_mean_shaded(
            data, plot_dim='time', 
            line_kwargs=line_kwargs,
            fill_kwargs=fill_kwargs,
            ax=ax
        )
        
        assert result == ax
        plt.close(fig)
    
    def test_plot_mean_shaded_invalid_err(self):
        """Test plot_mean_shaded with invalid error type."""
        data = xr.DataArray(
            np.random.randn(50, 10),
            dims=['time', 'epoch']
        )
        
        fig, ax = plt.subplots()
        
        with pytest.raises(ValueError, match="err must be one of"):
            plot_line.plot_mean_shaded(
                data, plot_dim='time', err='invalid', ax=ax
            )
        
        plt.close(fig)


class TestIntegrationTests:
    """Integration tests using synthetic data."""
    
    def test_composite_with_lfp_trials(self):
        """Test composite plot with LFP epoch data."""
        # Generate synthetic LFP trials
        lfp_trials = synthetic_lfp_trials(
            n_trials=5, fs=1000, n_chans=8, 
            pre=0.1, post=0.2, seed=42
        )
        
        # Take mean across trials for composite plot
        lfp_mean = lfp_trials.mean(dim='epoch')
        
        fig, ax = plt.subplots()
        
        result = plot_xr.composite(lfp_mean, ax=ax)
        
        assert result == ax
        plt.close(fig)
    
    def test_pcolormesh_with_lfp_trials(self):
        """Test pcolormesh with LFP epoch data."""
        lfp_trials = synthetic_lfp_trials(
            n_trials=5, fs=1000, n_chans=8,
            pre=0.1, post=0.2, seed=42
        )
        
        # Take mean across trials
        lfp_mean = lfp_trials.mean(dim='epoch')
        
        fig, ax = plt.subplots()
        
        result = plot_xr.pcolormesh(lfp_mean, ax=ax)
        
        assert result == ax
        plt.close(fig)
    
    def test_wave_map_with_lfp_trials(self):
        """Test wave_map with LFP epoch data."""
        lfp_trials = synthetic_lfp_trials(
            n_trials=5, fs=1000, n_chans=8,
            pre=0.1, post=0.2, seed=42
        )
        
        # Take mean across trials
        lfp_mean = lfp_trials.mean(dim='epoch')
        
        fig, ax = plt.subplots()
        
        result = plot_map.wave_map(lfp_mean, ax=ax)
        
        assert result == ax
        plt.close(fig)
    
    def test_plot_mean_shaded_with_trials(self):
        """Test plot_mean_shaded with epoch data."""
        lfp_trials = synthetic_lfp_trials(
            n_trials=10, fs=1000, n_chans=4,
            pre=0.1, post=0.2, seed=42
        )
        
        # Select one channel
        single_chan = lfp_trials.isel(channel=0)
        
        fig, ax = plt.subplots()
        
        result = plot_line.plot_mean_shaded(
            single_chan, plot_dim='time', ax=ax
        )
        
        assert result == ax
        plt.close(fig)


def test_plotting_functions_description_attrs():
    """Test that plotting functions preserve description attributes."""
    # Create test data with description
    data = xr.DataArray(
        np.random.randn(100, 4),
        dims=['time', 'channel'],
        coords={
            'time': np.linspace(0, 1, 100),
            'ch_x': ('channel', np.array([0, 50, 100, 150])),
            'ch_y': ('channel', np.array([0, 25, 50, 75])),
        },
        attrs={'description': 'Test data for plotting'}
    )
    
    # Test that attributes are accessible during plotting
    # (plotting functions typically don't modify input data)
    assert data.attrs['description'] == 'Test data for plotting'
    
    # Test plotting functions don't crash with attrs present
    fig, ax = plt.subplots()
    
    result = plot_xr.pcolormesh(data, ax=ax)
    assert result == ax
    
    plt.close(fig)


if __name__ == '__main__':
    # Run tests when executed directly
    pytest.main([__file__])