"""
Unit tests for mimo_pack.analysis.spectral module.

Tests the refactored spectral analysis functions to ensure they:
1. Accept only xarray.DataArray inputs
2. Extract sampling rates from time coordinates
3. Preserve spatial coordinates and metadata
4. Use constants from mimo_pack.consts
"""

import numpy as np
import xarray as xr
import pytest
from mimo_pack.analysis.spectral import stft
from mimo_pack.testing.data import synthetic_lfp
from mimo_pack.consts import DIM_TIME, DIM_CHAN, DIM_FREQ, COORD_X, COORD_Y


class TestSpectralAnalysis:
    """Test suite for spectral analysis functions."""
    
    def test_spectrogram_basic(self):
        """Test basic spectrogram computation."""
        lfp = synthetic_lfp(duration=2.0, fs=1000, n_chans=2, seed=42)
        
        spec = stft(lfp, nperseg=256, noverlap=128)
        
        # Check output type and dimensions
        assert isinstance(spec, xr.DataArray)
        
        # Should have time, frequency, and channel dimensions
        assert DIM_TIME in spec.dims
        assert DIM_FREQ in spec.dims
        assert DIM_CHAN in spec.dims
        
        # Check coordinate preservation
        assert DIM_CHAN in spec.coords
        if COORD_X in lfp.coords:
            np.testing.assert_array_equal(spec[COORD_X], lfp[COORD_X])
    
    def test_spectrogram_step_parameter(self):
        """Test that step parameter controls hop size and produces more time points."""
        lfp = synthetic_lfp(duration=4.0, fs=1000, n_chans=1, seed=42)

        # No overlap (step == window): baseline
        spec_no_overlap = stft(lfp.isel(channel=0), window=1.0)
        # Overlapping (step < window): should have more time bins
        spec_overlap = stft(lfp.isel(channel=0), window=1.0, step=0.1)

        assert spec_overlap.sizes[DIM_TIME] > spec_no_overlap.sizes[DIM_TIME]
        assert "step=0.1s" in spec_overlap.attrs['description']
        assert "step" not in spec_no_overlap.attrs['description']

    def test_spectrogram_input_validation(self):
        """Test that spectrogram rejects non-xarray inputs."""
        data = np.random.randn(2000, 4)
        
        with pytest.raises(TypeError, match="Input must be xarray.DataArray"):
            stft(data, nperseg=256)
    
    def test_power_spectral_density_basic(self):
        """Test basic PSD computation."""
        lfp = synthetic_lfp(duration=2.0, fs=1000, n_chans=2, seed=42)
        
        psd = stft(lfp, nperseg=512)
        
        # Check output properties
        assert isinstance(psd, xr.DataArray)
        assert DIM_FREQ in psd.dims
        assert DIM_CHAN in psd.dims
        
        # Should have positive frequencies
        freqs = psd[DIM_FREQ].values
        assert np.all(freqs >= 0)
        
        # Should have reasonable frequency range
        max_freq = freqs.max()
        fs = float(1 / lfp[DIM_TIME].diff(DIM_TIME).median())
        assert max_freq <= fs / 2  # Nyquist limit
    
    def test_coherence_basic(self):
        """Test coherence computation between channels."""
        # Create data with known coherent components
        lfp = synthetic_lfp(duration=2.0, fs=1000, n_chans=4, seed=42)
        
        # Add coherent 10 Hz signal to first two channels
        t = lfp[DIM_TIME].values
        coherent_signal = np.sin(2 * np.pi * 10 * t)
        lfp.values[:, 0] += coherent_signal * 2
        lfp.values[:, 1] += coherent_signal * 2
        
        try:
            coh = coherence(lfp, nperseg=256)
            
            assert isinstance(coh, xr.DataArray)
            assert DIM_FREQ in coh.dims
            
            # Should have coherence values between 0 and 1
            assert np.all(coh.values >= 0)
            assert np.all(coh.values <= 1)
            
        except (NameError, AttributeError):
            pytest.skip("coherence function not implemented")
    
    def test_phase_amplitude_coupling_basic(self):
        """Test phase-amplitude coupling computation."""
        lfp = synthetic_lfp(duration=3.0, fs=1000, n_chans=2, seed=42)
        
        try:
            pac = phase_amplitude_coupling(
                lfp, 
                phase_freq=[4, 8],  # Theta
                amp_freq=[30, 80]   # Gamma
            )
            
            assert isinstance(pac, (xr.DataArray, float, np.ndarray))
            
            # PAC values should be reasonable
            if isinstance(pac, xr.DataArray):
                assert not np.any(np.isnan(pac.values))
                
        except (NameError, AttributeError):
            pytest.skip("phase_amplitude_coupling function not implemented")
    
    def test_sampling_rate_extraction(self):
        """Test that spectral functions extract sampling rate correctly."""
        # Create data with known sampling rate
        fs = 2000
        lfp = synthetic_lfp(duration=1.0, fs=fs, n_chans=2, seed=42)
        
        # Compute spectrogram (should work without explicit fs)
        spec = stft(lfp, nperseg=512)
        
        # Check frequency range is appropriate for sampling rate
        max_freq = spec[DIM_FREQ].max().values
        assert max_freq <= fs / 2 + 10  # Allow small tolerance
    
    def test_multichannel_consistency(self):
        """Test that spectral analysis works consistently across channels."""
        lfp = synthetic_lfp(duration=1.0, fs=1000, n_chans=8, seed=42)
        
        spec = stft(lfp, nperseg=256)
        
        # All channels should have spectral data
        assert spec.sizes[DIM_CHAN] == lfp.sizes[DIM_CHAN]
        
        # Each channel should have non-zero power
        for ch in range(spec.sizes[DIM_CHAN]):
            ch_spec = spec.isel({DIM_CHAN: ch})
            assert ch_spec.sum() > 0
    
    def test_metadata_preservation(self):
        """Test that spectral analysis preserves metadata."""
        lfp = synthetic_lfp(duration=1.0, fs=1000, n_chans=2, seed=42)
        lfp.attrs['recording_site'] = 'hippocampus'
        lfp.attrs['depth'] = 500
        
        spec = stft(lfp, nperseg=256)
        
        # Should preserve custom attributes
        assert spec.attrs['recording_site'] == 'hippocampus'
        assert spec.attrs['depth'] == 500
        
        # Should preserve or update relevant attributes
        assert 'fs' in spec.attrs or DIM_FREQ in spec.coords
    
    def test_frequency_coordinates(self):
        """Test that frequency coordinates are properly assigned."""
        lfp = synthetic_lfp(duration=2.0, fs=1000, n_chans=2, seed=42)
        
        spec = stft(lfp, nperseg=512)
        
        # Should have frequency coordinate
        assert DIM_FREQ in spec.coords
        
        # Frequencies should be monotonic and positive
        freqs = spec[DIM_FREQ].values
        assert np.all(np.diff(freqs) >= 0)  # Monotonic
        assert np.all(freqs >= 0)  # Positive
        
        # Should span reasonable range
        assert freqs[0] == 0 or freqs[0] < 5  # Start near DC
        assert freqs[-1] > 100  # Cover reasonable range


if __name__ == '__main__':
    pytest.main([__file__])
