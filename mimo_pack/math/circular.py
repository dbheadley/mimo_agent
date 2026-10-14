# Circular statistics functions
# Author: Drew Headley
# Date: 2025-06-16

import numpy as np
import xarray as xr
import warnings
from mimo_pack.consts import DIM_TIME


def ppc2(phases: xr.DataArray, phase_dim: str = DIM_TIME) -> xr.DataArray:
    """
    Calculate the Pairwise Phase Consistency (PPC2) for phase values.
    
    PPC2 is an unbiased estimator of the squared Phase-Locking Value (PLV^2) 
    as described in Vinck et al. (2010). It measures the consistency of phase 
    angles, with values ranging from 0 (uniform distribution) to 1 (perfect 
    phase locking).

    Parameters
    ----------
    phases : xarray.DataArray
        Phase values in radians. PPC2 will be computed along the specified 
        phase dimension and broadcast over other dimensions.
    phase_dim : str, optional
        Name of the dimension along which to compute PPC2. Default is DIM_TIME.
        This dimension will be reduced in the output.

    Returns
    -------
    xarray.DataArray
        PPC2 values. The phase dimension is reduced, returning scalar values
        for each combination of remaining dimensions. Returns NaN when there 
        are fewer than 2 observations along the phase dimension.
        
    Raises
    ------
    TypeError
        If input is not an xarray.DataArray.
    ValueError
        If input does not have the specified phase dimension.
        
    Examples
    --------
    >>> import numpy as np
    >>> import xarray as xr
    >>> # Create phases concentrated around 0 (high PPC)
    >>> phases = xr.DataArray(np.random.vonmises(0, 5, 100), dims=['time'])
    >>> ppc_val = ppc2(phases)
    >>> ppc_val.values  # Should be close to 1
    
    >>> # Create uniform phases (low PPC)
    >>> phases_uniform = xr.DataArray(np.random.uniform(-np.pi, np.pi, 100), 
    ...                               dims=['time'])
    >>> ppc_uniform = ppc2(phases_uniform)
    >>> ppc_uniform.values  # Should be close to 0
    
    References
    ----------
    Vinck, M., van Wingerden, M., Womelsdorf, T., Fries, P., & Pennartz, C. M. (2010).
    The pairwise phase consistency: a bias-free measure of rhythmic neuronal synchronization.
    NeuroImage, 51(1), 112-122.
    
    Notes
    -----
    Created by Greg Glickert
    """
    if not isinstance(phases, xr.DataArray):
        raise TypeError(
            f"Input must be an xarray.DataArray, got {type(phases).__name__}. "
            "Convert numpy arrays to xarray.DataArray before calling this function."
        )
    
    if phase_dim not in phases.dims:
        raise ValueError(
            f"Input xarray must have a '{phase_dim}' dimension. "
            f"Found dimensions: {phases.dims}. "
            f"Specify the correct dimension using the phase_dim parameter."
        )
    
    def _ppc2_1d(phase_vals: np.ndarray) -> float:
        """Compute pairwise phase consistency (PPC2) for a 1-D array of phases.

        Parameters
        ----------
        phase_vals : numpy.ndarray
            1-D array of phase angles in radians.

        Returns
        -------
        float
            PPC2 value in ``[-1, 1]``, or ``NaN`` if fewer than 2 observations.
        """
        n = len(phase_vals)
        
        if n <= 1:
            warnings.warn(
                f"Cannot calculate PPC2 with {n} observation(s). "
                "Returning NaN. Need at least 2 phase values.",
                UserWarning
            )
            return np.nan
        
        # Convert phases to unit vectors in the complex plane
        unit_vectors = np.exp(1j * phase_vals)
        
        # Calculate the resultant vector
        resultant_vector = np.sum(unit_vectors)
        
        # PPC2 = (|∑(e^(i*φ_j))|² - n) / (n * (n - 1))
        ppc_val = (np.abs(resultant_vector) ** 2 - n) / (n * (n - 1))
        
        return ppc_val
    
    # Apply along phase dimension, broadcasting over other dimensions
    result = xr.apply_ufunc(
        _ppc2_1d,
        phases,
        input_core_dims=[[phase_dim]],
        vectorize=True,
        dask='parallelized'
    )
    
    # Add description for data provenance
    desc_base = phases.attrs.get('description', 'phase values')
    result.attrs['description'] = f"{desc_base} -> pairwise phase consistency (PPC2)"
    
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("CIRCULAR STATISTICS MODULE TESTS")
    print("=" * 60)
    
    # Test 1: High concentration (high PPC)
    print("\n1. Testing ppc2() with concentrated phases...")
    np.random.seed(42)
    phases_conc = xr.DataArray(
        np.random.vonmises(0, 5, 100),
        dims=['time'],
        attrs={'description': 'concentrated phases'}
    )
    ppc_conc = ppc2(phases_conc)
    print(f"   PPC2 for concentrated phases: {ppc_conc.values:.3f}")
    print(f"   Expected: ~0.8-1.0 (high phase locking)")
    
    # Test 2: Uniform distribution (low PPC)
    print("\n2. Testing ppc2() with uniform phases...")
    phases_uniform = xr.DataArray(
        np.random.uniform(-np.pi, np.pi, 100),
        dims=['time'],
        attrs={'description': 'uniform phases'}
    )
    ppc_uniform = ppc2(phases_uniform)
    print(f"   PPC2 for uniform phases: {ppc_uniform.values:.3f}")
    print(f"   Expected: ~0.0 (no phase locking)")
    
    # Test 3: Multi-dimensional input
    print("\n3. Testing ppc2() with multi-dimensional input...")
    # Generate phases for 3 units with different concentrations
    phases_multi_data = np.stack([
        np.random.vonmises(0, 1, 100),
        np.random.vonmises(0, 3, 100),
        np.random.vonmises(0, 5, 100)
    ])
    phases_multi = xr.DataArray(
        phases_multi_data,
        dims=['unit', 'time'],
        coords={'unit': [0, 1, 2]},
        attrs={'description': 'multi-unit phases'}
    )
    ppc_multi = ppc2(phases_multi, phase_dim='time')
    print(f"   PPC2 values for 3 units: {ppc_multi.values}")
    print(f"   Shape: {ppc_multi.shape} (time dimension reduced)")
    
    # Test 4: Edge case - insufficient data
    print("\n4. Testing ppc2() with edge cases...")
    phases_single = xr.DataArray([0.5], dims=['time'])
    ppc_single = ppc2(phases_single)
    print(f"   PPC2 with 1 observation: {ppc_single.values} (should be NaN)")
    
    # Test 5: Error handling
    print("\n5. Testing error handling...")
    try:
        ppc2(np.array([1, 2, 3]))
        print("   ✗ Should have raised TypeError")
    except TypeError:
        print("   ✓ Correctly raised TypeError for numpy array input")
    
    try:
        bad_xr = xr.DataArray([1, 2, 3], dims=['x'])
        ppc2(bad_xr, phase_dim='time')
        print("   ✗ Should have raised ValueError")
    except ValueError:
        print("   ✓ Correctly raised ValueError for missing dimension")
    
    print("\n" + "=" * 60)
    print("CIRCULAR STATISTICS TESTS COMPLETE")
    print("=" * 60)
