# Analysis functions for evoked responses

from mimo_pack.analysis.lfp import evoked_lfp
from mimo_pack.analysis.ap import evoked_ap
import inspect
import xarray as xr

def evoked_field_summary(ap_data, lfp_data, **kwargs):
    """Calculate and summarize evoked AP and LFP responses from epoch-windowed data.

    This function computes the evoked responses for both the action potential (AP)
    and local field potential (LFP) bands from pre-windowed xarray data (epoch windows).
    It then returns a summary of these responses.

    Parameters
    ----------
    ap_data : xarray.DataArray
        Epoch-windowed action potential (AP) band data with dimensions (epoch, time, channel).
        Use fileio.ap functions to load epoch windows first.
    lfp_data : xarray.DataArray
        Epoch-windowed local field potential (LFP) band data with dimensions (epoch, time, channel).
        Use fileio.lfp functions to load epoch windows first.

    Optional
    --------
    **kwargs
        Additional keyword arguments to pass to the evoked_lfp and evoked_ap functions.
        
    Returns
    -------
    dict
        A dictionary containing the summary of evoked responses with the following keys:
        - 'ap' : The evoked action potential envelope.
        - 'lfp' : The mean evoked local field potential.
        - 'csd' : The mean evoked current source density.
        
    Raises
    ------
    TypeError
        If inputs are not xarray.DataArray objects.

    See Also
    --------
    evoked_lfp : Function to compute evoked LFP and CSD from windowed data.
    evoked_ap : Function to compute evoked AP envelope from windowed data.
    
    Examples
    --------
    >>> # Load and window data first (done separately with fileio functions)
    >>> ap_windowed = load_ap_windows(ap_path, centers=times, pre=0.01, post=0.03)
    >>> lfp_windowed = load_lfp_windows(lfp_path, centers=times, pre=0.01, post=0.03)
    >>> # Then compute evoked responses
    >>> results = evoked_field_summary(ap_windowed, lfp_windowed)

    """
    # Validate inputs
    if not isinstance(ap_data, xr.DataArray):
        raise TypeError(
            f"ap_data must be an xarray.DataArray, got {type(ap_data).__name__}. "
            "Use fileio.ap functions to load windowed AP data first."
        )
        
    if not isinstance(lfp_data, xr.DataArray):
        raise TypeError(
            f"lfp_data must be an xarray.DataArray, got {type(lfp_data).__name__}. "
            "Use fileio.lfp functions to load windowed LFP data first."
        )
        
    # Get the expected keyword arguments for each function
    lfp_params = inspect.signature(evoked_lfp).parameters
    ap_params = inspect.signature(evoked_ap).parameters

    # Filter kwargs for each function
    lfp_kwargs = {k: v for k, v in kwargs.items() if k in lfp_params}
    ap_kwargs = {k: v for k, v in kwargs.items() if k in ap_params}

    # Call the functions with the windowed data
    lfp_mean, csd_mean = evoked_lfp(lfp_data, **lfp_kwargs)
    ap_env = evoked_ap(ap_data, **ap_kwargs)
    return {'ap': ap_env, 'lfp': lfp_mean, 'csd': csd_mean}
