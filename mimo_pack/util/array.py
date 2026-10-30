# Short generic utility functions for working with numeric arrays
# Author: Drew Headley
# Date: 2024-06-13

import numpy as np

# Split an array into multiple sub-arrays that are copies of the original.
def split_copy(array, *args, **kwargs):
    """
    Split an array into multiple sub-arrays that are copies of the original.
    The arguments are the same as numpy.split, but the resulting arrays are copies
    of the original, not views. This is useful when you need to modify the sub-arrays
    without affecting the original array. The original array is not modified.
    
    Parameters
    ----------
    array : np.ndarray
        The array to split.
    *args : int
        The arguments to pass to np.split.
    **kwargs : dict
        The keyword arguments to pass to np.split.
        
    Returns
    -------
    list
        A list of the sub-arrays, each of which is a copy of the original.
    """
    
    return [np.copy(x) for x in np.split(array, *args, **kwargs)]


def trimmed_mean(array, axis, trim=0.05):
    """
    Mean over an axis after discarding the items farthest from the mean.

    Each item along `axis` (e.g., one spike waveform) is ranked by its L2
    distance from the mean of all items. The farthest `trim` fraction is
    discarded and the mean of the remaining items is returned.

    Parameters
    ----------
    array : np.ndarray
        The array to average.
    axis : int
        The axis whose items are ranked and averaged over.
    trim : float, optional
        Fraction of items to discard. Default is 0.05.

    Returns
    -------
    np.ndarray
        The trimmed mean with `axis` removed.
    """

    items = np.moveaxis(array, axis, 0)
    flat = items.reshape(items.shape[0], -1)
    dist = np.linalg.norm(flat - flat.mean(axis=0), axis=1)
    keep = np.argsort(dist)[:items.shape[0] - int(trim * items.shape[0])]
    return items[keep].mean(axis=0)
