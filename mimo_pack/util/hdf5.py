# Short generic utility functions for working with hdf5 objects
# Author: Drew Headley
# Date: 2025-07-22

import h5py
import numpy as np

def get_h5_hierarchy(h5_data: h5py.File) -> list[tuple]:
    """Traverse an HDF5 file and return its complete hierarchy.

    Uses ``h5py.File.visititems`` to recursively visit every group and
    dataset, collecting the path, type, and (for datasets) the data.

    Parameters
    ----------
    h5_data : h5py.File
        An opened HDF5 file object to traverse.

    Returns
    -------
    list of tuple
        Each element is ``(path, item_type, value)`` where *path* is
        the full HDF5 path (str), *item_type* is ``'Group'``,
        ``'Dataset'``, or ``'Unknown'``, and *value* is a numpy array
        for datasets or ``None`` otherwise.
    """
    hierarchy = []
    
    def visit_item(name, obj):
        """Callback for ``visititems``: classify and store each HDF5 node.

        Parameters
        ----------
        name : str
            Hierarchical path of the item within the file.
        obj : h5py.Group or h5py.Dataset
            The HDF5 object at *name*.
        """
        item_type = ""
        if isinstance(obj, h5py.Group):
            item_type = "Group"
            item_value = None
        elif isinstance(obj, h5py.Dataset):
            item_type = "Dataset"
            item_value = np.array(obj)
        else:
            item_type = "Unknown"
            item_value = None
            
        hierarchy.append((name, item_type, item_value))

    h5_data.visititems(visit_item)
        
    return hierarchy
