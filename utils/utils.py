import numpy as np
import pandas as pd

def find_non_nan_segments(arr):
    """
    Find all contiguous segments in a numpy array that contain no NaN values.
    
    Returns a list of (start, end) index tuples (end is exclusive).
    """
    is_valid = ~np.isnan(arr)
    
    # Pad with False on both ends to detect edges at boundaries
    padded = np.concatenate(([False], is_valid, [False]))
    diff = np.diff(padded.astype(int))
    
    starts = np.where(diff == 1)[0]   # Rising edges  → segment start
    ends   = np.where(diff == -1)[0]  # Falling edges → segment end
    
    return list(zip(starts, ends))

