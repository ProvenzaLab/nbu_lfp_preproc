import numpy as np
import pandas as pd
from datetime import datetime, timedelta

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

def correct_time_drift(lfp_df, drift_df):
    # Manually adjust time drift for files with abnormal tablet end times

    for row in drift_df.itertuples():
        filename = row.filename

        if filename not in lfp_df.filename.unique():
            continue
        
        drift = row.time_drift_seconds
        mask = lfp_df['filename'] == filename

        lfp_df.index = lfp_df.index.where(~mask, lfp_df.index - timedelta(seconds=drift))
    
    return lfp_df



