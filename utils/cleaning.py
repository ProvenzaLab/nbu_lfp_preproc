import os
import json
import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.signal import find_peaks
from joblib import Parallel, delayed

from compile_timedomain import read_lfp_data
from utils.utils import find_non_nan_segments
from utils.config import LAB_WORLDS

ECG_METHODS = {None: 'raw', 'template': 'template', 'perceive': 'perceive'}


def load_all_ecg_methods(pt, window):
    """Load LFP data for a patient/visit window under all three ECG-cleaning methods.

    Returns {'raw': df, 'template': df, 'perceive': df}, omitting methods with no data.
    Methods whose compiled parquet already exists are read straight from disk instead
    of being re-parsed from the raw session files.
    """
    target_folder = LAB_WORLDS / 'Provenza' / 'NBUPipeline' / pt / window[0] / 'rawlfp'
    dfs = {}
    missing_methods = []

    for method in ECG_METHODS:
        fp = target_folder / ECG_METHODS[method] / f'{pt}_{window[0]}-{window[1]}_timedomain_LFP.parq'
        if os.path.exists(fp):
            dfs[ECG_METHODS[method]] = pd.read_parquet(fp)
        else:
            missing_methods.append(method)

    if missing_methods:
        results = Parallel(n_jobs=len(missing_methods))(
            delayed(read_lfp_data)(pt, window=window, ecg_method=m, save=True) for m in missing_methods
        )
        for method, df in zip(missing_methods, results):
            label = ECG_METHODS[method]
            if df is not None and not df.empty:
                dfs[label] = df

    return dfs


def group_by_hour(dfs):
    """Pre-slice each method's dataframe by hour once.

    Re-filtering the full multi-day dataframe on every Next/Back click (via
    `df['CT_times'].dt.floor('h') == hour`) is what made the wizard slow —
    that scan re-runs over the entire dataset, for every method, on every
    click. Doing it once up front turns each hour transition into an O(1)
    dict lookup instead.

    Returns {method: {hour: hour_df}}.
    """
    grouped = {}
    for method, df in dfs.items():
        if df is None or df.empty:
            continue
        grouped[method] = {hour: sub_df for hour, sub_df in df.groupby(df['CT_times'].dt.floor('h'))}
    return grouped


def get_available_hours(hourly):
    hours = set()
    for hour_dict in hourly.values():
        hours.update(hour_dict.keys())
    return sorted(hours)


def stitch_selected_hours(hourly, selections):
    """Concatenate the user-selected method's data for each hour into one clean dataframe.

    hourly: {method: {hour: hour_df}}, as returned by group_by_hour.
    selections: dict mapping hour (Timestamp) -> method label (e.g. 'raw', 'template', 'perceive')
    """
    chunks = []
    for hour, method in selections.items():
        chunk = hourly.get(method, {}).get(hour)
        if chunk is not None and not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame()

    stitched = pd.concat(chunks).sort_index()
    stitched = stitched.drop_duplicates(subset='CT_times')
    return stitched


def detect_outlier_peaks(df, chs, fs, sd_threshold=10, pad_seconds=5):
    """Find samples exceeding sd_threshold standard deviations (per channel, per non-NaN segment).

    Returns a list of candidate dicts: {channel, peak_time, win_start, win_end}.
    """
    pad_samples = int(pad_seconds * fs)
    candidates = []

    for ch in chs:
        values = df[ch].to_numpy(dtype=float)
        segments = find_non_nan_segments(values)

        for start, end in segments:
            seg = values[start:end]
            if len(seg) < 3:
                continue

            z = zscore(seg, nan_policy='omit')
            peaks, _ = find_peaks(np.abs(z), height=sd_threshold, distance=pad_samples)

            for p in peaks:
                abs_idx = start + p
                win_start_idx = max(start, abs_idx - pad_samples)
                win_end_idx = min(end - 1, abs_idx + pad_samples)

                candidates.append({
                    'channel': ch,
                    'peak_time': df.index[abs_idx],
                    'win_start': df.index[win_start_idx],
                    'win_end': df.index[win_end_idx],
                })

    return candidates


def remove_outlier_window(df, channel, win_start, win_end):
    df.loc[win_start:win_end, channel] = np.nan
    return df


def extract_template(df, channel, peak_time, fs, template_seconds=1.0):
    """Extract a fixed-length, z-scored window centered on a candidate peak.

    Used both to save a new artifact template (on manual "Remove") and to score
    a not-yet-reviewed candidate against saved templates. Must be called before
    the window is NaN'd out by remove_outlier_window.

    Returns a 1D np.ndarray, or None if a clean fixed-length window isn't
    available (e.g. peak too close to the data edge or a NaN gap, or a
    zero-variance window).
    """
    pos = df.index.get_loc(peak_time)
    if isinstance(pos, (slice, np.ndarray)):
        pos = np.atleast_1d(np.arange(len(df))[pos])[0]

    half = int(template_seconds / 2 * fs)
    start, end = pos - half, pos + half + 1
    if start < 0 or end > len(df):
        return None

    window = pd.to_numeric(df[channel].iloc[start:end], errors='coerce').to_numpy()
    if np.isnan(window).any():
        return None

    std = window.std()
    if std == 0:
        return None

    return (window - window.mean()) / std


def match_template(window, templates, max_lag=20):
    """Best normalized cross-correlation between a candidate window and any saved template.

    Searches lags in [-max_lag, max_lag] samples to tolerate peak-alignment jitter.
    Returns 0.0 if there are no templates or no template matches the window length.
    """
    if window is None or not templates:
        return 0.0

    n = len(window)
    best = 0.0
    for template in templates:
        if len(template) != n:
            continue
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                a, b = window[lag:], template[:n - lag]
            else:
                a, b = window[:n + lag], template[-lag:]

            if len(a) < n // 2:
                continue

            a_c, b_c = a - a.mean(), b - b.mean()
            denom = np.linalg.norm(a_c) * np.linalg.norm(b_c)
            if denom == 0:
                continue

            best = max(best, float(np.dot(a_c, b_c) / denom))

    return best


def save_clean_dataset(df, pt, window, hour_selections, removed_outliers):
    """Save the stitched, artifact-cleaned dataset + selection metadata to the 'clean' subfolder."""
    target_folder = LAB_WORLDS / 'Provenza' / 'NBUPipeline' / pt / window[0] / 'rawlfp' / 'clean'
    os.makedirs(target_folder, exist_ok=True)

    base_name = f'{pt}_{window[0]}-{window[1]}'
    parq_path = target_folder / f'{base_name}_timedomain_LFP.parq'
    meta_path = target_folder / f'{base_name}_clean_metadata.json'

    df.to_parquet(parq_path)

    metadata = {
        'patient': pt,
        'window': list(window),
        'hour_selections': {
            (hour.isoformat() if hasattr(hour, 'isoformat') else str(hour)): method
            for hour, method in hour_selections.items()
        },
        'removed_outliers': [
            {
                'channel': o['channel'],
                'peak_time': o['peak_time'].isoformat(),
                'win_start': o['win_start'].isoformat(),
                'win_end': o['win_end'].isoformat(),
            }
            for o in removed_outliers
        ],
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return parq_path, meta_path
