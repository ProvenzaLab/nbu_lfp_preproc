import os
import json
import numpy as np
import pandas as pd
from scipy.stats import zscore
from scipy.signal import find_peaks
from joblib import Parallel, delayed
from percept_parser.percept import PerceptParser

from compile_timedomain import read_lfp_data, get_file_map, parse_session_file, load_drift_df, finalize_lfp_df
from utils.utils import find_non_nan_segments
from utils.config import PROJECT_WORLDS, STUDY_IDS

ALT_ECG_METHODS = ('template', 'perceive')  # ECG-cleaning methods parsed on demand for unclean hours


def load_raw(pt, window):
    """Load the raw (non-ECG-cleaned) LFP data for a patient/visit window.

    Reads the compiled parquet if it already exists, otherwise parses the raw
    session files (and caches the result to disk).

    Returns (raw_df, file_map); raw_df is None if there's no data. file_map maps
    (ipg_hemisphere, filename) -> session file path, used to re-parse an hour's
    files with the other ECG methods.
    """
    study_id = STUDY_IDS[pt[:-3]]
    fp = PROJECT_WORLDS / study_id / pt / 'NBU' / f'{window[0]}_{window[1]}' / 'LFP' / 'raw' / f'{pt}_{window[0]}-{window[1]}_timedomain_LFP.parq'
    if os.path.exists(fp):
        df = pd.read_parquet(fp)
    else:
        df = read_lfp_data(pt, window=window, ecg_method=None, save=True)

    return df, get_file_map(pt)


def group_by_hour(df):
    """Pre-slice a dataframe by hour once.

    Re-filtering the full multi-day dataframe on every Next/Back click (via
    `df['CT_times'].dt.floor('h') == hour`) is what made the wizard slow —
    that scan re-runs over the entire dataset on every click. Doing it once up
    front turns each hour transition into an O(1) dict lookup instead.

    Returns {hour: hour_df}.
    """
    return {hour: sub_df.sort_index() for hour, sub_df in df.groupby(df['CT_times'].dt.floor('h'))}


def _parse_file_for_window(path, ecg_method, time_window, stream_cache):
    """Parse one session file's packages overlapping time_window; returns (chunks, newly parsed packages)."""
    parser = PerceptParser(path, ecg_method=ecg_method)
    cached = set(stream_cache)
    chunks = parse_session_file(parser, time_window=time_window, stream_cache=stream_cache)
    return chunks, {k: v for k, v in stream_cache.items() if k not in cached}


def load_hour_methods(pt, hour, raw_hour_df, file_map, stream_cache, methods=ALT_ECG_METHODS):
    """Parse a single hour of data under the other ECG-cleaning methods.

    Only the session files that contribute to this hour of raw data are
    re-parsed, and only their streaming packages overlapping the hour are
    reconstructed and ECG-cleaned (see compile_timedomain.read_timedomain_window).

    stream_cache: {(ipg_hemisphere, filename, method): {package key: cleaned df}},
        updated in place; keep it across calls so neighboring hours reuse work.
    Returns {method: hour_df}, omitting methods with no data.
    """
    files = raw_hour_df[['ipg_hemisphere', 'filename']].drop_duplicates()
    drift_df = load_drift_df(pt, files['filename'].unique())
    margin = pd.Timedelta(minutes=1)

    tasks = []
    for hem, fname in files.itertuples(index=False):
        path = file_map.get((hem, fname))
        if path is None:
            print(f'Session file {hem}/{fname} not found, skipping.')
            continue

        # finalize_lfp_df shifts this file's timestamps back by its manual drift
        # correction, so the hour sits that much later in the parser's time.
        shift = pd.Timedelta(seconds=drift_df.loc[drift_df['filename'] == fname, 'time_drift_seconds'].sum())
        time_window = (hour + shift - margin, hour + pd.Timedelta(hours=1) + shift + margin)
        for method in methods:
            tasks.append(((hem, fname, method), path, time_window))

    if not tasks:
        return {}

    results = Parallel(n_jobs=len(tasks))(
        delayed(_parse_file_for_window)(path, key[2], time_window, stream_cache.get(key, {}))
        for key, path, time_window in tasks
    )

    chunks_by_method = {m: [] for m in methods}
    for (key, _, _), (chunks, new_streams) in zip(tasks, results):
        stream_cache.setdefault(key, {}).update(new_streams)
        chunks_by_method[key[2]].extend(chunks)

    hour_dfs = {}
    for method, chunks in chunks_by_method.items():
        if not chunks:
            continue
        df = finalize_lfp_df(pt, chunks)
        df = df[df['CT_times'].dt.floor('h') == hour].sort_index()
        # Match the raw data's columns: channels recorded elsewhere in the visit
        # but not in this hour's packages would otherwise be missing.
        df = df.reindex(columns=raw_hour_df.columns)
        if not df.empty:
            hour_dfs[method] = df
    return hour_dfs


def find_most_complete_window(df, chs, seconds=30, step_seconds=1):
    """Start time of the `seconds`-long window with the most samples where every streaming channel is valid.

    `chs` spans every sensing configuration in the visit, so some channels can
    have no data in a given hour; a sample counts as valid when it has the most
    simultaneously valid channels seen in the hour. Slides the window in
    `step_seconds` steps; ties go to the earliest window, and the start is
    snapped forward to the first valid sample.
    """
    times = df['CT_times']
    n_valid = df[chs].apply(pd.to_numeric, errors='coerce').notna().sum(axis=1).to_numpy()
    valid = (n_valid == n_valid.max()) & (n_valid > 0) if len(n_valid) else n_valid.astype(bool)
    t0 = times.min()
    rel = (times - t0).dt.total_seconds().to_numpy()

    valid_t = np.sort(rel[valid])
    if valid_t.size == 0:
        return t0

    starts = np.arange(0, max(rel.max() - seconds, 0) + step_seconds, step_seconds)
    counts = np.searchsorted(valid_t, starts + seconds) - np.searchsorted(valid_t, starts)
    best = valid_t[np.searchsorted(valid_t, starts[np.argmax(counts)])]
    return t0 + pd.Timedelta(seconds=best)


def stitch_selected_hours(raw_hourly, method_hours, selections):
    """Concatenate the user-selected data for each hour into one clean dataframe.

    raw_hourly: {hour: raw hour_df}, as returned by group_by_hour.
    method_hours: {hour: {method: hour_df}}, as returned by load_hour_methods.
    selections: {hour: {'label': 'clean'|'unclean'|'removed', 'method': 'raw'|'template'|'perceive'|None}};
        hours with method None are excluded.
    """
    chunks = []
    for hour, sel in selections.items():
        method = sel['method']
        if method is None:
            continue
        chunk = raw_hourly.get(hour) if method == 'raw' else method_hours.get(hour, {}).get(method)
        if chunk is not None and not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame()

    stitched = pd.concat(chunks).sort_index()
    stitched = stitched.drop_duplicates(subset='CT_times')
    return stitched


def detect_outlier_peaks(df, chs, fs, pad_seconds=5):
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

            # Use median absolute deviation (MAD) to detect outliers
            mad = np.nanmedian(np.abs(seg - np.nanmedian(seg)))  
            norm = (seg - np.nanmedian(seg)) / mad
            iqr = np.subtract(*np.nanpercentile(norm, [75, 25]))
            peaks, _ = find_peaks(np.abs(norm), height=20*iqr, distance=pad_samples)

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
    study_id = STUDY_IDS[pt[:-3]]
    target_folder = PROJECT_WORLDS / study_id / pt / 'NBU' / f'{window[0]}_{window[1]}' / 'LFP' / 'clean'
    os.makedirs(target_folder, exist_ok=True)

    parq_path = target_folder / 'timedomain_LFP.parq'
    meta_path = target_folder / 'clean_metadata.json'

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
                'method': o['method']
            }
            for o in removed_outliers
        ],
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return parq_path, meta_path


def save_marked_artifacts(marked_artifacts, pt, window):
    """Save user-flagged 'important' artifacts (e.g. injected sync markers) to
    'marked_artifacts.json' in the same 'clean' subfolder as the clean dataset.

    marked_artifacts: list of dicts with channel, peak_time, win_start, win_end, note.
    Returns the file path, or None if there's nothing to save.
    """
    if not marked_artifacts:
        return None

    study_id = STUDY_IDS[pt[:-3]]
    target_folder = PROJECT_WORLDS / study_id / pt / 'NBU' / f'{window[0]}_{window[1]}' / 'LFP' / 'clean'
    os.makedirs(target_folder, exist_ok=True)
    marked_path = target_folder / 'marked_artifacts.json'

    data = [
        {
            'channel': m['channel'],
            'peak_time': m['peak_time'].isoformat() if hasattr(m['peak_time'], 'isoformat') else str(m['peak_time']),
            'win_start': m['win_start'].isoformat() if hasattr(m['win_start'], 'isoformat') else str(m['win_start']),
            'win_end': m['win_end'].isoformat() if hasattr(m['win_end'], 'isoformat') else str(m['win_end']),
            'note': m['note'],
        }
        for m in marked_artifacts
    ]
    with open(marked_path, 'w') as f:
        json.dump(data, f, indent=2)

    return marked_path
