import os
import glob
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
from pathlib import Path
from dateutil.parser import parse, isoparse
from dateutil.tz import tzoffset
import warnings
warnings.filterwarnings('ignore')

from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
from percept_parser.percept import PerceptParser
from percept_parser import plotter
from utils.utils import correct_time_drift
from utils.config import STUDY_IDS, TZ, DATALAKE, LAB_WORLDS, PROJECT_WORLDS, extract_consecutive_days


def get_session_paths(pt):
    """All Percept .json session files uploaded for a patient."""
    lfp_path = DATALAKE / STUDY_IDS[pt[:-3]] / pt / 'LFP'
    return glob.glob(os.path.join(f'{lfp_path}/**/*.json'), recursive=True)


def get_file_map(pt):
    """Map (ipg_hemisphere, filename) -> full path for every session file of a patient.

    Keys match the `ipg_hemisphere` / `filename` columns written by
    parse_session_file, so rows of a compiled dataframe can be traced back to
    the session file they came from without re-parsing anything.
    """
    return {(p.split(os.sep)[-2], p.split(os.sep)[-1]): p for p in get_session_paths(pt)}


def _package_may_overlap(streams, first_packet, time_shift, time_window):
    """Whether a streaming package (all channel streams sharing a FirstPacketDateTime)
    can overlap time_window, bounded from the JSON without reconstructing sample times.

    A package starts at FirstPacketDateTime (minus the parser's time shift) and spans
    at most the range of its packet ticks plus the duration of all its samples.
    """
    start = pd.Timestamp(first_packet) - time_shift
    span_ms = 0.0
    for stream in streams:
        ticks = np.fromstring(stream['TicksInMses'], sep=',', dtype=int)
        if len(ticks) == 0:
            return True
        n_samples = len(stream['TimeDomainData'])
        span_ms = max(span_ms, ticks.max() - ticks.min() + n_samples * 1000 / stream['SampleRateInHz'])
    return start <= time_window[1] and start + pd.Timedelta(milliseconds=span_ms) >= time_window[0]


def read_timedomain_window(parser, indefinite_streaming, time_window, stream_cache):
    """PerceptParser.read_timedomain_data, restricted to the packages overlapping time_window.

    Mirrors perceptparser's implementation (same num_chs over the whole file, same
    per-package ECG cleaning) but skips reconstructing sample times for packages that
    can't overlap the window, and only ECG-cleans packages that do. That's the bulk of
    the parse time for a session file spanning a whole day.

    stream_cache: {(indefinite_streaming, FirstPacketDateTime): cleaned df}; read and
        updated in place so packages spanning several hours are only cleaned once.
    """
    str_timedomain = "IndefiniteStreaming" if indefinite_streaming else "BrainSenseTimeDomain"
    if str_timedomain not in parser.js:
        return []

    td_data = parser.js[str_timedomain]
    first_package_times = np.array([stream["FirstPacketDateTime"] for stream in td_data])
    num_chs = len(np.unique([stream["Channel"] for stream in td_data]))
    time_shift = parser._safe_time_delta()

    dfs = []
    for first_package in map(str, np.unique(first_package_times)):
        idx_package_chs = np.where(first_package_times == first_package)[0]
        streams = [td_data[i] for i in idx_package_chs]
        if not _package_may_overlap(streams, first_package, time_shift, time_window):
            continue

        cache_key = (indefinite_streaming, first_package)
        if cache_key in stream_cache:
            dfs.append(stream_cache[cache_key])
            continue

        df_chs = []
        for stream in streams:
            try:
                df_ch, _, _ = parser.get_time_stream(js_td=stream, num_chs=num_chs)
            except (UnboundLocalError, ValueError) as e:
                print(e)
                continue
            df_chs.append(df_ch)

        if len(df_chs) == 0:
            continue

        df = pd.concat(df_chs, axis=0).reset_index().pivot(index="Time", columns="Channel", values="Data")
        if df.index.max() < time_window[0] or df.index.min() >= time_window[1]:
            continue

        if parser.ecg_remover is not None:
            try:
                df = parser.ecg_remover.clean(df, streams[0]["SampleRateInHz"])
            except Exception as e:
                print(f"Error applying ECG removal: {e}")

        stream_cache[cache_key] = df
        dfs.append(df)
    return dfs


def parse_session_file(parser, time_window=None, stream_cache=None, plot_dir=None):
    """Read the time-domain data from one parsed session file and merge in its stim settings.

    parser: a PerceptParser for the file (its ecg_method determines the ECG cleaning).
    time_window: optional (start, end) in the parser's (drift-corrected) device time;
        only packages overlapping it are parsed, and they're trimmed to [start, end)
        before the stim-settings merge.
    stream_cache: optional cache of parsed packages, see read_timedomain_window.
    Returns a list of dataframes, one per streaming mode with data.
    """
    filename = parser.filename
    chunks = []

    for id in [True, False]:
        if time_window is None:
            data = parser.read_timedomain_data(indefinite_streaming=id)
        else:
            data = read_timedomain_window(parser, id, time_window, stream_cache if stream_cache is not None else {})
        if len(data) == 0:
            continue

        df = pd.concat(data)
        if time_window is not None:
            df = df[(df.index >= time_window[0]) & (df.index < time_window[1])]
            if df.empty:
                continue
        df['filename'] = filename.split(os.sep)[-1]

        # Create columns for other info
        df['lead_location'] = parser.lead_location
        df['lead_model'] = parser.lead_model
        df['ipg_hemisphere'] = filename.split(os.sep)[-2]

        # Read stimulation settings
        setting_cols = ['frequency', 'stim_cathode', 'stim_anode', 'sensing_contacts', 'pulse_width', 'suspend_amplitude', 'lower_amplitude', 'upper_amplitude']
        index_cols = ['start_time', 'end_time']

        stim_settings = parser.stim_settings.active_group_history.drop_duplicates(subset=[*index_cols, 'hem', 'filename']).pivot(
            index=index_cols, columns='hem', values=setting_cols)

        stim_settings.columns = [f'{"left" if h == "Left" else "right"}_{p}' for p, h in stim_settings.columns]
        stim_settings = stim_settings.reset_index().sort_values('start_time')
        # merge_asof needs matching datetime resolutions (streams are ns, stim settings can be us)
        stim_settings['start_time'] = stim_settings['start_time'].astype(df.index.dtype)
        df.sort_index(inplace=True)
        merged_df = pd.merge_asof(df, stim_settings, left_on='Time', right_on='start_time', direction='backward')
        merged_df.index = df.index
        chunks.append(merged_df)

        if plot_dir is not None:
            os.makedirs(plot_dir, exist_ok=True)

            plotter.plot_time_domain_ranges(data, out_path=plot_dir, lfp_filename=filename.split(os.sep)[-1])
            for _, df_bs_td_i in tqdm(list(enumerate(data)), desc="BrainSenseTimeDomain Plot Index"):
                str_idx = f"{df_bs_td_i.index[0].strftime("%Y-%m-%d_%H%M%S")}_{df_bs_td_i.index[-1].strftime('%H%M%S')}"
                df_bs_td_i.to_csv(os.path.join(plot_dir, f"BrainSenseTimeDomain_{str_idx}.csv"), index=True)
                plotter.plot_df_timeseries(df_bs_td_i, out_path=plot_dir)
                plotter.time_frequency_plot_td(df_bs_td_i, indefinite_streaming=id, parser=parser, out_path=plot_dir)

    return chunks


def load_drift_df(pt, filenames):
    """Manual time-drift corrections for files with abnormal (missing) tablet end times."""
    drift_df = pd.read_csv(f'{PROJECT_WORLDS}/{STUDY_IDS[pt[:-3]]}/{pt}/{pt}_time_drift.csv', delimiter=',')
    drift_df['filename'] = drift_df['filename'].apply(lambda f: f.split(os.sep)[-1])
    drift_df.query('pt_id == @pt and filename.isin(@filenames)', inplace=True)
    return drift_df[drift_df['tablet_end_time'].isna()]


def finalize_lfp_df(pt, chunks):
    """Concatenate parsed chunks, drop duplicate NaN rows, correct time drift, and add CT times."""
    pt_df = pd.concat(chunks)

    # Drop duplicate rows with nan LFP values
    chs = [ch for ch in pt_df.columns if 'LEFT' in ch or 'RIGHT' in ch]
    pt_df = pt_df[~(pt_df.index.duplicated(keep=False) & pt_df[chs].isna().all(axis=1))]

    # Manually adjust times for files with abnormal tablet end times
    pt_df = correct_time_drift(pt_df, load_drift_df(pt, pt_df.filename.unique()))

    # Convert timestamps to Central + create CT times column
    pt_df['CT_times'] = pt_df.index.to_series().dt.tz_convert('America/Chicago')
    return pt_df


def read_lfp_data(pt, window=None, plot=False, ecg_method=None, save=True):
    if pt[:-3] not in STUDY_IDS.keys() or pt[-3:] == '000' or window is None:
        return

    study_id = STUDY_IDS[pt[:-3]]
    ecg_label = ecg_method if ecg_method else 'raw'
    target_folder = PROJECT_WORLDS / study_id / pt / 'NBU' / f'{window[0]}_{window[1]}' / 'LFP' / ecg_label

    start = parse(window[0]).replace(tzinfo=TZ)
    end = parse(window[1]).replace(tzinfo=TZ) + timedelta(hours=12)

    # Run percept parser on the LFP files to create a single dataframe
    chunks = []
    files_ = []

    for filename in get_session_paths(pt):
        try:
            parser = PerceptParser(filename, ecg_method=ecg_method)

        except Exception as e:
            print(f'Error processing {filename}: {e}')
            continue

        # Check if session was within specified date window
        session_start = parser.js['SessionDate'] # Tablet start time
        session_tz = parser.js['ProgrammerUtcOffset']

        session_utc = isoparse(session_start)
        tz = isoparse(f"2000-01-01T00:00:00{session_tz}").tzinfo
        session_ts = session_utc.astimezone(tz)

        if session_ts < start or session_ts > end:
            continue

        # Check if file is a duplicate
        ipg_start = parser.js['DeviceInformation']['Initial']['DeviceDateTime']
        ipg_serial = parser.js['DeviceInformation']['Initial']['NeurostimulatorSerialNumber']

        file_meta = (session_start, ipg_start, ipg_serial)
        if file_meta in files_:
            continue # Skip file with duplicate tablet, ipg start times
        else:
            files_.append(file_meta)

        # Read Timedomain data from file
        chunks.extend(parse_session_file(parser, plot_dir=target_folder / 'plots' if plot else None))

    if len(chunks) == 0:
        print(f'No Time Domain data for {pt}')
        return None

    pt_df = finalize_lfp_df(pt, chunks)

    if save:
        # Make project worlds save directory
        os.makedirs(target_folder, exist_ok=True)

        target_fp = target_folder / f'{pt}_{start.strftime("%Y-%m-%d")}-{end.strftime("%Y-%m-%d")}_timedomain_LFP.parq'
        pt_df.to_parquet(target_fp)

    return pt_df


if __name__ == '__main__':
    # Compiles raw (non-ECG-cleaned) LFP data from each NBU visit. The template/perceive
    # ECG-cleaned versions are parsed on demand, per hour, by the preprocessing app.

    pt_paths = sum([glob.glob(os.path.join(DATALAKE / study_id, "*/")) for study_id in STUDY_IDS.values()], [])

    for path in pt_paths:
        pt = path.split(os.sep)[-2]
        if pt != 'TRBD001':
            continue

        if os.path.exists(os.path.join(path, 'NBU')):
            nbu_paths = glob.glob('NBU/*', root_dir=path)
            nbu_dates = extract_consecutive_days(nbu_paths)
            for nbu_visit in nbu_dates:
                print(f'Processing {pt} NBU visit from {nbu_visit[0]} to {nbu_visit[1]}')
                study_id = STUDY_IDS[pt[:-3]]
                if os.path.exists(PROJECT_WORLDS / study_id / pt / 'NBU' / f'{nbu_visit[0]}_{nbu_visit[1]}' / 'LFP' / 'raw' / f'{pt}_{nbu_visit[0]}-{nbu_visit[1]}_timedomain_LFP.parq'):
                    print('Already processed')
                    continue
                read_lfp_data(pt, window=nbu_visit, plot=False, ecg_method=None, save=True)
