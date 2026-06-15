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
from utils.config import STUDY_IDS, TZ, DATALAKE, LAB_WORLDS, extract_consecutive_days

def read_lfp_data(pt, window=None, plot=False, ecg_method=None, save=True):
    if pt[:-3] not in STUDY_IDS.keys() or pt[-3:] == '000' or window is None:
        return
    
    # Make project worlds save directory
    study_id = STUDY_IDS[pt[:-3]]

    if save:
        ecg_label = ecg_method if ecg_method else 'raw'
        target_folder = LAB_WORLDS / 'Provenza' / 'NBUPipeline' / pt / window[0] / 'rawlfp' / ecg_label
        os.makedirs(target_folder, exist_ok=True)

    # Gather all pt LFP files that have been uploaded
    lfp_path = DATALAKE / study_id / pt / 'LFP'
    paths = glob.glob(os.path.join(f'{lfp_path}/**/*.json'), recursive=True)
    
    # Run percept parser on the LFP files to create a single dataframe
    chunks = []
    files_ = []

    for filename in paths:
        try:
            parser = PerceptParser(filename, ecg_method=ecg_method)
        
        except Exception as e:
            print(f'Error processing {filename}: {e}')
            continue

        # Check if session was within specified date window
        if window:
            session_start = parser.js['SessionDate'] # Tablet start time
            session_tz = parser.js['ProgrammerUtcOffset']

            start = parse(window[0]).replace(tzinfo=TZ)
            end = parse(window[1]).replace(tzinfo=TZ)

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

        for id in [True, False]:
            data = parser.read_timedomain_data(indefinite_streaming=id)

            if len(data) != 0:
                df = pd.concat(data)
                df['filename'] = filename.split(os.sep)[-1]
                chunks.append(df)

                if plot:
                    out_path = target_folder / 'plots'
                    os.makedirs(out_path, exist_ok=True)

                    plotter.plot_time_domain_ranges(data, out_path=out_path, lfp_filename=filename.split(os.sep)[-1])
                    for _, df_bs_td_i in tqdm(list(enumerate(data)), desc="BrainSenseTimeDomain Plot Index"):
                        str_idx = f"{df_bs_td_i.index[0].strftime("%Y-%m-%d_%H%M%S")}_{df_bs_td_i.index[-1].strftime('%H%M%S')}"
                        df_bs_td_i.to_csv(os.path.join(out_path, f"BrainSenseTimeDomain_{str_idx}.csv"), index=True)
                        plotter.plot_df_timeseries(df_bs_td_i, out_path=out_path)
                        plotter.time_frequency_plot_td(df_bs_td_i, indefinite_streaming=id, parser=parser, out_path=out_path)

    if len(chunks) == 0:
        print(f'No Time Domain data for {pt}')
        return
    
    pt_df = pd.concat(chunks)

    # Drop duplicate rows with nan LFP values
    chs = [ch for ch in pt_df.columns if 'LEFT' in ch or 'RIGHT' in ch]
    pt_df = pt_df[~(pt_df.index.duplicated(keep=False) & pt_df[chs].isna().all(axis=1))]

    # Convert timestamps to Central + create CT times column
    pt_df['CT_times'] = pt_df.index.to_series().dt.tz_convert('America/Chicago')
    
    # Manually adjust times for files with abnormal tablet end times
    drift_df = pd.read_csv('utils/time_drift_per_file_final.csv', delimiter=';')
    drift_df['filename'] = drift_df['filename'].apply(lambda f: f.split(os.sep)[-1])
    drift_df.query('pt_id == @pt and filename.isin(@pt_df.filename)', inplace=True)
    pt_df = correct_time_drift(pt_df, drift_df[drift_df['tablet_end_time'].isna()])

    # Create columns for other info
    pt_df['lead_location'] = parser.lead_location
    pt_df['lead_model'] = parser.lead_model
    pt_df['ipg_hemisphere'] = filename.split(os.sep)[-2]
    
    if save:
        target_fp = target_folder / f'{pt}_{start.strftime("%Y-%m-%d")}-{end.strftime("%Y-%m-%d")}_timedomain_LFP.parq'
        pt_df.to_parquet(target_fp)
    else:
        return pt_df
    return


if __name__ == '__main__':
    # Compiles LFP data from each NBU visit

    pt_paths = sum([glob.glob(os.path.join(DATALAKE / study_id, "*/")) for study_id in STUDY_IDS.values()], [])

    for path in pt_paths:
        pt = path.split(os.sep)[-2]
        if os.path.exists(os.path.join(path, 'NBU')):
            nbu_paths = glob.glob('NBU/*', root_dir=path)
            nbu_dates = extract_consecutive_days(nbu_paths)

            for nbu_visit in nbu_dates:
                print(f'Processing {pt} NBU visit from {nbu_visit[0]} to {nbu_visit[1]}')
                for method in [None, 'template', 'perceive']:
                    print(f'ECG method: {method}')
                    """ if os.path.exists(LAB_WORLDS / 'Provenza' / 'NBUPipeline' / pt / nbu_visit[0] / 'rawlfp' / method if method else 'raw' / f'{pt}_{nbu_visit[0]}-{nbu_visit[1]}_timedomain_LFP.parq'):
                        print('Already processed')
                        continue """
                    read_lfp_data(pt, window=nbu_visit, plot=False, ecg_method=method)