from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
import glob
import pandas as pd

DATALAKE = Path('Z:')
LAB_WORLDS = Path('Y:')
PROJECT_WORLDS = Path('X:')
STUDY_IDS =  {'P': 'PerceptOCD-48392', 'AA': 'AA-56119', 'TRBD': 'TRBD-53761'}
TZ = ZoneInfo("America/Chicago")
FS = 250

def extract_consecutive_days(subpaths: list[str]) -> list[tuple[str, str]]:
    if not subpaths:
        return []

    date_format = "%Y-%m-%d"
    dates = sorted(
        datetime.strptime(path.split("\\")[-1], date_format)
        for path in subpaths
    )

    runs = []
    run_start = dates[0]
    run_end = dates[0]

    for date in dates[1:]:
        if date - run_end == timedelta(days=1):
            run_end = date  # extend the current run
        else:
            if run_start != run_end:  # only keep runs of 2+ days
                runs.append((run_start.strftime(date_format), run_end.strftime(date_format)))
            run_start = date
            run_end = date

    if run_start != run_end:  # flush the last run
        runs.append((run_start.strftime(date_format), run_end.strftime(date_format)))

    return runs

def get_events(pt, window):
    fps = glob.glob(f'{DATALAKE}/{STUDY_IDS[pt[:-3]]}/{pt}/NBU/{window[0]}/events/*.csv') + glob.glob(f'{DATALAKE}/{STUDY_IDS[pt[:-3]]}/{pt}/NBU/{window[1]}/events/*.csv')
    if len(fps) == 0:
        return pd.DataFrame()

    events_df = pd.concat([pd.read_csv(fp) for fp in fps], ignore_index=True)
    events_df.query('Event != "SESSION START" and Event != "SESSION END"', inplace=True)
    events_df['start_dt'] = pd.to_datetime(
        events_df['Start Date'].astype(str) + ' ' + events_df['Start Time'].astype(str)
    ).dt.tz_localize(TZ)
    events_df['end_dt'] = pd.to_datetime(
        events_df['End Date'].astype(str) + ' ' + events_df['End Time'].astype(str)
    ).dt.tz_localize(TZ)
    return events_df