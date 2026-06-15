from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta

DATALAKE = Path('Z:')
LAB_WORLDS = Path('Y:')
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