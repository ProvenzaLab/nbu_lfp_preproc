import pandas as pd
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np
from pickle import load
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from pathlib import Path

def get_oura_scores(oura_scores, date_first, date_last, subject):
    date_range = pd.date_range(start=date_first, end=date_last, freq="D")
    bpm_list = []
    met_list = []
    sleep_list = []
    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")
        key_oura = f"{subject}_{date_str}"
        if key_oura in oura_scores.keys():
            df_oura = oura_scores[key_oura]
            if "bpm" in df_oura:
                bpm_series = df_oura["bpm"][["bpm", "timestamp"]]
                # set timestamp as index and convert to datetime
                bpm_series.set_index("timestamp", inplace=True)
                # resample every 10 minutes and take the mean
                # set index to datetime
                bpm_series.index = pd.to_datetime(bpm_series.index)
                bpm_list.append(bpm_series)
        
            if "mets" in df_oura:
                met_series = df_oura["mets"][["met"]]
                met_list.append(met_series)

            if "sleep_phases" in df_oura:
                sleep_phases_series = df_oura["sleep_phases"]
                # 1 - deep sleep, 2 - light sleep, 3 - rem sleep, 4 - awake
                # get a series with only deep sleep and light sleep
                #nremleep = sleep_phases_series.isin([2])
                sleep_list.append(sleep_phases_series)
    return pd.concat(bpm_list), pd.concat(met_list), pd.concat(sleep_list)

#path_ = "/mnt/labworlds/Provenza/NBUPipeline/TRBD001/2026-02-19/rawlfp/raw/spectra_fooof_bandpower_ONE_THREE_LEFT.pkl"

oura = pd.read_pickle("oura_scores.pkl")

base_dir = Path("/mnt/labworlds/Provenza/NBUPipeline/TRBD001")

pkl_files = sorted(base_dir.glob("*/rawlfp/raw/*.pkl"))
pdf_ = PdfPages("9Hzpower.pdf")

path_sleep_times_TRBD001 = "trbd001_nbu_sleep_onset.pkl"
with open(path_sleep_times_TRBD001, "rb") as f:
    sleep_times_TRBD001 = load(f)
# nbu_visit is currently object like "2025-05-05", transform to datetime
sleep_times_TRBD001["nbu_visit"] = pd.to_datetime(sleep_times_TRBD001["nbu_visit"])
sleep_dates = pd.Series(sleep_times_TRBD001["nbu_visit"].dt.date.unique())

for pkl_file in pkl_files:

    with open(pkl_file, "rb") as f:
        data = load(f)

    df_features = data["df_features"]
    # sort by index
    df_features = df_features.sort_index()
    # change index timezone from utc to local time
    df_features.index = df_features.index.tz_convert("America/Chicago")

    active_sensing_band = df_features["active_sensing_band"]
    offset = df_features["offset"]
    theta = df_features["theta"] # range: 4-8 Hz
    alpha = df_features["alpha"] # range: 8-12 Hz

    spectrum = data["df_spec"].iloc[:, 1:31]

    first_date = df_features.index[0]
    last_date = df_features.index[-1]

    bpm_oura, met_oura, sleep_oura = get_oura_scores(oura, first_date, last_date, "TRBD001")
    
    # compute the avg for every 10 min
    active_sensing_band_10min = active_sensing_band.resample("10min").mean()
    unique_dates = pd.Series(pd.Series(df_features.index.date).unique())
    has_sleep_interval = False
    # check if any of the unique dates are in the sleep times
    if any(date in sleep_dates.values for date in unique_dates):
        date_sleep = unique_dates[unique_dates.isin(sleep_dates.values)][0]
        sleep_onset = sleep_times_TRBD001[sleep_times_TRBD001["nbu_visit"].dt.date == date_sleep]["sleep_start"].iloc[0]
        sleep_offset = sleep_times_TRBD001[sleep_times_TRBD001["nbu_visit"].dt.date == date_sleep]["sleep_end"].iloc[0]

        def _sleep_time_to_timestamp(value):
            if isinstance(value, pd.Timestamp):
                ts = value
            elif isinstance(value, (int, float, np.number)):
                ts = pd.Timestamp(date_sleep) + pd.Timedelta(hours=float(value))
            elif hasattr(value, "hour") and hasattr(value, "minute"):
                ts = pd.Timestamp(datetime.combine(date_sleep, value))
            else:
                ts = pd.to_datetime(value)

            if ts.tzinfo is None:
                ts = ts.tz_localize("America/Chicago")
            else:
                ts = ts.tz_convert("America/Chicago")
            return ts

        sleep_start_ts = _sleep_time_to_timestamp(sleep_onset)
        sleep_end_ts = _sleep_time_to_timestamp(sleep_offset)
        if sleep_end_ts < sleep_start_ts:
            sleep_end_ts = sleep_end_ts + pd.Timedelta(days=1)
        has_sleep_interval = True

    # make a scatter plot of active sensing band over time
    plt.figure(figsize=(18, 6))
    ax1 = plt.subplot(1, 4, 1)
    plt.scatter(df_features.index, active_sensing_band, s=10, color="blue", alpha=0.2)
    plt.plot(active_sensing_band_10min.index, active_sensing_band_10min, color="red", linewidth=2)
    if has_sleep_interval:
        ax1.axvspan(sleep_start_ts, sleep_end_ts, color="green", alpha=0.2)
    plt.title("Active Sensing Band Power Over Time")
    plt.xlabel("Time")
    plt.ylabel("Active Sensing Band Power")
    plt.tight_layout()

    plt.xticks(rotation=90)
    plt.subplot(1, 4, 2)
    plt.hist(active_sensing_band, bins=200, color="blue", alpha=0.7)
    plt.title("Distribution of Active Sensing Band Power")
    plt.xlabel("Active Sensing Band Power")
    plt.ylabel("Frequency")
    date = pkl_file.parent.parent.parent.name
    plt.suptitle(f"{date} = {pkl_file.name}")

    ax3 = plt.subplot(1, 4, 3)
    plt.scatter(df_features.index, active_sensing_band, s=10, color="blue", alpha=0.2)
    # make log y axis
    plt.yscale("log")
    plt.plot(active_sensing_band_10min.index, active_sensing_band_10min, color="red", linewidth=2)
    if has_sleep_interval:
        ax3.axvspan(sleep_start_ts, sleep_end_ts, color="green", alpha=0.2)
    plt.title("Active Sensing Band Power Over Time")
    plt.xlabel("Time")
    plt.ylabel("Active Sensing Band Power")
    plt.xticks(rotation=90)
    plt.subplot(1, 4, 4)
    plt.hist(active_sensing_band, bins=200, color="blue", alpha=0.7)
    
    # make log x axis
    plt.xscale("log")
    plt.title("Distribution of Active Sensing Band Power")
    plt.xlabel("Active Sensing Band Power")
    plt.ylabel("Frequency")
    date = pkl_file.parent.parent.parent.name
    plt.suptitle(f"{date} = {pkl_file.name}")
    plt.tight_layout()
    pdf_.savefig(plt.gcf())

pdf_.close()
