import pandas as pd
from scipy import signal
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from pickle import load
from datetime import datetime
from pathlib import Path


LOCAL_TZ = "America/Chicago"


def _concat_or_empty(items, as_series=False, name=None):
    if not items:
        if as_series:
            return pd.Series(dtype=float, name=name)
        return pd.DataFrame()
    out = pd.concat(items)
    return out.sort_index()


def _ensure_datetime_index(obj, tz=LOCAL_TZ):
    if obj is None or len(obj) == 0:
        return obj
    out = obj.copy()
    idx = pd.to_datetime(out.index)
    if idx.tz is None:
        idx = idx.tz_localize(tz)
    else:
        idx = idx.tz_convert(tz)
    out.index = idx
    return out.sort_index()


def _sleep_time_to_timestamp(value, date_sleep, tz=LOCAL_TZ):
    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, (int, float, np.number)):
        ts = pd.Timestamp(date_sleep) + pd.Timedelta(hours=float(value))
    elif hasattr(value, "hour") and hasattr(value, "minute"):
        ts = pd.Timestamp(datetime.combine(date_sleep, value))
    else:
        ts = pd.to_datetime(value)

    if ts.tzinfo is None:
        ts = ts.tz_localize(tz)
    else:
        ts = ts.tz_convert(tz)
    return ts


def _centers_to_edges(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values
    if values.size == 1:
        step = 1.0
        return np.array([values[0] - step / 2.0, values[0] + step / 2.0], dtype=float)
    deltas = np.diff(values)
    left_edge = values[0] - deltas[0] / 2.0
    right_edge = values[-1] + deltas[-1] / 2.0
    mids = values[:-1] + deltas / 2.0
    return np.concatenate(([left_edge], mids, [right_edge]))


def _plot_measure(ax, series, title, ylabel, color, has_sleep_interval=False, sleep_start_ts=None, sleep_end_ts=None):
    series = pd.Series(series).dropna().sort_index()
    if len(series) == 0:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return

    if has_sleep_interval:
        ax.axvspan(sleep_start_ts, sleep_end_ts, color="green", alpha=0.2)

    avg_10min = series.resample("10min").mean().dropna()
    ax.scatter(series.index, series.values, s=10, color=color, alpha=0.3)
    ax.plot(avg_10min.index, avg_10min.values, color="red", linewidth=3)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def get_oura_scores(oura_scores, date_first, date_last, subject):
    # get list of unique days inluding last day and first day
    date_range = pd.DatetimeIndex(
        list(pd.date_range(start=date_first, end=date_last, freq="D"))
        + [pd.Timestamp(date_first), pd.Timestamp(date_last)]
    ).unique().sort_values()

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
                sleep_list.append(sleep_phases_series)

    bpm_out = _concat_or_empty(bpm_list)
    met_out = _concat_or_empty(met_list)
    sleep_out = _concat_or_empty(sleep_list, as_series=True, name="sleep_stage")
    if isinstance(sleep_out, pd.DataFrame):
        sleep_out = sleep_out.iloc[:, 0]
    sleep_out.name = "sleep_stage"
    return bpm_out, met_out, sleep_out

#path_ = "/mnt/labworlds/Provenza/NBUPipeline/TRBD001/2026-02-19/rawlfp/raw/spectra_fooof_bandpower_ONE_THREE_LEFT.pkl"

oura = pd.read_pickle("oura_scores.pkl")

base_dir = Path("nbu_concat")

pkl_files = sorted(base_dir.glob("spectra_fooof_bandpower_*.pkl"))
output_dir = Path("out_figures_concat")
output_dir.mkdir(parents=True, exist_ok=True)

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
    spectrum = data["df_spec"].iloc[:, 1:31].copy()

    # sort by index
    df_features = df_features.sort_index()
    spectrum = spectrum.sort_index()
    spectrum = spectrum.tz_convert("America/Chicago")
    df_features.index = df_features.index.tz_convert("America/Chicago")

    # Resample to 10-second bins and preserve NaN in empty bins.
    df_features = df_features.resample("10s").mean()
    spectrum = spectrum.resample("10s").mean()

    # Drop duplicated timestamps before plotting/resampling consumers.
    df_features = df_features[~df_features.index.duplicated(keep="first")]
    spectrum = spectrum[~spectrum.index.duplicated(keep="first")]

    active_sensing_band = df_features["active_sensing_band"]
    
    offset = df_features["offset"]
    theta = df_features["theta"]  # range: 4-8 Hz
    alpha = df_features["alpha"]  # range: 8-12 Hz

    
    #spectrum.index = pd.to_datetime(spectrum.index)
    # change the timezone to America/Chicago
    #spectrum.index = spectrum.index.tz_convert("America/Chicago")

    spectrum.columns = pd.to_numeric(spectrum.columns, errors="coerce")
    spectrum = spectrum.loc[:, ~spectrum.columns.isna()]
    spectrum = spectrum.reindex(sorted(spectrum.columns), axis=1)

    spectrum_valid = spectrum.notna().any(axis=1)
    if spectrum_valid.any():
        first_spec_ts = spectrum.index[spectrum_valid][0]
        last_spec_ts = spectrum.index[spectrum_valid][-1]
    else:
        first_spec_ts = spectrum.index[0]
        last_spec_ts = spectrum.index[-1]
    
    first_date = first_spec_ts
    last_date = last_spec_ts

    bpm_oura, met_oura, sleep_oura = get_oura_scores(oura, first_date, last_date, "TRBD001")
    bpm_oura = _ensure_datetime_index(bpm_oura)
    met_oura = _ensure_datetime_index(met_oura)
    sleep_oura = _ensure_datetime_index(sleep_oura)
    bpm_oura = bpm_oura.tz_convert("America/Chicago")
    met_oura = met_oura.tz_convert("America/Chicago")
    sleep_oura = sleep_oura.tz_convert("America/Chicago")

    met_oura_series = pd.Series(dtype=float)
    if isinstance(met_oura, pd.DataFrame) and "met" in met_oura.columns:
        met_oura_series = met_oura["met"]

    sleep_oura_series = pd.to_numeric(pd.Series(sleep_oura), errors="coerce")

    unique_dates = pd.Series(pd.Series(df_features.index.date).unique())
    has_sleep_interval = False
    # check if any of the unique dates are in the sleep times
    if any(date in sleep_dates.values for date in unique_dates):
        date_sleep = unique_dates[unique_dates.isin(sleep_dates.values)][0]
        sleep_onset = sleep_times_TRBD001[sleep_times_TRBD001["nbu_visit"].dt.date == date_sleep]["sleep_start"].iloc[0]
        sleep_offset = sleep_times_TRBD001[sleep_times_TRBD001["nbu_visit"].dt.date == date_sleep]["sleep_end"].iloc[0]

        sleep_start_ts = _sleep_time_to_timestamp(sleep_onset, date_sleep)
        sleep_end_ts = _sleep_time_to_timestamp(sleep_offset, date_sleep)
        if sleep_end_ts < sleep_start_ts:
            sleep_end_ts = sleep_end_ts + pd.Timedelta(days=1)
        has_sleep_interval = True

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(18, 22),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1, 1, 1, 1, 1, 1]},
    )

    ax_spec = axes[0]
    if not spectrum.empty:
        x_num = mdates.date2num(spectrum.index.to_pydatetime())
        y_freq = spectrum.columns.to_numpy(dtype=float)
        #x_edges = _centers_to_edges(x_num)
        #y_edges = _centers_to_edges(y_freq)
        z_power = np.log(spectrum.to_numpy(dtype=float).T)
        ax_spec.pcolormesh(
            x_num,
            y_freq,
            z_power,
            shading="gouraud",
            cmap="viridis",
        )
        ax_spec.xaxis_date()
    if has_sleep_interval:
        ax_spec.axvspan(sleep_start_ts, sleep_end_ts, color="green", alpha=0.2)
    #ax_spec.set_title("Spectrogram (Datetime Index)")
    ax_spec.set_ylabel("Frequency (Hz)")

    _plot_measure(
        axes[1],
        active_sensing_band,
        "Active Sensing Band Power",
        "Band Power",
        color="blue",
        has_sleep_interval=has_sleep_interval,
        sleep_start_ts=sleep_start_ts if has_sleep_interval else None,
        sleep_end_ts=sleep_end_ts if has_sleep_interval else None,
    )

    _plot_measure(
        axes[2],
        offset,
        "Offset",
        "Offset",
        color="tab:orange",
        has_sleep_interval=has_sleep_interval,
        sleep_start_ts=sleep_start_ts if has_sleep_interval else None,
        sleep_end_ts=sleep_end_ts if has_sleep_interval else None,
    )

    _plot_measure(
        axes[3],
        theta,
        "Theta (4-8 Hz)",
        "Theta",
        color="tab:purple",
        has_sleep_interval=has_sleep_interval,
        sleep_start_ts=sleep_start_ts if has_sleep_interval else None,
        sleep_end_ts=sleep_end_ts if has_sleep_interval else None,
    )

    _plot_measure(
        axes[4],
        alpha,
        "Alpha (8-12 Hz)",
        "Alpha",
        color="tab:cyan",
        has_sleep_interval=has_sleep_interval,
        sleep_start_ts=sleep_start_ts if has_sleep_interval else None,
        sleep_end_ts=sleep_end_ts if has_sleep_interval else None,
    )

    _plot_measure(
        axes[5],
        met_oura_series,
        "Oura MET",
        "MET",
        color="tab:brown",
        has_sleep_interval=has_sleep_interval,
        sleep_start_ts=sleep_start_ts if has_sleep_interval else None,
        sleep_end_ts=sleep_end_ts if has_sleep_interval else None,
    )

    _plot_measure(
        axes[6],
        sleep_oura_series,
        "Oura Sleep Stages (1=Deep, 2=Light, 3=REM, 4=Awake)",
        "Stage",
        color="tab:green",
        has_sleep_interval=has_sleep_interval,
        sleep_start_ts=sleep_start_ts if has_sleep_interval else None,
        sleep_end_ts=sleep_end_ts if has_sleep_interval else None,
    )

    axes[-1].set_xlabel("Time")
    for ax in axes:
        ax.set_xlim(first_spec_ts, last_spec_ts)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=LOCAL_TZ))
        ax.tick_params(axis="x", labelrotation=0, labelsize=8)
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    fig.suptitle(pkl_file.name)
    fig.tight_layout()
    png_name = f"{pkl_file.stem}.png"
    fig.savefig(output_dir / png_name, dpi=200)
    plt.close(fig)
