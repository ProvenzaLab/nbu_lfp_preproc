import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import re
from datetime import datetime
from matplotlib import pyplot as plt
from pathlib import Path
from pickle import load


LOCAL_TZ = "America/Chicago"
SPECTRUM_VMIN = -13
SPECTRUM_VMAX = 3
FALLBACK_SLEEP_START_HOUR = 20
FALLBACK_SLEEP_END_HOUR = 8

VISIT_FILE_RE = re.compile(
    r"^spectra_fooof_bandpower_(?P<channel>.+?)_visit(?P<visit>\d+)_(?P<start>\d{4}-\d{2}-\d{2})_to_(?P<end>\d{4}-\d{2}-\d{2})$"
)


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


def _parse_visit_metadata(file_stem):
    match = VISIT_FILE_RE.match(file_stem)
    if match is None:
        return {
            "visit_label": "visit_unknown",
            "visit_order": -1,
            "visit_start_date": pd.NaT,
            "visit_end_date": pd.NaT,
            "side": "UNKNOWN",
        }

    channel = match.group("channel")
    side = channel.split("_")[-1] if "_" in channel else "UNKNOWN"
    visit_order = int(match.group("visit"))
    return {
        "visit_label": f"visit{visit_order:02d}",
        "visit_order": visit_order,
        "visit_start_date": pd.Timestamp(match.group("start")),
        "visit_end_date": pd.Timestamp(match.group("end")),
        "side": side,
    }


def _build_default_sleep_intervals(index, tz=LOCAL_TZ):
    if index is None or len(index) == 0:
        return []

    idx = pd.DatetimeIndex(index)
    if idx.tz is None:
        idx = idx.tz_localize(tz)
    else:
        idx = idx.tz_convert(tz)

    idx_min = idx.min()
    idx_max = idx.max()
    day_start = (idx_min - pd.Timedelta(days=1)).normalize()
    day_end = (idx_max + pd.Timedelta(days=1)).normalize()

    intervals = []
    for day in pd.date_range(day_start, day_end, freq="D", tz=idx.tz):
        start_ts = day + pd.Timedelta(hours=FALLBACK_SLEEP_START_HOUR)
        end_ts = day + pd.Timedelta(days=1, hours=FALLBACK_SLEEP_END_HOUR)
        if end_ts < idx_min or start_ts > idx_max:
            continue
        intervals.append((start_ts, end_ts))
    return intervals


def _build_stage_highlight_intervals(series, stage_values=(1, 2, 3), pad_minutes=5):
    series = pd.to_numeric(pd.Series(series), errors="coerce").dropna().sort_index()
    if len(series) == 0:
        return []

    selected = []
    padding = pd.Timedelta(minutes=pad_minutes)
    stage_set = set(stage_values)
    for ts, val in series.items():
        stage_val = int(val)
        if stage_val not in stage_set:
            continue
        ts = pd.Timestamp(ts)
        selected.append((ts - padding, ts + padding))

    if not selected:
        return []

    selected.sort(key=lambda pair: pair[0])
    merged = [selected[0]]
    for start_ts, end_ts in selected[1:]:
        prev_start, prev_end = merged[-1]
        if start_ts <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end_ts))
        else:
            merged.append((start_ts, end_ts))
    return merged


def _segment_overlaps_sleep(segment_start_ts, segment_end_ts, sleep_intervals):
    for sleep_start_ts, sleep_end_ts in sleep_intervals:
        if segment_start_ts < sleep_end_ts and segment_end_ts > sleep_start_ts:
            return True
    return False


def _compute_segment_correlations(
    feature_df,
    sleep_intervals,
    visit_meta,
    sleep_rule,
):
    value_cols = ["active_sensing_band_log", "offset", "theta", "alpha", "oura_met"]
    pair_cols = [(value_cols[i], value_cols[j]) for i in range(len(value_cols)) for j in range(i + 1, len(value_cols))]

    segment_records = []
    grouped = feature_df.groupby(feature_df.index.floor("10min"))
    for segment_start, segment_df in grouped:
        segment_end = segment_start + pd.Timedelta(minutes=10)
        valid_df = segment_df[value_cols].dropna(how="all")
        if len(valid_df) < 3:
            continue

        corr_matrix = valid_df.corr(min_periods=3)
        if corr_matrix.empty:
            continue

        sleep_present = _segment_overlaps_sleep(segment_start, segment_end, sleep_intervals)
        n_rows = len(valid_df)
        for var_a, var_b in pair_cols:
            corr_val = corr_matrix.loc[var_a, var_b]
            if pd.isna(corr_val):
                continue
            segment_records.append(
                {
                    "file_name": visit_meta["file_name"],
                    "side": visit_meta["side"],
                    "visit_label": visit_meta["visit_label"],
                    "visit_order": visit_meta["visit_order"],
                    "visit_start_date": visit_meta["visit_start_date"],
                    "visit_end_date": visit_meta["visit_end_date"],
                    "segment_start": segment_start,
                    "segment_end": segment_end,
                    "var_a": var_a,
                    "var_b": var_b,
                    "pair": f"{var_a}__{var_b}",
                    "n_points": n_rows,
                    "correlation": float(corr_val),
                    "sleep_present": bool(sleep_present),
                    "sleep_rule": sleep_rule,
                }
            )

    return segment_records


def _plot_measure(
    ax,
    series,
    title,
    ylabel,
    color,
    sleep_intervals=None,
    scatter_clip_min=None,
    scatter_clip_max=None,
):
    series = pd.to_numeric(pd.Series(series), errors="coerce").dropna().sort_index()
    if len(series) == 0:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return

    if sleep_intervals:
        for sleep_start_ts, sleep_end_ts in sleep_intervals:
            ax.axvspan(sleep_start_ts, sleep_end_ts, color="green", alpha=0.2)

    resampled = series.resample("10min")
    avg_10min = resampled.mean().dropna()
    q10_10min = resampled.quantile(0.10).dropna()
    q90_10min = resampled.quantile(0.90).dropna()
    scatter_vals = series.values
    if scatter_clip_min is not None or scatter_clip_max is not None:
        scatter_vals = np.clip(scatter_vals, scatter_clip_min, scatter_clip_max)
    ax.scatter(series.index, scatter_vals, s=10, color=color, alpha=0.3)
    ax.plot(avg_10min.index, avg_10min.values, color="red", linewidth=3)
    ax.plot(q10_10min.index, q10_10min.values, color="red", linestyle="--", linewidth=1.5, alpha=0.9)
    ax.plot(q90_10min.index, q90_10min.values, color="red", linestyle="--", linewidth=1.5, alpha=0.9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def _plot_sleep_stages(
    ax,
    series,
    title,
    ylabel,
    sleep_intervals=None,
):
    sleep_map = {
        1: "Deep",
        2: "Light",
        3: "REM",
        4: "Awake",
    }

    series = pd.to_numeric(pd.Series(series), errors="coerce").dropna().sort_index()

    if len(series) == 0:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return

    if sleep_intervals:
        for sleep_start_ts, sleep_end_ts in sleep_intervals:
            ax.axvspan(sleep_start_ts, sleep_end_ts, color="green", alpha=0.2)

    # Draw discrete 10-minute bars (±5 min around each sample) without connecting entries.
    half_window = pd.Timedelta(minutes=2.5)
    for ts, stage in series.items():
        if pd.isna(stage):
            continue
        stage_val = int(stage)
        if stage_val not in sleep_map:
            continue
        start_num = mdates.date2num((pd.Timestamp(ts) - half_window).to_pydatetime())
        end_num = mdates.date2num((pd.Timestamp(ts) + half_window).to_pydatetime())
        width_num = end_num - start_num
        ax.broken_barh(
            [(start_num, width_num)],
            (stage_val - 0.35, 0.7),
            facecolors="tab:green",
            edgecolors="none",
            alpha=0.55,
        )
    ax.xaxis_date()

    present = sorted({int(v) for v in series.values if int(v) in sleep_map})
    if not present:
        present = [1, 2, 3, 4]
    ax.set_yticks(present)
    ax.set_yticklabels([f"{v}={sleep_map[v]}" for v in present])
    ax.set_ylim(min(present) - 0.5, max(present) + 0.5)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def _plot_source_channel(
    ax,
    series,
    title,
    sleep_intervals=None,
):
    series = pd.Series(series).dropna().sort_index().astype(str)

    if len(series) == 0:
        ax.set_title(title)
        ax.set_ylabel("Channel")
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return

    if sleep_intervals:
        for sleep_start_ts, sleep_end_ts in sleep_intervals:
            ax.axvspan(sleep_start_ts, sleep_end_ts, color="green", alpha=0.2)

    channels = sorted(series.unique())
    cmap = plt.get_cmap("tab10")
    for idx, channel in enumerate(channels):
        times = series.index[series == channel]
        color = cmap(idx % 10)
        # Dense vertical markers create a bar-like timeline for selected channel provenance.
        ax.vlines(times, idx - 0.35, idx + 0.35, color=color, linewidth=1.0, alpha=0.95)

    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels(channels)
    ax.set_ylim(-0.7, len(channels) - 0.3)
    ax.set_title(title)
    ax.set_ylabel("Channel")


def get_oura_scores(oura_scores, date_first, date_last, subject):
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
                bpm_series.set_index("timestamp", inplace=True)
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


oura = pd.read_pickle("oura_scores.pkl")

base_dir = Path("nbu_concat_single_channel")
pkl_files = sorted(base_dir.glob("spectra_fooof_bandpower_*.pkl"))

output_dir = Path("out_figures_concat_single_channel_tight")
output_dir.mkdir(parents=True, exist_ok=True)

path_sleep_times_TRBD001 = "trbd001_nbu_sleep_onset.pkl"
with open(path_sleep_times_TRBD001, "rb") as f:
    sleep_times_TRBD001 = load(f)

sleep_times_TRBD001["nbu_visit"] = pd.to_datetime(sleep_times_TRBD001["nbu_visit"])
sleep_dates = pd.Series(sleep_times_TRBD001["nbu_visit"].dt.date.unique())

all_corr_records = []
plot_payloads = []

for pkl_file in pkl_files:
    with open(pkl_file, "rb") as f:
        data = load(f)

    df_features = data["df_features"]
    spectrum = data["df_spec"].iloc[:, 1:31].copy()

    df_features = df_features.sort_index()
    spectrum = spectrum.sort_index()
    spectrum = spectrum.tz_convert(LOCAL_TZ)
    df_features.index = df_features.index.tz_convert(LOCAL_TZ)

    source_channel_series = pd.Series(dtype="object")
    if "source_channel" in df_features.columns:
        source_channel_series = df_features["source_channel"].copy()

    df_features_numeric = df_features.drop(columns=["source_channel"], errors="ignore")
    # Keep numeric blocks numeric so resample().mean() uses fast vectorized code paths.
    df_features_numeric = df_features_numeric.apply(pd.to_numeric, errors="coerce")
    spectrum = spectrum.apply(pd.to_numeric, errors="coerce")
    df_features = df_features_numeric.resample("10s").mean()
    spectrum = spectrum.resample("10s").mean()

    df_features = df_features[~df_features.index.duplicated(keep="first")]
    spectrum = spectrum[~spectrum.index.duplicated(keep="first")]

    if len(source_channel_series) > 0:
        source_channel_series = source_channel_series.sort_index()
        source_channel_series = source_channel_series[~source_channel_series.index.duplicated(keep="first")]
        source_channel_series = source_channel_series.resample("10s").ffill()
        source_channel_series = source_channel_series.reindex(df_features.index, method="ffill")

    active_sensing_band = pd.to_numeric(df_features["active_sensing_band"], errors="coerce")
    active_sensing_band_log = np.log(active_sensing_band.where(active_sensing_band > 0))

    source_channel_plot_series = source_channel_series
    if len(source_channel_series) > 0:
        source_channel_plot_series = source_channel_series.where(active_sensing_band > 0)

    offset = pd.to_numeric(df_features["offset"], errors="coerce")
    theta = pd.to_numeric(df_features["theta"], errors="coerce")
    alpha = pd.to_numeric(df_features["alpha"], errors="coerce")

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

    bpm_oura, met_oura, sleep_oura = get_oura_scores(oura, first_spec_ts, last_spec_ts, "TRBD001")
    bpm_oura = _ensure_datetime_index(bpm_oura)
    met_oura = _ensure_datetime_index(met_oura)
    sleep_oura = _ensure_datetime_index(sleep_oura)
    bpm_oura = bpm_oura.tz_convert(LOCAL_TZ)
    met_oura = met_oura.tz_convert(LOCAL_TZ)
    sleep_oura = sleep_oura.tz_convert(LOCAL_TZ)

    met_oura_series = pd.Series(dtype=float)
    if isinstance(met_oura, pd.DataFrame) and "met" in met_oura.columns:
        met_oura_series = met_oura["met"]

    sleep_oura_series = pd.to_numeric(pd.Series(sleep_oura), errors="coerce")
    sleep_stage_highlight_intervals = _build_stage_highlight_intervals(
        sleep_oura_series,
        stage_values=(1, 2, 3),
        pad_minutes=5,
    )

    unique_dates = pd.Series(pd.Series(df_features.index.date).unique())
    sleep_intervals = []
    matching_dates = unique_dates[unique_dates.isin(sleep_dates.values)]
    for date_sleep in matching_dates:
        sleep_rows = sleep_times_TRBD001[
            sleep_times_TRBD001["nbu_visit"].dt.date == date_sleep
        ]
        for _, sleep_row in sleep_rows.iterrows():
            sleep_start_ts = _sleep_time_to_timestamp(sleep_row["sleep_start"], date_sleep)
            sleep_end_ts = _sleep_time_to_timestamp(sleep_row["sleep_end"], date_sleep)
            if sleep_end_ts < sleep_start_ts:
                sleep_end_ts = sleep_end_ts + pd.Timedelta(days=1)
            sleep_intervals.append((sleep_start_ts, sleep_end_ts))

    sleep_rule = "sleep_intervals"
    if not sleep_intervals:
        sleep_intervals = _build_default_sleep_intervals(df_features.index)
        sleep_rule = "fallback_20_08"

    met_oura_10s = pd.to_numeric(met_oura_series, errors="coerce")
    met_oura_10s = pd.Series(met_oura_10s).sort_index()
    if len(met_oura_10s) > 0:
        met_oura_10s = met_oura_10s.resample("10s").mean().reindex(df_features.index, method="nearest")
    else:
        met_oura_10s = pd.Series(index=df_features.index, dtype=float)

    corr_input_df = pd.DataFrame(
        {
            "active_sensing_band_log": active_sensing_band_log,
            "offset": offset,
            "theta": theta,
            "alpha": alpha,
            "oura_met": met_oura_10s,
        }
    )

    visit_meta = _parse_visit_metadata(pkl_file.stem)
    visit_meta["file_name"] = pkl_file.name
    all_corr_records.extend(
        _compute_segment_correlations(
            feature_df=corr_input_df,
            sleep_intervals=sleep_intervals,
            visit_meta=visit_meta,
            sleep_rule=sleep_rule,
        )
    )

    plot_payloads.append(
        {
            "file_name": pkl_file.name,
            "file_stem": pkl_file.stem,
            "side": visit_meta["side"],
            "visit_start_date": visit_meta["visit_start_date"],
            "spectrum": spectrum,
            "active_sensing_band_log": active_sensing_band_log,
            "sleep_oura_series": sleep_oura_series,
            "sleep_stage_highlight_intervals": sleep_stage_highlight_intervals,
            "sleep_intervals": sleep_intervals,
            "first_spec_ts": first_spec_ts,
            "last_spec_ts": last_spec_ts,
        }
    )


if plot_payloads:
    side_groups = {}
    for payload in plot_payloads:
        side_groups.setdefault(payload["side"], []).append(payload)

    for side, side_payloads in sorted(side_groups.items()):
        n_cols = len(side_payloads)
        fig, axes = plt.subplots(
            3,
            n_cols,
            figsize=(4.6 * n_cols, 10),
            gridspec_kw={"height_ratios": [2, 1.5, 1.5]},
            squeeze=False,
        )

        for col_idx, payload in enumerate(side_payloads):
            ax_spec = axes[0, col_idx]
            ax_active = axes[1, col_idx]
            ax_sleep = axes[2, col_idx]

            spectrum = payload["spectrum"]
            sleep_intervals = payload["sleep_intervals"]
            x_start = payload["first_spec_ts"].floor("h")
            x_end = payload["last_spec_ts"].ceil("10min")
            full_hour_ticks = pd.date_range(start=x_start, end=x_end, freq="1h", tz=x_start.tz)

            if not spectrum.empty:
                x_num = mdates.date2num(spectrum.index.to_pydatetime())
                y_freq = spectrum.columns.to_numpy(dtype=float)
                z_power = np.log(spectrum.to_numpy(dtype=float).T)
                ax_spec.pcolormesh(
                    x_num,
                    y_freq,
                    z_power,
                    shading="gouraud",
                    cmap="viridis",
                    vmin=SPECTRUM_VMIN,
                    vmax=SPECTRUM_VMAX,
                )
                ax_spec.xaxis_date()

            _plot_measure(
                ax_active,
                payload["active_sensing_band_log"],
                "Active Sensing Band Power (log)",
                "log(Band Power)",
                color="blue",
                sleep_intervals=sleep_intervals,
                scatter_clip_min=-1.5,
                scatter_clip_max=2,
            )
            for stage_start_ts, stage_end_ts in payload["sleep_stage_highlight_intervals"]:
                ax_active.axvspan(
                    stage_start_ts,
                    stage_end_ts,
                    color="limegreen",
                    alpha=0.28,
                    zorder=0.15,
                )
            ax_active.set_ylim(-2, 2)

            _plot_sleep_stages(
                ax_sleep,
                payload["sleep_oura_series"],
                "Oura Sleep Stages",
                "Stage",
                sleep_intervals=sleep_intervals,
            )

            title_side = str(payload["side"]).upper()
            visit_start_date = payload.get("visit_start_date", pd.NaT)
            if pd.notna(visit_start_date):
                title_label = f"{title_side} {pd.Timestamp(visit_start_date).strftime('%Y-%m-%d')}"
            else:
                title_label = title_side
            ax_spec.set_title(title_label, fontsize=9)
            if col_idx == 0:
                ax_spec.set_ylabel("Frequency (Hz)")
            else:
                ax_spec.set_ylabel("")
                ax_active.set_ylabel("")
                ax_sleep.set_ylabel("")

            for row_idx, ax in enumerate((ax_spec, ax_active, ax_sleep)):
                ax.set_xlim(x_start, x_end)
                ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30], tz=LOCAL_TZ))
                ax.xaxis.set_major_formatter(
                    mticker.FuncFormatter(
                        lambda x, pos: pd.Timestamp(mdates.num2date(x)).tz_convert(LOCAL_TZ).strftime("%m-%d %H:%M")
                        if pd.Timestamp(mdates.num2date(x)).tz_convert(LOCAL_TZ).minute % 30 == 0
                        else ""
                    )
                )
                for hour_ts in full_hour_ticks:
                    ax.axvline(hour_ts, linestyle="--", linewidth=0.7, color="0.65", alpha=0.45, zorder=0)
                ax.tick_params(axis="x", labelrotation=90, labelsize=5, labelbottom=(row_idx == 2))
                plt.setp(ax.get_xticklabels(), rotation=90, ha="center")

            ax_sleep.set_xlabel("Time")

        fig.suptitle(f"{side} side", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        side_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(side))
        fig.savefig(output_dir / f"all_visits_side_by_side_{side_slug}_clipped.png", dpi=200)
        plt.close(fig)


corr_out = pd.DataFrame(all_corr_records)
if not corr_out.empty:
    corr_out = corr_out.sort_values(
        ["side", "visit_order", "segment_start", "pair"],
        kind="mergesort",
    ).reset_index(drop=True)

corr_csv = output_dir / "corr_10min_segments.csv"
corr_pkl = output_dir / "corr_10min_segments.pkl"
corr_out.to_csv(corr_csv, index=False)
corr_out.to_pickle(corr_pkl)

print(f"Saved correlations to: {corr_csv}")
print(f"Saved correlations to: {corr_pkl}")
