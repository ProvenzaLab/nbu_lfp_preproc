import re
from datetime import datetime
from pathlib import Path
from pickle import load

import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


LOCAL_TZ = "America/Chicago"
BASE_DIR = Path("nbu_concat_single_channel")
OUTPUT_DIR = Path("out_figures_write_out_events_NBU")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FALLBACK_SLEEP_START_HOUR = 20
FALLBACK_SLEEP_END_HOUR = 8
SLEEP_STAGE_VALUES = (1, 2, 3)
ANNOTE_90TH_PERCENTILE = True
RANDOM_SELECTION_SEED = 7

VISIT_FILE_RE = re.compile(
    r"^spectra_highres_fooof_bandpower_(?P<channel>.+?)_visit(?P<visit>\d+)_(?P<start>\d{4}-\d{2}-\d{2})_to_(?P<end>\d{4}-\d{2}-\d{2})$"
)


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


def _concat_or_empty(items, as_series=False, name=None):
    if not items:
        if as_series:
            return pd.Series(dtype=float, name=name)
        return pd.DataFrame()
    out = pd.concat(items)
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


def get_oura_scores(oura_scores, date_first, date_last, subject):
    date_range = pd.DatetimeIndex(
        list(pd.date_range(start=date_first, end=date_last, freq="D"))
        + [pd.Timestamp(date_first), pd.Timestamp(date_last)]
    ).unique().sort_values()

    sleep_list = []
    for date in date_range:
        date_str = date.strftime("%Y-%m-%d")
        key_oura = f"{subject}_{date_str}"
        if key_oura in oura_scores.keys():
            df_oura = oura_scores[key_oura]
            if "sleep_phases" in df_oura:
                sleep_list.append(df_oura["sleep_phases"])

    sleep_out = _concat_or_empty(sleep_list, as_series=True, name="sleep_stage")
    if isinstance(sleep_out, pd.DataFrame):
        sleep_out = sleep_out.iloc[:, 0]
    sleep_out.name = "sleep_stage"
    return sleep_out


def _build_stage_highlight_intervals(series, stage_values=SLEEP_STAGE_VALUES, pad_minutes=5):
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


def _build_pickle_sleep_intervals(sleep_times_df, visit_dates):
    if sleep_times_df is None or len(sleep_times_df) == 0 or len(visit_dates) == 0:
        return []

    intervals = []
    matching_dates = pd.Series(pd.to_datetime(list(visit_dates)).date).drop_duplicates()
    for date_sleep in matching_dates:
        sleep_rows = sleep_times_df[sleep_times_df["nbu_visit"].dt.date == date_sleep]
        for _, sleep_row in sleep_rows.iterrows():
            sleep_start_ts = _sleep_time_to_timestamp(sleep_row["sleep_start"], date_sleep)
            sleep_end_ts = _sleep_time_to_timestamp(sleep_row["sleep_end"], date_sleep)
            if sleep_end_ts < sleep_start_ts:
                sleep_end_ts = sleep_end_ts + pd.Timedelta(days=1)
            intervals.append((sleep_start_ts, sleep_end_ts))
    return intervals


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


def _merge_intervals(intervals):
    intervals = [(pd.Timestamp(start_ts), pd.Timestamp(end_ts)) for start_ts, end_ts in intervals if pd.notna(start_ts) and pd.notna(end_ts)]
    if not intervals:
        return []

    intervals.sort(key=lambda pair: pair[0])
    merged = [intervals[0]]
    for start_ts, end_ts in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start_ts <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end_ts))
        else:
            merged.append((start_ts, end_ts))
    return merged


def _build_interval_overlap_mask(index, intervals):
    idx = pd.DatetimeIndex(index)
    overlap_mask = np.zeros(len(idx), dtype=bool)
    if not intervals or len(idx) == 0:
        return overlap_mask

    for start_ts, end_ts in intervals:
        overlap_mask |= ((idx >= start_ts) & (idx <= end_ts))
    return overlap_mask


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


def _prepare_active_sensing_band(df_features):
    if "active_sensing_band" not in df_features.columns:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    df_features = df_features.sort_index().copy()
    df_features.index = pd.to_datetime(df_features.index)
    if df_features.index.tz is None:
        df_features.index = df_features.index.tz_localize(LOCAL_TZ)
    else:
        df_features.index = df_features.index.tz_convert(LOCAL_TZ)

    active_sensing_band = pd.to_numeric(df_features["active_sensing_band"], errors="coerce")
    active_sensing_band = active_sensing_band.resample("10s").mean().dropna()
    if len(active_sensing_band) == 0:
        return active_sensing_band, pd.Series(dtype=float)

    mean_10min_late = active_sensing_band.resample("10min").mean()
    mean_10min_late.index = mean_10min_late.index + pd.Timedelta(minutes=10)
    return active_sensing_band, mean_10min_late


def _build_visit_table(
    active_sensing_band,
    mean_10min_late,
    visit_meta,
    file_name,
    sleep_intervals,
    annote_90th_percentile,
    rng,
):
    if len(active_sensing_band) == 0:
        return pd.DataFrame()

    sample_frame = pd.DataFrame(
        {
            "time_point": active_sensing_band.index,
            "active_sensing_band": active_sensing_band.values,
        }
    )

    mean_frame = pd.DataFrame(
        {
            "time_point": mean_10min_late.index,
            "active_sensing_band_10min_late_mean": mean_10min_late.values,
        }
    )

    merged = pd.merge_asof(
        sample_frame.sort_values("time_point"),
        mean_frame.sort_values("time_point"),
        on="time_point",
        direction="backward",
    )

    sleep_mask = _build_interval_overlap_mask(merged["time_point"], sleep_intervals)
    day_mask = ~sleep_mask
    day_frame = merged.loc[day_mask].copy()
    if len(day_frame) == 0:
        return pd.DataFrame()

    percentile_90 = float(day_frame["active_sensing_band"].quantile(0.90))
    percentile_03 = float(day_frame["active_sensing_band"].quantile(0.03))
    percentile_97 = float(day_frame["active_sensing_band"].quantile(0.97))
    above_90th_percentile = day_frame["active_sensing_band"] >= percentile_90

    if annote_90th_percentile:
        selected_for_annotation = above_90th_percentile.to_numpy(copy=True)
        annotation_mode = "90th_percentile"
    else:
        selected_for_annotation = np.zeros(len(day_frame), dtype=bool)
        sample_count = int(above_90th_percentile.sum())
        candidate_idx = np.flatnonzero(~above_90th_percentile.to_numpy())
        if sample_count > 0 and len(candidate_idx) > 0:
            chosen_idx = rng.choice(candidate_idx, size=min(sample_count, len(candidate_idx)), replace=False)
            selected_for_annotation[chosen_idx] = True
        annotation_mode = "random_non_90th_percentile"

    day_frame["is_sleep"] = False
    day_frame["during_day"] = True
    day_frame["visit_label"] = visit_meta["visit_label"]
    day_frame["visit_order"] = visit_meta["visit_order"]
    day_frame["side"] = visit_meta["side"]
    day_frame["file_name"] = file_name
    day_frame["visit_start_date"] = visit_meta["visit_start_date"]
    day_frame["visit_end_date"] = visit_meta["visit_end_date"]
    day_frame["active_sensing_band_p90"] = percentile_90
    day_frame["active_sensing_band_p03"] = percentile_03
    day_frame["active_sensing_band_p97"] = percentile_97
    day_frame["above_90th_percentile"] = above_90th_percentile
    day_frame["selected_for_annotation"] = selected_for_annotation
    day_frame["annotation_mode"] = annotation_mode
    return day_frame


def _plot_visit_panel(
    ax_ts,
    ax_hist,
    active_sensing_band,
    histogram_series,
    mean_10min_late,
    visit_meta,
    percentile_90,
    percentile_03,
    percentile_97,
    sleep_intervals,
    selected_time_points,
    annotation_label,
):
    ax_ts.scatter(active_sensing_band.index, active_sensing_band.values, s=10, color="tab:blue", alpha=0.25)
    selected_mask = active_sensing_band.index.isin(selected_time_points) if selected_time_points is not None else None
    if selected_mask is not None and np.any(selected_mask):
        ax_ts.scatter(
            active_sensing_band.index[selected_mask],
            active_sensing_band.values[selected_mask],
            s=26,
            color="red",
            alpha=0.95,
            zorder=3,
            linewidths=0.25,
            edgecolors="white",
            label=annotation_label,
        )
    ax_ts.plot(mean_10min_late.index, mean_10min_late.values, color="red", linewidth=2.5)
    for sleep_start_ts, sleep_end_ts in sleep_intervals:
        ax_ts.axvspan(sleep_start_ts, sleep_end_ts, color="green", alpha=0.2, zorder=0)
    ax_ts.set_title(f"{visit_meta['visit_label']} ({visit_meta['side']})", fontsize=10)
    ax_ts.set_ylabel("9 Hz power")
    ax_ts.set_ylim(percentile_03, percentile_97)
    ax_ts.grid(True, alpha=0.2)
    ax_ts.tick_params(axis="x", labelbottom=False)

    sns.histplot(
        histogram_series.dropna(),
        bins=50,
        binrange=(percentile_03, percentile_97),
        ax=ax_hist,
        color="steelblue",
        alpha=0.75,
    )
    ax_hist.axvline(
        percentile_90,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"90th percentile = {percentile_90:.3f}",
    )
    ax_hist.text(
        0.98,
        0.95,
        f"90th percentile\n{percentile_90:.3f}",
        transform=ax_hist.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )
    ax_hist.set_xlabel("9 Hz power")
    ax_hist.set_ylabel("Count")
    ax_hist.legend(frameon=False, fontsize=8, loc="best")
    ax_hist.grid(True, alpha=0.2)
    ax_hist.set_xlim(percentile_03, percentile_97)


def _build_visit_sleep_intervals(df_features, sleep_times_df, sleep_oura_series):
    visit_dates = pd.Series(pd.to_datetime(df_features.index).date).drop_duplicates()
    pickle_intervals = _build_pickle_sleep_intervals(sleep_times_df, visit_dates)
    oura_intervals = _build_stage_highlight_intervals(sleep_oura_series, stage_values=SLEEP_STAGE_VALUES, pad_minutes=5)
    combined = _merge_intervals(pickle_intervals + oura_intervals)
    if not combined:
        combined = _build_default_sleep_intervals(df_features.index)
    return combined


def main(annote_90th_percentile=ANNOTE_90TH_PERCENTILE):
    pkl_files = sorted(BASE_DIR.glob("spectra_highres_fooof_bandpower_*.pkl"))
    if not pkl_files:
        print(f"No input files found in {BASE_DIR}")
        return

    oura = pd.read_pickle("oura_scores.pkl")
    path_sleep_times_TRBD001 = "trbd001_nbu_sleep_onset.pkl"
    with open(path_sleep_times_TRBD001, "rb") as f:
        sleep_times_TRBD001 = load(f)
    sleep_times_TRBD001["nbu_visit"] = pd.to_datetime(sleep_times_TRBD001["nbu_visit"])

    visit_frames = []
    visit_payloads = []
    rng = np.random.default_rng(RANDOM_SELECTION_SEED)
    mode_slug = "90th_percentile" if annote_90th_percentile else "random_non_90th_percentile"

    pdf_path = OUTPUT_DIR / f"write_out_events_NBU_plots_{mode_slug}.pdf"
    with PdfPages(pdf_path) as pdf:
        for pkl_file in pkl_files:
            with open(pkl_file, "rb") as f:
                data = load(f)

            df_features = data.get("df_features", pd.DataFrame())
            if len(df_features) == 0:
                continue

            df_features = df_features.sort_index().copy()
            df_features.index = pd.to_datetime(df_features.index)
            if df_features.index.tz is None:
                df_features.index = df_features.index.tz_localize(LOCAL_TZ)
            else:
                df_features.index = df_features.index.tz_convert(LOCAL_TZ)

            visit_meta = _parse_visit_metadata(pkl_file.stem)
            active_sensing_band, mean_10min_late = _prepare_active_sensing_band(df_features)
            if len(active_sensing_band) == 0:
                continue

            sleep_oura_series = _ensure_datetime_index(
                get_oura_scores(oura, active_sensing_band.index.min(), active_sensing_band.index.max(), "TRBD001")
            )
            sleep_oura_series = pd.to_numeric(pd.Series(sleep_oura_series), errors="coerce")
            sleep_intervals = _build_visit_sleep_intervals(df_features, sleep_times_TRBD001, sleep_oura_series)

            visit_table = _build_visit_table(
                active_sensing_band=active_sensing_band,
                mean_10min_late=mean_10min_late,
                visit_meta=visit_meta,
                file_name=pkl_file.name,
                sleep_intervals=sleep_intervals,
                annote_90th_percentile=annote_90th_percentile,
                rng=rng,
            )

            if len(visit_table) == 0:
                continue

            visit_frames.append(visit_table)
            visit_payloads.append(
                {
                    "visit_meta": visit_meta,
                    "active_sensing_band": active_sensing_band,
                    "day_series": visit_table["active_sensing_band"],
                    "mean_10min_late": mean_10min_late,
                    "percentile_90": float(visit_table["active_sensing_band_p90"].iloc[0]),
                    "sleep_intervals": sleep_intervals,
                    "selected_time_points": visit_table.loc[visit_table["selected_for_annotation"], "time_point"].to_list(),
                    "annotation_label": mode_slug,
                }
            )

        if visit_payloads:
            side_groups = {}
            for payload in visit_payloads:
                side_groups.setdefault(payload["visit_meta"]["side"], []).append(payload)

            for side, side_payloads in sorted(side_groups.items()):
                side_payloads = sorted(
                    side_payloads,
                    key=lambda payload: (
                        payload["visit_meta"]["visit_order"],
                        payload["visit_meta"]["visit_label"],
                        payload["active_sensing_band"].index.min(),
                    ),
                )

                n_cols = len(side_payloads)
                fig, axes = plt.subplots(
                    2,
                    n_cols,
                    figsize=(5.2 * n_cols, 7.4),
                    gridspec_kw={"height_ratios": [2.1, 1.2]},
                    squeeze=False,
                )

                for col_idx, payload in enumerate(side_payloads):
                    ax_ts = axes[0, col_idx]
                    ax_hist = axes[1, col_idx]
                    active_sensing_band = payload["active_sensing_band"]
                    day_only = payload["day_series"]
                    mean_10min_late = payload["mean_10min_late"]
                    percentile_90 = payload["percentile_90"]
                    percentile_03 = float(day_only.quantile(0.03))
                    percentile_97 = float(day_only.quantile(0.97))

                    _plot_visit_panel(
                        ax_ts=ax_ts,
                        ax_hist=ax_hist,
                        active_sensing_band=active_sensing_band,
                        histogram_series=day_only,
                        mean_10min_late=mean_10min_late,
                        visit_meta=payload["visit_meta"],
                        percentile_90=percentile_90,
                        percentile_03=percentile_03,
                        percentile_97=percentile_97,
                        sleep_intervals=payload["sleep_intervals"],
                        selected_time_points=payload["selected_time_points"],
                        annotation_label=payload["annotation_label"],
                    )

                    x_start = active_sensing_band.index.min().floor("h")
                    x_end = active_sensing_band.index.max().ceil("10min")
                    ax_ts.set_xlim(x_start, x_end)

                    ax_hist.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))

                fig.suptitle(f"{side} side", fontsize=12)
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                side_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(side))
                fig.savefig(OUTPUT_DIR / f"write_out_events_NBU_{mode_slug}_{side_slug}.png", dpi=200)
                pdf.savefig(fig, dpi=200)
                plt.close(fig)

    events_out = pd.concat(visit_frames, ignore_index=True) if visit_frames else pd.DataFrame()
    if not events_out.empty:
        events_out = events_out.sort_values(
            ["side", "visit_order", "time_point"],
            kind="mergesort",
        ).reset_index(drop=True)

    csv_path = OUTPUT_DIR / f"write_out_events_NBU_{mode_slug}.csv"
    events_out.to_csv(csv_path, index=False)
    events_out.query("selected_for_annotation == True and side == 'RIGHT'").to_csv(
        OUTPUT_DIR / f"write_out_events_NBU_{mode_slug}_right_selected.csv",
        index=False,
    )
    print(f"Saved plots to: {pdf_path}")
    print(f"Saved plots to: {OUTPUT_DIR}")
    print(f"Saved event table to: {csv_path}")


if __name__ == "__main__":
    main()