from __future__ import annotations

import argparse
import pickle
import re
from datetime import date, datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOCAL_TZ = "America/Chicago"
FREQUENCY_MIN = 0.1
FREQUENCY_MAX = 44.9
SLEEP_STAGE_COLORS = {
    1: "#4c5a8a",
    2: "#6fa8dc",
    3: "#55a868",
    4: "#d98c3f",
}
SLEEP_STAGE_NAMES = {1: "Deep", 2: "Light", 3: "REM", 4: "Awake"}
VISIT_FILE_RE = re.compile(
    r"_(?P<side>LEFT|RIGHT)_visit(?P<visit>\d+)_"
    r"(?P<start>\d{4}-\d{2}-\d{2})_to_(?P<end>\d{4}-\d{2}-\d{2})"
    r"(?:_(?P<split>before_split|after_split))?$"
)


def _local_index(index: pd.Index) -> pd.DatetimeIndex:
    result = pd.DatetimeIndex(pd.to_datetime(index))
    if result.tz is None:
        return result.tz_localize(LOCAL_TZ)
    return result.tz_convert(LOCAL_TZ)


def _read_visit_data(input_dir: Path) -> dict[tuple[str, int], dict[str, object]]:
    visits: dict[tuple[str, int], dict[str, object]] = {}
    files = sorted(input_dir.glob("spectra_highres_spectra_*.pkl"))
    if not files:
        raise FileNotFoundError(f"No visit spectra pickles found in {input_dir}")

    for path in files:
        match = VISIT_FILE_RE.search(path.stem)
        if match is None:
            continue

        with path.open("rb") as file:
            payload = pickle.load(file)
        spectrum = payload.get("df_spec")
        features = payload.get("df_features")
        if not isinstance(spectrum, pd.DataFrame) or spectrum.empty:
            continue

        spectrum = spectrum.copy()
        spectrum.index = _local_index(spectrum.index)
        spectrum.columns = pd.to_numeric(spectrum.columns, errors="coerce")
        spectrum = spectrum.loc[:, spectrum.columns.notna()]
        spectrum = spectrum.loc[:, (spectrum.columns > 0) & (spectrum.columns < 45)]
        spectrum = spectrum.apply(pd.to_numeric, errors="coerce").resample("10s").mean()
        if spectrum.empty:
            continue

        if isinstance(features, pd.DataFrame):
            features = features.copy()
            features.index = _local_index(features.index)
            features = features.apply(pd.to_numeric, errors="coerce").resample("10s").mean()
        else:
            features = pd.DataFrame(index=spectrum.index)

        key = (match.group("side"), int(match.group("visit")))
        visit = visits.setdefault(
            key,
            {
                "spectra": [],
                "features": [],
                "start": pd.Timestamp(match.group("start")),
                "end": pd.Timestamp(match.group("end")),
            },
        )
        visit["spectra"].append(spectrum)
        visit["features"].append(features)
        visit["start"] = min(visit["start"], pd.Timestamp(match.group("start")))
        visit["end"] = max(visit["end"], pd.Timestamp(match.group("end")))

    for visit in visits.values():
        for key in ("spectra", "features"):
            frames = visit[key]
            merged = pd.concat(frames).sort_index()
            visit[key] = merged.loc[~merged.index.duplicated(keep="first")]

    return visits


def _load_oura_scores(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Oura data not found: {path}")
    scores = pd.read_pickle(path)
    if not isinstance(scores, dict):
        raise TypeError(f"Expected a dictionary in {path}, got {type(scores).__name__}")
    return scores


def _sleep_stage_series(
    oura_scores: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    pieces = []
    for score_date in pd.date_range(
        start.normalize() - pd.Timedelta(days=1),
        end.normalize() + pd.Timedelta(days=1),
        freq="D",
    ).date:
        entry = oura_scores.get(f"TRBD001_{score_date:%Y-%m-%d}", {})
        stages = entry.get("sleep_phases") if isinstance(entry, dict) else None
        if isinstance(stages, pd.DataFrame):
            column = "sleep_phase_5_min" if "sleep_phase_5_min" in stages.columns else stages.columns[0]
            stages = stages[column]
        if isinstance(stages, pd.Series) and not stages.empty:
            stages = stages.copy()
            stages.index = _local_index(stages.index)
            pieces.append(stages)

    if not pieces:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz=LOCAL_TZ))

    result = pd.concat(pieces).sort_index()
    result = result[~result.index.duplicated(keep="first")]
    return pd.to_numeric(result, errors="coerce").loc[
        (result.index >= start) & (result.index <= end)
    ].dropna()


def _timestamp(value: object, day: date) -> pd.Timestamp:
    if isinstance(value, (int, float, np.number)):
        result = pd.Timestamp(day, tz=LOCAL_TZ) + pd.Timedelta(hours=float(value))
    elif hasattr(value, "hour") and hasattr(value, "minute") and not isinstance(value, (str, pd.Timestamp)):
        result = pd.Timestamp(datetime.combine(day, value))
    else:
        result = pd.Timestamp(value)
    if result.tzinfo is None:
        return result.tz_localize(LOCAL_TZ)
    return result.tz_convert(LOCAL_TZ)


def _sleep_intervals(path: Path) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not path.exists():
        raise FileNotFoundError(f"Sleep-onset data not found: {path}")
    frame = pd.read_pickle(path)
    required = {"nbu_visit", "sleep_start", "sleep_end"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain columns: {', '.join(sorted(required))}")

    intervals = []
    for _, row in frame.iterrows():
        day = pd.Timestamp(row["nbu_visit"]).date()
        start = _timestamp(row["sleep_start"], day)
        end = _timestamp(row["sleep_end"], day)
        if end < start:
            end += pd.Timedelta(days=1)
        intervals.append((start, end))
    return intervals


def _stage_intervals(stages: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    selected = []
    padding = pd.Timedelta(minutes=5)
    for timestamp, value in stages.items():
        if int(value) in (1, 2, 3):
            selected.append((timestamp - padding, timestamp + padding))
    if not selected:
        return []

    selected.sort(key=lambda interval: interval[0])
    merged = [selected[0]]
    for start, end in selected[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _plot_active_sensing(ax, series: pd.Series, sleep_intervals, stages: pd.Series) -> None:
    series = pd.to_numeric(series, errors="coerce")
    series = np.log(series.where(series > 0)).dropna().sort_index()
    for start, end in sleep_intervals:
        ax.axvspan(start, end, color="green", alpha=0.2)
    stage_intervals = _stage_intervals(stages)
    for start, end in stage_intervals:
        ax.axvspan(start, end, color="limegreen", alpha=0.28, zorder=0.15)

    if series.empty:
        ax.text(0.5, 0.5, "No active-sensing data", transform=ax.transAxes, ha="center", va="center")
    else:
        keep = np.ones(len(series), dtype=bool)
        for start, end in sleep_intervals:
            keep &= ~((series.index >= start) & (series.index <= end))
        for start, end in stage_intervals:
            keep &= ~((series.index >= start) & (series.index <= end))
        daytime = series.loc[keep]
        resampled = daytime.resample("10min")
        average = resampled.mean().dropna()
        lower = resampled.quantile(0.10).dropna()
        upper = resampled.quantile(0.90).dropna()
        ax.scatter(daytime.index, daytime.clip(-1.5, 2), s=10, color="blue", alpha=0.3)
        ax.plot(average.index, average.values, color="red", linewidth=3)
        ax.plot(lower.index, lower.values, color="red", linestyle="--", linewidth=1.5)
        ax.plot(upper.index, upper.values, color="red", linestyle="--", linewidth=1.5)

    ax.set_title("Active Sensing Band Power (log)")
    ax.set_ylabel("Band Power")
    ax.set_ylim(-2, 2)


def _plot_stages(ax, stages: pd.Series, sleep_intervals) -> None:
    for start, end in sleep_intervals:
        ax.axvspan(start, end, color="green", alpha=0.2)
    half_window = pd.Timedelta(minutes=2.5)
    for timestamp, stage in stages.items():
        stage_value = int(stage)
        if stage_value not in SLEEP_STAGE_NAMES:
            continue
        left = mdates.date2num((timestamp - half_window).to_pydatetime())
        right = mdates.date2num((timestamp + half_window).to_pydatetime())
        ax.broken_barh(
            [(left, right - left)],
            (stage_value - 0.35, 0.7),
            facecolors=SLEEP_STAGE_COLORS[stage_value],
            edgecolors="none",
            alpha=0.7,
        )
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels([f"{value}={SLEEP_STAGE_NAMES[value]}" for value in (1, 2, 3, 4)])
    ax.set_ylim(0.5, 4.5)
    ax.set_title("Oura Sleep Stages")
    ax.set_ylabel("Stage")


def _plot_side(side: str, visits, oura_scores, sleep_intervals, output_dir: Path) -> Path:
    side_visits = [(number, payload) for (visit_side, number), payload in visits.items() if visit_side == side]
    side_visits.sort(key=lambda item: item[0])
    if not side_visits:
        raise ValueError(f"No {side} visits found")

    fig, axes = plt.subplots(
        3,
        len(side_visits),
        figsize=(4.6 * len(side_visits), 10),
        gridspec_kw={"height_ratios": [2, 1.5, 1.5]},
        squeeze=False,
    )

    for column, (visit_number, payload) in enumerate(side_visits):
        spectrum = payload["spectra"]
        features = payload["features"]
        ax_spec, ax_active, ax_stages = axes[:, column]
        frequencies = spectrum.columns.to_numpy(dtype=float)
        power = spectrum.to_numpy(dtype=float).T
        log_power = np.log(np.where(power > 0, power, np.nan))
        x_values = mdates.date2num(spectrum.index.to_pydatetime())
        spectrum_cmap = plt.get_cmap("viridis").copy()
        spectrum_cmap.set_bad(color="white")
        ax_spec.set_facecolor("white")
        ax_spec.pcolormesh(
            x_values,
            frequencies,
            np.ma.masked_invalid(log_power),
            shading="gouraud",
            cmap=spectrum_cmap,
            rasterized=True,
            vmin=-5.5,
            vmax=1.0,
        )
        ax_spec.xaxis_date()
        ax_spec.set_xticks([])
        ax_spec.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False, labeltop=False)
        ax_spec.set_ylim(FREQUENCY_MIN, FREQUENCY_MAX)
        ax_spec.set_title(f"{side} {payload['start']:%Y-%m-%d}", fontsize=9)
        ax_spec.set_ylabel("Frequency (Hz)" if column == 0 else "")

        start = spectrum.index.min()
        end = spectrum.index.max()
        stages = _sleep_stage_series(oura_scores, start, end)
        _plot_active_sensing(
            ax_active,
            features.get("active_sensing_band", pd.Series(dtype=float)),
            sleep_intervals,
            stages,
        )
        _plot_stages(ax_stages, stages, sleep_intervals)
        if column:
            ax_active.set_ylabel("")
            ax_stages.set_ylabel("")

        for row_index, axis in enumerate((ax_active, ax_stages)):
            axis.set_xlim(start.floor("h"), end.ceil("10min"))
            axis.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30], tz=LOCAL_TZ))
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=LOCAL_TZ))
            for hour in pd.date_range(start.floor("h"), end.ceil("10min"), freq="1h"):
                axis.axvline(hour, linestyle="--", linewidth=0.7, color="0.65", alpha=0.45, zorder=0)
            axis.tick_params(axis="x", labelrotation=90, labelsize=5, labelbottom=(row_index == 1))
            plt.setp(axis.get_xticklabels(), rotation=90, ha="center")
        ax_stages.set_xlabel("Time")

    fig.suptitle(f"{side} side", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"all_visits_side_by_side_{side}_daytime_only_fullspec.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create one all-visits daytime plot per side.")
    parser.add_argument("--input-dir", type=Path, default=Path("clean/nbu_concat_single_channel_TRBD001"))
    parser.add_argument("--output-dir", type=Path, default=Path("clean/plots2"))
    parser.add_argument("--oura", type=Path, default=Path("oura_scores.pkl"))
    parser.add_argument("--sleep-onset", type=Path, default=Path("trbd001_nbu_sleep_onset.pkl"))
    args = parser.parse_args()

    visits = _read_visit_data(args.input_dir)
    oura_scores = _load_oura_scores(args.oura)
    sleep_intervals = _sleep_intervals(args.sleep_onset)
    outputs = [
        _plot_side(side, visits, oura_scores, sleep_intervals, args.output_dir)
        for side in ("LEFT", "RIGHT")
    ]
    for output in outputs:
        print(f"Saved {output}")
    print(f"Wrote {len(outputs)} side-by-side visit plots to {args.output_dir}")


if __name__ == "__main__":
    main()