from __future__ import annotations

import argparse
import pickle
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import specparam
from joblib import Parallel, delayed


LOCAL_TZ = "America/Chicago"
FIRST_SPLIT_VISIT_DATE = date(2025, 6, 3)
STIM_SPLIT_TIMESTAMP = pd.Timestamp("2025-06-04 22:56:14", tz=LOCAL_TZ)
STIM_OFF_LABEL = "2025-06-03 Stim off"
STIM_ON_LABEL = "2025-06-04 Stim ON"
FIT_RANGE = (1.0, 100.0)
PEAK_WIDTH_LIMITS = (3.0, 5.0)
USE_BAND_POWER_SUM = True
BAND_METRIC_SUFFIX = "band_power_sum" if USE_BAND_POWER_SUM else "peak_height"
MAX_N_PEAKS = 0 if USE_BAND_POWER_SUM else 7
RESPONSE_THRESHOLD = 0.5
SPECTRUM_FILE_RE = re.compile(
    r"^spectra_highres_spectra_.+?_(?P<side>LEFT|RIGHT)_"
    r"visit(?P<visit>\d+)_(?P<start>\d{4}-\d{2}-\d{2})_to_"
    r"(?P<end>\d{4}-\d{2}-\d{2})(?:_(?P<split>before_split|after_split))?$"
)

PEAK_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "low_beta": (13.0, 20.0),
    "high_beta": (20.0, 30.0),
    "low_gamma": (30.0, 60.0),
    "high_gamma": (60.0, 100.0),
}


def _local_index(index: pd.Index) -> pd.DatetimeIndex:
    result = pd.DatetimeIndex(pd.to_datetime(index))
    if result.tz is None:
        return result.tz_localize(LOCAL_TZ)
    return result.tz_convert(LOCAL_TZ)


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


def _load_sleep_onset_intervals(path: Path) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
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


def _oura_sleep_stages(
    oura_scores: dict,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    pieces = []
    score_dates = pd.date_range(
        start.normalize() - pd.Timedelta(days=1),
        end.normalize() + pd.Timedelta(days=1),
        freq="D",
    ).date
    for score_date in score_dates:
        entry = oura_scores.get(f"TRBD001_{score_date:%Y-%m-%d}", {})
        stages = entry.get("sleep_phases") if isinstance(entry, dict) else None
        if isinstance(stages, pd.DataFrame):
            column = "sleep_phase_5_min" if "sleep_phase_5_min" in stages else stages.columns[0]
            stages = stages[column]
        if isinstance(stages, pd.Series) and not stages.empty:
            stages = stages.copy()
            stages.index = _local_index(stages.index)
            pieces.append(stages)

    if not pieces:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz=LOCAL_TZ))

    result = pd.concat(pieces).sort_index()
    result = result.loc[~result.index.duplicated(keep="first")]
    return pd.to_numeric(result, errors="coerce").loc[
        (result.index >= start) & (result.index <= end)
    ].dropna()


def _stage_exclusion_intervals(stages: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    padding = pd.Timedelta(minutes=5)
    selected = [
        (timestamp - padding, timestamp + padding)
        for timestamp, value in stages.items()
        if int(value) in (1, 2, 3)
    ]
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


def _overlap_mask(index: pd.DatetimeIndex, intervals) -> np.ndarray:
    mask = np.zeros(len(index), dtype=bool)
    for start, end in intervals:
        mask |= (index >= start) & (index <= end)
    return mask


def _load_madrs(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)
    if not {"Date", "MADRS"}.issubset(frame.columns):
        raise ValueError(f"{path} must contain Date and MADRS columns")
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["MADRS"] = pd.to_numeric(frame["MADRS"], errors="coerce")
    frame = frame.dropna(subset=["Date", "MADRS"])
    frame["date"] = frame["Date"].dt.strftime("%Y-%m-%d")
    return frame.groupby("date")["MADRS"].mean().astype(float).to_dict()


def _madrs_lookup_date(day_label: str) -> str:
    if day_label == STIM_OFF_LABEL:
        return "2025-06-03"
    if day_label == STIM_ON_LABEL:
        return "2025-06-04"
    return day_label


def _mean_peak_height(fm: specparam.SpectralModel, band: tuple[float, float]) -> float:
    peak_params = np.asarray(fm.results.get_params("peak"), dtype=float).reshape(-1, 3)
    if peak_params.size == 0:
        return np.nan
    center_freqs = peak_params[:, 0]
    heights = peak_params[:, 1]
    in_band = (center_freqs >= band[0]) & (center_freqs <= band[1])
    return float(np.max(heights[in_band])) if in_band.any() else np.nan


def _band_power_sum(
    freqs: np.ndarray,
    power_spectrum: np.ndarray,
    fm: specparam.SpectralModel,
    band: tuple[float, float],
) -> float:
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    aperiodic_power = np.asarray(
        fm.results.model.get_component("aperiodic", space="linear"),
        dtype=float,
    )
    corrected_power = power_spectrum - aperiodic_power
    values = corrected_power[band_mask]
    return float(np.nansum(values)) if np.isfinite(values).any() else np.nan


def _fit_psd_chunk(
    spectrum: pd.DataFrame,
    fit_freqs: np.ndarray,
    fit_mask: np.ndarray,
    side: str,
    visit: int,
    day: str,
    madrs: float,
) -> dict:
    raw_curve_sum = np.zeros(len(fit_freqs), dtype=float)
    flattened_curve_sum = np.zeros(len(fit_freqs), dtype=float)
    window_features = []
    skipped_invalid = 0
    failed_fits = 0

    for timestamp, psd_row in spectrum.iterrows():
        fit_psd = psd_row.to_numpy(dtype=float)[fit_mask]
        if not np.isfinite(fit_psd).all() or (fit_psd <= 0).any():
            skipped_invalid += 1
            continue

        try:
            fm = specparam.SpectralModel(
                verbose=False,
                peak_width_limits=list(PEAK_WIDTH_LIMITS),
                max_n_peaks=MAX_N_PEAKS,
            )
            fm.fit(fit_freqs, fit_psd, list(FIT_RANGE))

            raw_log_power = np.asarray(fm.data.power_spectrum, dtype=float)
            aperiodic_log_power = np.asarray(
                fm.results.model.get_component("aperiodic", space="log"),
                dtype=float,
            )
            raw_curve_sum += raw_log_power
            flattened_curve_sum += raw_log_power - aperiodic_log_power

            aperiodic_params = np.asarray(fm.results.get_params("aperiodic"), dtype=float)
            features = {
                "datetime": timestamp.isoformat(),
                "date": day,
                "hemisphere": side.lower(),
                "visit": visit,
                "madrs": madrs,
                "aperiodic_offset": float(aperiodic_params[0]),
                "aperiodic_exponent": float(aperiodic_params[1]),
            }
            for band_name, band_range in PEAK_BANDS.items():
                if USE_BAND_POWER_SUM:
                    features[f"{band_name}_{BAND_METRIC_SUFFIX}"] = _band_power_sum(
                        fit_freqs,
                        fit_psd,
                        fm,
                        band_range,
                    )
                else:
                    features[f"{band_name}_{BAND_METRIC_SUFFIX}"] = _mean_peak_height(
                        fm,
                        band_range,
                    )
            window_features.append(features)
        except Exception:
            failed_fits += 1

    return {
        "key": (side, visit, day),
        "raw_curve_sum": raw_curve_sum,
        "flattened_curve_sum": flattened_curve_sum,
        "window_features": window_features,
        "n_attempted": len(spectrum),
        "n_successful": len(window_features),
        "n_skipped_invalid": skipped_invalid,
        "n_failed": failed_fits,
    }


def _summarize_visit_fits(
    side: str,
    visit: int,
    day: str,
    madrs: float,
    fit_freqs: np.ndarray,
    n_psd_windows: int,
    chunks: list[dict],
) -> tuple[dict, dict, list[dict]]:
    successful_fits = sum(chunk["n_successful"] for chunk in chunks)
    if successful_fits == 0:
        raise ValueError(f"{side} {day}: no valid 10-second PSD fits")

    raw_curve_sum = sum((chunk["raw_curve_sum"] for chunk in chunks), np.zeros(len(fit_freqs)))
    flattened_curve_sum = sum(
        (chunk["flattened_curve_sum"] for chunk in chunks),
        np.zeros(len(fit_freqs)),
    )
    window_features = [row for chunk in chunks for row in chunk["window_features"]]
    fit_curve = {
        "date": day,
        "hemisphere": side.lower(),
        "visit": visit,
        "madrs": madrs,
        "n_psd_windows": n_psd_windows,
        "n_successful_fits": successful_fits,
        "freqs": fit_freqs,
        "raw_log_power": raw_curve_sum / successful_fits,
        "aperiodic_removed_log_power": flattened_curve_sum / successful_fits,
    }

    window_frame = pd.DataFrame(window_features)
    daily_features = {
        "date": day,
        "hemisphere": side.lower(),
        "visit": visit,
        "madrs": madrs,
        "n_psd_windows": n_psd_windows,
        "n_successful_fits": successful_fits,
        "n_skipped_invalid_psds": sum(chunk["n_skipped_invalid"] for chunk in chunks),
        "n_failed_fits": sum(chunk["n_failed"] for chunk in chunks),
    }
    feature_columns = [
        "aperiodic_offset",
        "aperiodic_exponent",
        *(f"{band_name}_{BAND_METRIC_SUFFIX}" for band_name in PEAK_BANDS),
    ]
    for column in feature_columns:
        daily_features[column] = window_frame[column].mean()
        if column.endswith("_peak_height"):
            daily_features[column.replace("_peak_height", "_n_windows_with_peak")] = int(
                window_frame[column].notna().sum()
            )
    return fit_curve, daily_features, window_features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit Specparam models to mean daytime PSDs for each visit and side."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("clean/nbu_concat_single_channel_TRBD001"))
    parser.add_argument("--oura", type=Path, default=Path("oura_scores.pkl"))
    parser.add_argument("--sleep-onset", type=Path, default=Path("trbd001_nbu_sleep_onset.pkl"))
    parser.add_argument("--madrs", type=Path, default=Path("TRBD Programming Videos(TRBD001).csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("clean/analysis_results"))
    parser.add_argument("--n-jobs", type=int, default=15, help="Parallel fit workers; use 1 for serial execution.")
    parser.add_argument(
        "--max-psds-per-visit",
        type=int,
        default=None,
        help="Optional per-visit fit limit for a quick smoke test.",
    )
    args = parser.parse_args()

    with args.oura.open("rb") as file:
        oura_scores = pickle.load(file)
    sleep_onset_intervals = _load_sleep_onset_intervals(args.sleep_onset)
    madrs_by_date = _load_madrs(args.madrs)

    input_files = sorted(args.input_dir.glob("spectra_highres_*.pkl"))
    if not input_files:
        raise FileNotFoundError(f"No visit spectra pickles found in {args.input_dir}")

    grouped_spectra: dict[tuple[str, int, str], list[pd.DataFrame]] = {}
    for path in input_files:
        match = SPECTRUM_FILE_RE.match(path.stem)
        if match is None:
            continue
        with path.open("rb") as file:
            payload = pickle.load(file)
        spectrum = payload.get("df_spec")
        if not isinstance(spectrum, pd.DataFrame) or spectrum.empty:
            continue

        spectrum = spectrum.copy()
        spectrum.index = _local_index(spectrum.index)
        spectrum = spectrum.sort_index()
        spectrum.columns = pd.to_numeric(spectrum.columns, errors="coerce")
        spectrum = spectrum.loc[:, spectrum.columns.notna()]
        spectrum = spectrum.loc[:, (spectrum.columns >= 0) & (spectrum.columns <= 100)]
        visit_date = pd.Timestamp(match.group("start"))
        if visit_date.date() == FIRST_SPLIT_VISIT_DATE:
            split_part = match.group("split")
            if split_part == "before_split":
                segments = [(STIM_OFF_LABEL, spectrum.loc[spectrum.index <= STIM_SPLIT_TIMESTAMP])]
            elif split_part == "after_split":
                segments = [(STIM_ON_LABEL, spectrum.loc[spectrum.index > STIM_SPLIT_TIMESTAMP])]
            else:
                segments = [
                    (STIM_OFF_LABEL, spectrum.loc[spectrum.index <= STIM_SPLIT_TIMESTAMP]),
                    (STIM_ON_LABEL, spectrum.loc[spectrum.index > STIM_SPLIT_TIMESTAMP]),
                ]
        else:
            segments = [(visit_date.strftime("%Y-%m-%d"), spectrum)]

        for day_label, segment in segments:
            if segment.empty:
                continue
            segment = segment.apply(pd.to_numeric, errors="coerce").resample("10s").mean()
            if segment.empty:
                continue

            stages = _oura_sleep_stages(oura_scores, segment.index.min(), segment.index.max())
            exclusion_intervals = list(sleep_onset_intervals) + _stage_exclusion_intervals(stages)
            if visit_date.date() == date(2025, 6, 10):
                exclusion_intervals.append(
                    (
                        pd.Timestamp("2025-06-10 15:30", tz=LOCAL_TZ),
                        pd.Timestamp("2025-06-10 17:00", tz=LOCAL_TZ),
                    )
                )
            segment = segment.loc[~_overlap_mask(segment.index, exclusion_intervals)]
            if segment.empty:
                print(f"Skipping {path.name} ({day_label}): no daytime spectra remain after sleep exclusions")
                continue

            key = (match.group("side"), int(match.group("visit")), day_label)
            grouped_spectra.setdefault(key, []).append(segment)

    fit_tasks = []
    visit_inputs = {}
    for (side, visit, day), frames in sorted(grouped_spectra.items()):
        combined = pd.concat(frames).sort_index()
        combined = combined.loc[~combined.index.duplicated(keep="first")]
        frequencies = pd.to_numeric(combined.columns, errors="coerce").to_numpy(dtype=float)
        fit_mask = (frequencies >= FIT_RANGE[0]) & (frequencies <= FIT_RANGE[1])
        fit_freqs = frequencies[fit_mask]
        if len(fit_freqs) < 3 or not np.allclose(np.diff(fit_freqs), np.diff(fit_freqs)[0]):
            raise ValueError(f"{side} {day}: fit frequencies are not a complete regular grid")

        key = (side, visit, day)
        visit_inputs[key] = {
            "madrs": madrs_by_date.get(_madrs_lookup_date(day), np.nan),
            "fit_freqs": fit_freqs,
            "n_psd_windows": len(combined),
        }
        if args.max_psds_per_visit is None:
            selected = combined
        else:
            valid_rows = combined.loc[:, fit_mask].notna().all(axis=1)
            valid_rows &= combined.loc[:, fit_mask].gt(0).all(axis=1)
            selected = combined.loc[valid_rows].iloc[:args.max_psds_per_visit]

        for chunk_start in range(0, len(selected), 128):
            chunk = selected.iloc[chunk_start : chunk_start + 128]
            fit_tasks.append(
                delayed(_fit_psd_chunk)(
                    chunk,
                    fit_freqs,
                    fit_mask,
                    side,
                    visit,
                    day,
                    madrs_by_date.get(_madrs_lookup_date(day), np.nan),
                )
            )

    results = Parallel(n_jobs=args.n_jobs, batch_size=1)(fit_tasks)
    chunks_by_visit: dict[tuple[str, int, str], list[dict]] = {}
    for result in results:
        chunks_by_visit.setdefault(result["key"], []).append(result)

    fit_curves = []
    feature_rows = []
    window_feature_rows = []
    for (side, visit, day), inputs in sorted(visit_inputs.items()):
        visit_chunks = chunks_by_visit.get((side, visit, day), [])
        fit_curve, features, window_features = _summarize_visit_fits(
            side=side,
            visit=visit,
            day=day,
            madrs=inputs["madrs"],
            fit_freqs=inputs["fit_freqs"],
            n_psd_windows=inputs["n_psd_windows"],
            chunks=visit_chunks,
        )
        fit_curves.append(fit_curve)
        feature_rows.append(features)
        window_feature_rows.extend(window_features)
        print(
            f"Fitted {fit_curve['hemisphere'].upper()} visit {fit_curve['visit']} "
            f"({fit_curve['date']}): {fit_curve['n_successful_fits']} fits; "
            f"{features['n_skipped_invalid_psds']} invalid PSD windows skipped; "
            f"{features['n_failed_fits']} model errors"
        )

    if not feature_rows:
        raise ValueError("No daytime PSDs were available to fit")

    features = pd.DataFrame(feature_rows).sort_values(["date", "hemisphere"]).reset_index(drop=True)
    baseline_madrs = features["madrs"].max()
    if pd.notna(baseline_madrs) and baseline_madrs > 0:
        reduction = (baseline_madrs - features["madrs"]) / baseline_madrs
        features["responder"] = reduction > RESPONSE_THRESHOLD
    else:
        features["responder"] = False
    responder_by_date = features.drop_duplicates("date").set_index("date")["responder"]
    window_features = pd.DataFrame(window_feature_rows)
    window_features["responder"] = window_features["date"].map(responder_by_date).fillna(False).astype(bool)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.output_dir / "daytime_specparam_features.csv"
    window_feature_path = args.output_dir / "daytime_specparam_10s_features.csv"
    curve_path = args.output_dir / "daytime_specparam_fits.pkl"
    features.to_csv(feature_path, index=False)
    window_features.to_csv(window_feature_path, index=False)
    with curve_path.open("wb") as file:
        pickle.dump(fit_curves, file)
    print(f"Saved features: {feature_path}")
    print(f"Saved 10-second features: {window_feature_path}")
    print(f"Saved fit curves: {curve_path}")


if __name__ == "__main__":
    main()