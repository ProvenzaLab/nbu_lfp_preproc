from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


INPUT_DIR = Path("out_figures_concat_single_channel_tight")
INPUT_PKL = INPUT_DIR / "non_sleep_10s_spectra_with_active_sensing.pkl"
INPUT_CSV = INPUT_DIR / "non_sleep_10s_spectra_with_active_sensing.csv"
OUTPUT_FIG = INPUT_DIR / "avg_psd_non_sleep_by_visit_colored_by_time.pdf"
MADRS_CSV_PATH = Path("TRBD Programming Videos(TRBD001).csv")

# Very thin traces for individual 10-second spectra.
INDIVIDUAL_LINEWIDTH = 0.12
INDIVIDUAL_ALPHA = 0.04
# Thick trace for per-visit average spectrum.
AVERAGE_LINEWIDTH = 2.6

# If True, plot only mean per visit (no individual 10s traces).
PLOT_MEAN_PM_STD_ONLY = True

# Color coding scheme: either "madrs" or "time"
# "madrs": color by MADRS score from MADRS_CSV_PATH
# "time": color by days since first visit
COLOR_BY = "time"

# Optional manual overrides/additions for visit date -> MADRS score.
# Expected date format is YYYY-MM-DD to match nbu_visit_date.
MADRS_BY_DATE: dict[str, float] = {
}


def _load_non_sleep_dataframe() -> pd.DataFrame:
    if INPUT_PKL.exists():
        df = pd.read_pickle(INPUT_PKL)
    elif INPUT_CSV.exists():
        df = pd.read_csv(INPUT_CSV)
    else:
        raise FileNotFoundError(
            f"Could not find input dataframe at {INPUT_PKL} or {INPUT_CSV}"
        )

    if "exact_datetime" in df.columns:
        df["exact_datetime"] = pd.to_datetime(df["exact_datetime"], errors="coerce")

    if "nbu_visit_date" not in df.columns:
        if "visit_label" in df.columns:
            df["nbu_visit_date"] = df["visit_label"].astype(str)
        else:
            df["nbu_visit_date"] = "visit_unknown"

    if "visit_label" not in df.columns:
        if "nbu_visit_date" in df.columns:
            df["visit_label"] = df["nbu_visit_date"].astype(str)
        else:
            df["visit_label"] = "visit_unknown"

    if "side" not in df.columns:
        df["side"] = "ALL"

    return df


def _extract_spectrum_columns(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    spec_cols = [col for col in df.columns if str(col).startswith("spectrum_")]
    if not spec_cols:
        raise ValueError("No spectrum columns found (expected columns like 'spectrum_12.5').")

    col_freq_pairs = []
    for col in spec_cols:
        freq_str = str(col).replace("spectrum_", "", 1)
        try:
            freq_val = float(freq_str)
        except ValueError:
            continue
        col_freq_pairs.append((col, freq_val))

    if not col_freq_pairs:
        raise ValueError("Spectrum columns were found, but none had numeric frequencies.")

    col_freq_pairs.sort(key=lambda x: x[1])
    sorted_cols = [pair[0] for pair in col_freq_pairs]
    sorted_freqs = np.array([pair[1] for pair in col_freq_pairs], dtype=float)
    return sorted_cols, sorted_freqs


def _sorted_visits(df: pd.DataFrame, visit_col: str) -> list[str]:
    if "exact_datetime" in df.columns and df["exact_datetime"].notna().any():
        visit_order = (
            df.groupby(visit_col)["exact_datetime"]
            .min()
            .sort_values()
            .index.astype(str)
            .tolist()
        )
        return visit_order

    return sorted(df[visit_col].astype(str).unique().tolist())


def _visit_time_since_first(
    df: pd.DataFrame,
    visits: list[str],
    visit_col: str,
) -> dict[str, float]:
    """Calculate days since first visit for each visit."""
    time_map: dict[str, float] = {}
    
    if "exact_datetime" in df.columns and df["exact_datetime"].notna().any():
        # Use exact datetime if available
        visit_datetimes = {}
        for visit in visits:
            visit_df = df[df[visit_col] == visit]
            if not visit_df.empty:
                visit_datetimes[visit] = visit_df["exact_datetime"].min()
        
        if visit_datetimes:
            first_datetime = min(visit_datetimes.values())
            for visit, visit_dt in visit_datetimes.items():
                time_map[visit] = (visit_dt - first_datetime).total_seconds() / (24 * 3600)
        else:
            # Fallback to visit order
            for i, visit in enumerate(visits):
                time_map[visit] = float(i)
    else:
        # Use visit order as fallback
        for i, visit in enumerate(visits):
            time_map[visit] = float(i)
    
    return time_map


def _time_colors(visits: list[str], time_map: dict[str, float]) -> dict[str, tuple[float, float, float, float]]:
    """Generate unique colors for each visit while preserving time order."""
    cmap = plt.get_cmap("turbo")
    if not visits:
        return {}

    # Sort by time (then label) so the color ramp still follows progression in time.
    ordered_visits = sorted(
        visits,
        key=lambda v: (
            time_map.get(v, np.inf) if np.isfinite(time_map.get(v, np.nan)) else np.inf,
            str(v),
        ),
    )

    # Sample evenly across the colormap to guarantee distinct colors per visit.
    if len(ordered_visits) == 1:
        positions = [0.5]
    else:
        positions = np.linspace(0.05, 0.95, len(ordered_visits))

    colors = {
        visit: cmap(float(pos))
        for visit, pos in zip(ordered_visits, positions)
    }

    return {visit: colors[visit] for visit in visits}


def _load_madrs_by_date_from_csv(csv_path: Path) -> dict[str, float]:
    if not csv_path.exists():
        return {}

    madrs_df = pd.read_csv(csv_path)
    if "Date" not in madrs_df.columns or "MADRS" not in madrs_df.columns:
        return {}

    madrs_df = madrs_df[["Date", "MADRS"]].copy()
    madrs_df["Date"] = pd.to_datetime(madrs_df["Date"], errors="coerce")
    madrs_df["MADRS"] = pd.to_numeric(madrs_df["MADRS"], errors="coerce")
    madrs_df = madrs_df.dropna(subset=["Date", "MADRS"])
    if madrs_df.empty:
        return {}

    madrs_df["date_str"] = madrs_df["Date"].dt.strftime("%Y-%m-%d")
    # If multiple rows exist for one date, use the average score for that date.
    madrs_by_date = madrs_df.groupby("date_str")["MADRS"].mean().to_dict()
    return {str(k): float(v) for k, v in madrs_by_date.items()}


def _visit_madrs_map(visits: list[str], madrs_by_date: dict[str, float]) -> dict[str, float]:
    madrs_map: dict[str, float] = {}
    for visit in visits:
        visit_key = str(visit)
        score = madrs_by_date.get(visit_key)
        if score is None:
            score = MADRS_BY_DATE.get(visit_key)
        if score is None:
            madrs_map[visit] = np.nan
        else:
            madrs_map[visit] = float(score)
    return madrs_map


def _madrs_colors(visits: list[str], madrs_map: dict[str, float]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("viridis_r")
    known_scores = [score for score in madrs_map.values() if np.isfinite(score)]
    if known_scores:
        vmin = min(known_scores)
        vmax = max(known_scores)
        if vmin == vmax:
            vmax = vmin + 1.0
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        return {
            visit: cmap(norm(madrs_map[visit])) if np.isfinite(madrs_map[visit]) else (0.55, 0.55, 0.55, 1.0)
            for visit in visits
        }

    # If no mapping is provided, keep a deterministic viridis ramp by visit order.
    return {
        visit: cmap(i / max(len(visits) - 1, 1))
        for i, visit in enumerate(visits)
    }


def main() -> None:
    df = _load_non_sleep_dataframe()
    spec_cols, freqs = _extract_spectrum_columns(df)

    df = df.copy()
    df[spec_cols] = df[spec_cols].apply(pd.to_numeric, errors="coerce")
    df["nbu_visit_date"] = df["nbu_visit_date"].astype(str)
    df["visit_label"] = df["visit_label"].astype(str)
    df["side"] = df["side"].astype(str)

    sides = sorted(df["side"].dropna().unique().tolist())
    visit_col = "visit_label" if COLOR_BY.lower() == "time" else "nbu_visit_date"
    visits = _sorted_visits(df, visit_col=visit_col)
    # set the 
    if not visits:
        raise ValueError("No visit groups found in dataframe.")

    # Choose coloring scheme based on configuration
    if COLOR_BY.lower() == "madrs":
        madrs_by_date = _load_madrs_by_date_from_csv(MADRS_CSV_PATH)
        visit_madrs = _visit_madrs_map(visits, madrs_by_date)
        visit_colors = _madrs_colors(visits, visit_madrs)
        color_values = visit_madrs
    else:  # default to "time"
        visit_times = _visit_time_since_first(df, visits, visit_col=visit_col)
        visit_colors = _time_colors(visits, visit_times)
        color_values = visit_times
    fig, axes = plt.subplots(
        len(sides),
        1,
        figsize=(6, 4.8 * max(len(sides), 1)),
        sharex=True,
        squeeze=False,
    )

    avg_psd_data = {}

    for row_idx, side in enumerate(sides):
        ax = axes[row_idx, 0]
        side_df = df[df["side"] == side]

        for visit in visits:
            visit_df = side_df[side_df[visit_col] == visit]
            if visit_df.empty:
                continue

            y = visit_df[spec_cols].to_numpy(dtype=float)
            color = visit_colors[visit]
            y_log = np.log(np.where(y > 0, y, np.nan))

            if not PLOT_MEAN_PM_STD_ONLY:
                for row in y_log:
                    if np.isnan(row).all():
                        continue
                    ax.plot(
                        freqs,
                        row,
                        color=color,
                        linewidth=INDIVIDUAL_LINEWIDTH,
                        alpha=INDIVIDUAL_ALPHA,
                    )

            avg = np.nanmean(y_log, axis=0) # manmean
            avg_psd_data[f"{side}_{visit}"] = avg
            if COLOR_BY.lower() == "madrs":
                label_text = (
                    f"{visit} | MADRS={color_values[visit]:.1f} (n={len(visit_df)})"
                    if np.isfinite(color_values[visit])
                    else f"{visit} | MADRS=NA (n={len(visit_df)})"
                )
            else:
                label_text = f"{visit} | +{color_values[visit]:.1f}d (n={len(visit_df)})"
            
            ax.plot(
                freqs,
                avg,
                color=color,
                linewidth=AVERAGE_LINEWIDTH,
                alpha=0.98,
                label=label_text,
            )

        ax.set_title(f"{side} side")
        ax.set_ylabel("Power")
        ax.grid(alpha=0.25)
        ax.legend(title="NBU visit", fontsize=8)

        # make avg_psd_data a DataFrame and save it to CSV
    avg_psd_df = []
    for key, avg in avg_psd_data.items():
        side, visit = key.split("_", 1)
        avg_psd_df.append({"side": side, "visit": visit, **{f"spectrum_{freq}": val for freq, val in zip(freqs, avg)}})
    avg_psd_df = pd.DataFrame(avg_psd_df)
    avg_psd_csv_path = INPUT_DIR / "avg_psd_non_sleep_by_visit.csv"
    avg_psd_df.to_csv(avg_psd_csv_path, index=False)
    print(f"Saved average PSD data to CSV: {avg_psd_csv_path}")

    axes[-1, 0].set_xlabel("Frequency (Hz)")
    if PLOT_MEAN_PM_STD_ONLY:
        color_label = "MADRS" if COLOR_BY.lower() == "madrs" else "time-since-first-visit"
        fig.suptitle(f"Non-sleep PSD by visit date, color-coded by {color_label}: mean only")
    else:
        color_label = "MADRS" if COLOR_BY.lower() == "madrs" else "time-since-first-visit"
        fig.suptitle(f"Non-sleep PSD by visit date, color-coded by {color_label}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUTPUT_FIG)
    plt.close(fig)

    print(f"Saved figure: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
