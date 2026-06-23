from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd


INPUT_DIR = Path("out_figures_concat_single_channel_tight")
INPUT_PKL = INPUT_DIR / "non_sleep_10s_spectra_with_active_sensing.pkl"
INPUT_CSV = INPUT_DIR / "non_sleep_10s_spectra_with_active_sensing.csv"
OUTPUT_FIG = INPUT_DIR / "avg_psd_non_sleep_by_visit.png"
MADRS_CSV_PATH = Path("TRBD Programming Videos(TRBD001).csv")

# Very thin traces for individual 10-second spectra.
INDIVIDUAL_LINEWIDTH = 0.12
INDIVIDUAL_ALPHA = 0.04
# Thick trace for per-visit average spectrum.
AVERAGE_LINEWIDTH = 2.6
SEM_BAND_ALPHA = 0.2

# If True, plot only mean +/- 1 SEM per visit (no individual 10s traces).
PLOT_MEAN_PM_STD_ONLY = True

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


def _sorted_visits(df: pd.DataFrame) -> list[str]:
    if "exact_datetime" in df.columns and df["exact_datetime"].notna().any():
        visit_order = (
            df.groupby("nbu_visit_date")["exact_datetime"]
            .min()
            .sort_values()
            .index.astype(str)
            .tolist()
        )
        return visit_order

    return sorted(df["nbu_visit_date"].astype(str).unique().tolist())


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
    df["side"] = df["side"].astype(str)

    sides = sorted(df["side"].dropna().unique().tolist())
    visits = _sorted_visits(df)
    if not visits:
        raise ValueError("No visit groups found in dataframe.")

    madrs_by_date = _load_madrs_by_date_from_csv(MADRS_CSV_PATH)
    visit_madrs = _visit_madrs_map(visits, madrs_by_date)
    visit_colors = _madrs_colors(visits, visit_madrs)
    fig, axes = plt.subplots(
        len(sides),
        1,
        figsize=(6, 4.8 * max(len(sides), 1)),
        sharex=True,
        squeeze=False,
    )

    for row_idx, side in enumerate(sides):
        ax = axes[row_idx, 0]
        side_df = df[df["side"] == side]

        for visit in visits:
            visit_df = side_df[side_df["nbu_visit_date"] == visit]
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

            avg = np.nanmean(y_log, axis=0)
            n_valid = np.sum(~np.isnan(y_log), axis=0)
            std = np.nanstd(y_log, axis=0)
            sem = np.divide(
                std,
                np.sqrt(n_valid),
                out=np.full_like(std, np.nan),
                where=n_valid > 0,
            )
            ax.fill_between(
                freqs,
                avg - sem,
                avg + sem,
                color=color,
                alpha=SEM_BAND_ALPHA,
                linewidth=0,
            )
            ax.plot(
                freqs,
                avg,
                color=color,
                linewidth=AVERAGE_LINEWIDTH,
                alpha=0.98,
                label=(
                    f"{visit} | MADRS={visit_madrs[visit]:.1f} (n={len(visit_df)})"
                    if np.isfinite(visit_madrs[visit])
                    else f"{visit} | MADRS=NA (n={len(visit_df)})"
                ),
            )

        ax.set_title(f"{side} side")
        ax.set_ylabel("Power")
        ax.grid(alpha=0.25)
        ax.legend(title="NBU visit", fontsize=8)

    axes[-1, 0].set_xlabel("Frequency (Hz)")
    if PLOT_MEAN_PM_STD_ONLY:
        fig.suptitle("Non-sleep PSD by visit date, color-coded by MADRS: mean +/- 1 SEM")
    else:
        fig.suptitle("Non-sleep PSD by visit date, color-coded by MADRS")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUTPUT_FIG, dpi=220)
    plt.close(fig)

    print(f"Saved figure: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
