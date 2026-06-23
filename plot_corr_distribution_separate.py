from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


INPUT_PATH = Path("out_figures_concat_single_channel/corr_10min_segments.pkl")
OUTPUT_DIR = Path("out_figures_concat_single_channel")
OUTPUT_FIG = OUTPUT_DIR / "corr_distribution_separate_by_visit.png"

TARGET_PAIRS = [
    "active_sensing_band_log__offset",
    "active_sensing_band_log__theta",
    "active_sensing_band_log__alpha",
    "active_sensing_band_log__oura_met",
]
PAIR_LABELS = {
    "active_sensing_band_log__offset": "ASB vs Offset",
    "active_sensing_band_log__theta": "ASB vs Theta",
    "active_sensing_band_log__alpha": "ASB vs Alpha",
    "active_sensing_band_log__oura_met": "ASB vs MET",
}


def _sorted_visits(df: pd.DataFrame) -> list[str]:
    order_df = (
        df[["visit_label", "visit_order", "visit_start_date"]]
        .drop_duplicates()
        .sort_values(["visit_order", "visit_start_date"], kind="mergesort")
    )
    return order_df["visit_label"].tolist()


def _visit_days_from_first(df: pd.DataFrame) -> dict[str, int]:
    order_df = (
        df[["visit_label", "visit_order", "visit_start_date"]]
        .drop_duplicates()
        .sort_values(["visit_order", "visit_start_date"], kind="mergesort")
    )
    if order_df.empty:
        return {}

    first_date = pd.Timestamp(order_df["visit_start_date"].min())
    out: dict[str, int] = {}
    for _, row in order_df.iterrows():
        visit = row["visit_label"]
        dt = pd.Timestamp(row["visit_start_date"])
        out[visit] = int((dt - first_date).days)
    return out


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    df = pd.read_pickle(INPUT_PATH)
    if df.empty:
        raise ValueError(f"Input dataframe is empty: {INPUT_PATH}")

    required_cols = {
        "side",
        "visit_label",
        "visit_order",
        "visit_start_date",
        "pair",
        "correlation",
        "sleep_present",
    }
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["visit_start_date"] = pd.to_datetime(df["visit_start_date"])
    df = df[~df["sleep_present"].astype(bool)].copy()
    df = df[df["pair"].isin(TARGET_PAIRS)].copy()
    df = df[np.isfinite(df["correlation"].to_numpy(dtype=float))]

    if df.empty:
        raise ValueError("No outside-sleep correlation rows available for plotting")

    sides = [side for side in ["LEFT", "RIGHT"] if side in set(df["side"].astype(str))]
    if not sides:
        sides = sorted(df["side"].astype(str).unique())

    visits = _sorted_visits(df)
    day_offsets = _visit_days_from_first(df)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(
        nrows=max(len(visits), 1),
        ncols=max(len(sides), 1),
        figsize=(8.5 * max(len(sides), 1), 5.2 * max(len(visits), 1)),
        sharey=True,
        squeeze=False,
    )

    for row_idx, visit in enumerate(visits):
        for col_idx, side in enumerate(sides):
            ax = axes[row_idx, col_idx]
            sub = df[(df["visit_label"] == visit) & (df["side"].astype(str) == side)].copy()

            if sub.empty:
                ax.set_title(f"{side} | {visit} (no data)")
                ax.set_axis_off()
                continue

            sub["pair"] = pd.Categorical(sub["pair"], categories=TARGET_PAIRS, ordered=True)

            sns.boxplot(
                data=sub,
                x="pair",
                y="correlation",
                order=TARGET_PAIRS,
                color="lightgray",
                showfliers=False,
                showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "crimson", "markeredgecolor": "black", "markersize": 5},
                ax=ax,
            )
            sns.swarmplot(
                data=sub,
                x="pair",
                y="correlation",
                order=TARGET_PAIRS,
                size=2.8,
                alpha=0.55,
                color="tab:blue",
                ax=ax,
            )

            labels = [PAIR_LABELS[p] for p in TARGET_PAIRS]
            ax.set_xticks(range(len(TARGET_PAIRS)))
            ax.set_xticklabels(labels, rotation=25, ha="right")
            ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
            ax.set_xlabel("Correlation Measure")
            ax.set_ylabel("Correlation")
            ax.set_title(f"{side} | {visit} (+{day_offsets.get(visit, 0)}d)")

    fig.suptitle("Outside-sleep correlations by visit (rows) and side (columns)")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG, dpi=220)
    plt.close(fig)

    print(f"Saved figure: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
