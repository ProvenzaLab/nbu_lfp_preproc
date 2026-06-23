from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


INPUT_PATH = Path("out_figures_concat_single_channel/corr_10min_segments.pkl")
OUTPUT_DIR = Path("out_figures_concat_single_channel")
OUTPUT_FIG = OUTPUT_DIR / "corr_distribution_hue_visits.png"

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


def _visit_order_and_labels(df: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    order_df = (
        df[["visit_label", "visit_order", "visit_start_date"]]
        .drop_duplicates()
        .sort_values(["visit_order", "visit_start_date"], kind="mergesort")
    )
    if order_df.empty:
        return [], {}

    first_date = pd.Timestamp(order_df["visit_start_date"].min())
    visit_order = order_df["visit_label"].tolist()
    label_map: dict[str, str] = {}
    for _, row in order_df.iterrows():
        visit = row["visit_label"]
        day_delta = int((pd.Timestamp(row["visit_start_date"]) - first_date).days)
        label_map[visit] = f"{visit} (+{day_delta}d)"
    return visit_order, label_map


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

    df["measure"] = df["pair"].map(PAIR_LABELS)
    measure_order = [PAIR_LABELS[p] for p in TARGET_PAIRS]

    visit_order, visit_label_map = _visit_order_and_labels(df)
    if not visit_order:
        raise ValueError("No visits available after filtering")

    df["visit_label"] = pd.Categorical(df["visit_label"], categories=visit_order, ordered=True)

    sides = [side for side in ["LEFT", "RIGHT"] if side in set(df["side"].astype(str))]
    if not sides:
        sides = sorted(df["side"].astype(str).unique())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(
        nrows=1,
        ncols=max(len(sides), 1),
        figsize=(9.5 * max(len(sides), 1), 9.5),
        sharey=True,
        squeeze=False,
    )

    for col_idx, side in enumerate(sides):
        ax = axes[0, col_idx]
        side_df = df[df["side"].astype(str) == side].copy()

        if side_df.empty:
            ax.set_title(f"{side} (no data)")
            ax.set_axis_off()
            continue

        sns.boxplot(
            data=side_df,
            x="measure",
            y="correlation",
            hue="visit_label",
            order=measure_order,
            hue_order=visit_order,
            showfliers=False,
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "crimson", "markeredgecolor": "black", "markersize": 5},
            ax=ax,
        )
        sns.swarmplot(
            data=side_df,
            x="measure",
            y="correlation",
            hue="visit_label",
            order=measure_order,
            hue_order=visit_order,
            dodge=True,
            size=2.3,
            alpha=0.5,
            linewidth=0,
            ax=ax,
        )

        handles, labels = ax.get_legend_handles_labels()
        unique = []
        seen = set()
        for h, l in zip(handles, labels):
            if l in seen:
                continue
            seen.add(l)
            unique.append((h, l))

        if unique:
            legend_handles = [h for h, _ in unique]
            legend_labels = [visit_label_map.get(l, str(l)) for _, l in unique]
            ax.legend(legend_handles, legend_labels, title="Visit", loc="upper right", fontsize=8, title_fontsize=9)

        ax.set_title(f"{side} | outside sleep")
        ax.set_xlabel("Correlation Measure")
        ax.set_ylabel("Correlation")
        ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
        ax.tick_params(axis="x", rotation=22)

    fig.suptitle("Correlation distributions by measure, hue=visit")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG, dpi=220)
    plt.close(fig)

    print(f"Saved figure: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
