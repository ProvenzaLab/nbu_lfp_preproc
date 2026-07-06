from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


INPUT_PATH = Path("out_figures_concat_single_channel_tight_TRBD001/corr_10min_segments.pkl")
OUTPUT_DIR = Path("out_figures_concat_single_channel_tight_TRBD001")

OUTPUT_PVALS = OUTPUT_DIR / "corr_distribution_permutation_pvalues.csv"

DAYTIME = False
if DAYTIME:
    OUTPUT_FIG = OUTPUT_DIR / "corr_distribution_daytime.png"
else:
    OUTPUT_FIG = OUTPUT_DIR / "corr_distribution_nighttime.png"

N_PERMUTATIONS = 10000
RANDOM_SEED = 42
TARGET_PAIRS = [
    "active_sensing_band_log__offset",
    "active_sensing_band_log__theta",
    "active_sensing_band_log__alpha",
    "active_sensing_band_log__oura_met",
    "active_sensing_band_log__audio_word_duration",
    "active_sensing_band_log__audio_word_count",
]
PAIR_LABELS = {
    "active_sensing_band_log__offset": "active sensing band vs offset",
    "active_sensing_band_log__theta": "active sensing band vs theta",
    "active_sensing_band_log__alpha": "active sensing band vs alpha",
    "active_sensing_band_log__oura_met": "active sensing band vs met",
    "active_sensing_band_log__audio_word_duration": "active sensing band vs audio word duration",
    "active_sensing_band_log__audio_word_count": "active sensing band vs audio word count",
}


def pvalue_to_stars(pvalue: float) -> str:
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "ns"


def permutation_test_mean_diff(a: np.ndarray, b: np.ndarray, n_permutations: int, rng: np.random.Generator) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]

    if len(a) == 0 or len(b) == 0:
        return np.nan

    observed = abs(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    n_a = len(a)

    extreme_count = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(pooled)
        diff = abs(shuffled[:n_a].mean() - shuffled[n_a:].mean())
        if diff >= observed:
            extreme_count += 1

    return (extreme_count + 1) / (n_permutations + 1)


def build_visit_label_map(side_df: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    order_df = (
        side_df[["visit_label", "visit_order", "visit_start_date"]]
        .drop_duplicates()
        .sort_values(["visit_order", "visit_start_date"], kind="mergesort")
    )

    if order_df.empty:
        return [], {}

    first_date = pd.Timestamp(order_df["visit_start_date"].min())
    label_map: dict[str, str] = {}
    visit_order = []
    for _, row in order_df.iterrows():
        visit = row["visit_label"]
        visit_date = pd.Timestamp(row["visit_start_date"])
        delta_days = int((visit_date - first_date).days)
        label_map[visit] = f"{visit}\n+{delta_days}d"
        visit_order.append(visit)

    return visit_order, label_map


def annotate_pairwise_tests(
    ax: plt.Axes,
    side_df: pd.DataFrame,
    visit_order: list[str],
    n_permutations: int,
    rng: np.random.Generator,
    pair_name: str,
) -> list[dict[str, object]]:
    if len(visit_order) < 2:
        return []

    y_values = side_df["correlation"].to_numpy(dtype=float)
    y_values = y_values[np.isfinite(y_values)]
    if len(y_values) == 0:
        return []

    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    y_span = max(y_max - y_min, 1e-6)

    results: list[dict[str, object]] = []
    pairs = list(combinations(visit_order, 2))
    for test_idx, (visit_a, visit_b) in enumerate(pairs):
        vals_a = side_df.loc[side_df["visit_label"] == visit_a, "correlation"].to_numpy(dtype=float)
        vals_b = side_df.loc[side_df["visit_label"] == visit_b, "correlation"].to_numpy(dtype=float)
        vals_a = vals_a[np.isfinite(vals_a)]
        vals_b = vals_b[np.isfinite(vals_b)]

        pvalue = permutation_test_mean_diff(vals_a, vals_b, n_permutations=n_permutations, rng=rng)
        star = pvalue_to_stars(pvalue) if np.isfinite(pvalue) else "na"

        x1 = visit_order.index(visit_a)
        x2 = visit_order.index(visit_b)
        level = test_idx
        y = y_max + y_span * (0.08 + 0.06 * level)
        h = y_span * 0.025

        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1)
        p_text = f"{star} p={pvalue:.3g}" if np.isfinite(pvalue) else "na"
        ax.text((x1 + x2) / 2.0, y + h, p_text, ha="center", va="bottom", fontsize=8)

        results.append(
            {
                "side": side_df["side"].iloc[0],
                "pair": pair_name,
                "visit_a": visit_a,
                "visit_b": visit_b,
                "n_a": int(len(vals_a)),
                "n_b": int(len(vals_b)),
                "p_value": pvalue,
                "stars": star,
                "n_permutations": n_permutations,
            }
        )

    ax.set_ylim(y_min - y_span * 0.05, y_max + y_span * (0.14 + 0.07 * max(len(pairs), 1)))
    return results


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
    if DAYTIME:
        df = df[~df["sleep_present"].astype(bool)].copy()
    else:
        df = df[df["sleep_present"].astype(bool)].copy()
    df = df[df["pair"].isin(TARGET_PAIRS)].copy()
    df = df[np.isfinite(df["correlation"].to_numpy(dtype=float))]

    if df.empty:
        raise ValueError("No outside-sleep correlation rows available for plotting")

    sides = [side for side in ["LEFT", "RIGHT"] if side in set(df["side"].astype(str))]
    if not sides:
        sides = sorted(df["side"].astype(str).unique())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        nrows=len(TARGET_PAIRS),
        ncols=len(sides),
        figsize=(7.5 * max(len(sides), 1), 7.6 * len(TARGET_PAIRS)),
        sharey=False,
        squeeze=False,
    )

    rng = np.random.default_rng(RANDOM_SEED)
    all_pvals: list[dict[str, object]] = []

    for row_idx, pair_name in enumerate(TARGET_PAIRS):
        pair_df = df[df["pair"] == pair_name].copy()
        for col_idx, side in enumerate(sides):
            ax = axes[row_idx, col_idx]
            side_df = pair_df[pair_df["side"].astype(str) == side].copy()
            visit_order, label_map = build_visit_label_map(side_df)
            if not visit_order:
                ax.set_title(f"{side}: {PAIR_LABELS[pair_name]} (no data)")
                ax.set_axis_off()
                continue

            side_df["visit_label"] = pd.Categorical(side_df["visit_label"], categories=visit_order, ordered=True)

            sns.boxplot(
                data=side_df,
                x="visit_label",
                y="correlation",
                order=visit_order,
                color="lightgray",
                showfliers=False,
                showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "crimson", "markeredgecolor": "black", "markersize": 5},
                ax=ax,
            )
            sns.swarmplot(
                data=side_df,
                x="visit_label",
                y="correlation",
                order=visit_order,
                size=3,
                alpha=0.55,
                color="tab:blue",
                ax=ax,
            )

            all_pvals.extend(
                annotate_pairwise_tests(
                    ax=ax,
                    side_df=side_df,
                    visit_order=visit_order,
                    n_permutations=N_PERMUTATIONS,
                    rng=rng,
                    pair_name=pair_name,
                )
            )

            ax.set_title(f"{side}: {PAIR_LABELS[pair_name]}")
            ax.set_xlabel("Visit (days from first visit)")
            ax.set_ylabel("Correlation")
            ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
            ax.set_xticks(range(len(visit_order)))
            ax.set_xticklabels([label_map[v] for v in visit_order], rotation=35, ha="right")

    fig.suptitle("10-minute correlation distributions outside sleep")
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG, dpi=220)
    plt.close(fig)

    pvals_df = pd.DataFrame(all_pvals)
    pvals_df.to_csv(OUTPUT_PVALS, index=False)

    print(f"Saved figure: {OUTPUT_FIG}")
    print(f"Saved permutation p-values: {OUTPUT_PVALS}")


if __name__ == "__main__":
    main()
