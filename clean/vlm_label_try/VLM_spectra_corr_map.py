"""Frequency x VLM-behavior association maps (per environment and hemisphere).

Run from anywhere:
    python VLM_spectra_corr_map.py --metric roc_auc            # 0-40 Hz
    python VLM_spectra_corr_map.py --metric cohens_d --wide    # 0-90 Hz
    python VLM_spectra_corr_map.py --metric perm_p --n-perm 1000
"""
import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import rankdata

from VLM_spectra import (
    BASE_DIR,
    ENVIRONMENTS,
    FREQ_RANGE_DEFAULT,
    FREQ_RANGE_WIDE,
    label_columns,
    load_all,
)

BIN_HZ = 0.1  # native frequency resolution (row height in the maps)
P_CLIP = 0.5  # permutation p-values above this are clipped for display
SIDES = ("left", "right")


def frequency_bins(spectrum: pd.DataFrame, freq_range: tuple[float, float]) -> pd.DataFrame:
    """log10 power at the native 0.1 Hz resolution (no band averaging)."""
    freqs = spectrum.columns.to_numpy(dtype=float)
    keep = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    power = spectrum.loc[:, keep]
    return np.log10(power.where(power > 0))


def auc_from_rank_sum(rank_sum, n_pos, n_neg):
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def cohens_d(values: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = values[labels == 1], values[labels == 0]
    pooled = np.sqrt(
        ((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1)) / (len(pos) + len(neg) - 2)
    )
    return float((pos.mean() - neg.mean()) / pooled) if pooled > 0 else np.nan


def auc_and_perm_p(values, labels, shifts):
    """AUC and a circular time-shift permutation p-value for |AUC - 0.5|.

    Shifting keeps the autocorrelation of both the labels and the spectrum.
    """
    n = len(labels)
    ranks = rankdata(values)
    positives = np.flatnonzero(labels)
    n_pos, n_neg = len(positives), n - len(positives)
    observed = auc_from_rank_sum(ranks[positives].sum(), n_pos, n_neg)
    shifted_sums = ranks[(positives[None, :] + shifts[:, None]) % n].sum(axis=1)
    null = np.abs(auc_from_rank_sum(shifted_sums, n_pos, n_neg) - 0.5)
    return float(observed), float((1 + np.sum(null >= abs(observed - 0.5))) / (1 + len(shifts)))


def compute_associations(labels, bins, args) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    rows = []
    for env in ENVIRONMENTS:
        env_labels = labels[env]
        behaviors = label_columns(env_labels)
        visible = env_labels.index[env_labels["Fully_Visible"] == 1]
        for side in SIDES:
            joined = bins[side].reindex(visible).join(env_labels.loc[visible, behaviors], rsuffix="_label")
            joined = joined.sort_index()
            for behavior in behaviors:
                for freq in bins[side].columns:
                    pair = joined[[freq, behavior]].dropna()
                    y = (pair[behavior].to_numpy() > 0).astype(int)
                    x = pair[freq].to_numpy(dtype=float)
                    n_pos = int(y.sum())
                    if min(n_pos, len(y) - n_pos) < args.min_positive:
                        continue
                    row = {"environment": env, "side": side, "behavior": behavior, "frequency": freq, "n_positive": n_pos}
                    if args.metric == "cohens_d":
                        row["value"] = cohens_d(x, y)
                    else:
                        shifts = rng.integers(1, len(y), size=args.n_perm if args.metric == "perm_p" else 1)
                        auc, p_value = auc_and_perm_p(x, y, shifts) if args.metric == "perm_p" else (
                            auc_and_perm_p(x, y, shifts)[0],
                            np.nan,
                        )
                        row["auc"] = auc
                        row["perm_p"] = p_value
                        row["value"] = (
                            min(p_value, P_CLIP) if args.metric == "perm_p" else max(auc, 1 - auc)
                        )
                    rows.append(row)
    return pd.DataFrame(rows)


def plot_maps(results: pd.DataFrame, args, freq_range) -> None:
    if args.metric == "roc_auc":
        style = {"cmap": "coolwarm", "vmin": 0.5, "vmax": results["value"].max(), "label": "AUC strength"}
        title = "ROC AUC strength: spectrum vs VLM labels"
    elif args.metric == "cohens_d":
        limit = 0.8
        style = {"cmap": "coolwarm", "vmin": -limit, "vmax": limit, "label": "Cohen's d (behavior present - absent)"}
        title = "Cohen's d: spectrum vs VLM labels"
    else:
        style = {"cmap": "viridis_r", "vmin": 0, "vmax": P_CLIP, "label": f"Permutation p (clipped at {P_CLIP})"}
        title = f"Time-shift permutation p of AUC ({args.n_perm} permutations)"

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True, layout="constrained", squeeze=False)
    image = None
    for row, env in enumerate(ENVIRONMENTS):
        for column, side in enumerate(SIDES):
            axis = axes[row, column]
            subset = results.query("environment == @env and side == @side")
            matrix = subset.pivot_table(index="frequency", columns="behavior", values="value", dropna=False)
            axis.set_title(f"{env.title()} | {side.title()}", fontsize=12, loc="left")
            if matrix.empty:
                axis.text(0.5, 0.5, "No behavior with enough windows", ha="center", va="center", transform=axis.transAxes)
                continue
            image = axis.imshow(
                np.ma.masked_invalid(matrix.to_numpy(dtype=float)),
                aspect="auto",
                interpolation="nearest",
                origin="lower",
                cmap=style["cmap"],
                vmin=style["vmin"],
                vmax=style["vmax"],
                extent=(-0.5, matrix.shape[1] - 0.5, matrix.index.min(), matrix.index.max() + BIN_HZ),
            )
            axis.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=90, fontsize=11)
            if column == 0:
                axis.set_ylabel("Frequency (Hz)", fontsize=12)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.01)
        colorbar.set_label(style["label"], fontsize=12)
        colorbar.ax.tick_params(labelsize=10)
    fig.suptitle(f"{title} | {freq_range[0]:g}-{freq_range[1]:g} Hz", fontsize=15)

    suffix = "_0-90Hz" if args.wide else ""
    output = BASE_DIR / f"VLM_spectra_corr_map_{args.metric}{suffix}.pdf"
    fig.savefig(output)
    plt.close(fig)
    print(f"Saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["roc_auc", "cohens_d", "perm_p"], default="roc_auc")
    parser.add_argument("--wide", action="store_true", help="Use 0-90 Hz instead of 0-40 Hz")
    parser.add_argument("--n-perm", type=int, default=1000, help="Permutations for --metric perm_p")
    parser.add_argument("--min-positive", type=int, default=5, help="Minimum windows per class to score a pair")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    freq_range = FREQ_RANGE_WIDE if args.wide else FREQ_RANGE_DEFAULT

    labels, spectra, _ = load_all(freq_range)
    bins = {side: frequency_bins(spectra[side], freq_range) for side in SIDES}
    results = compute_associations(labels, bins, args)
    if results.empty:
        raise ValueError("No feature/behavior pair had enough windows")

    csv_path = BASE_DIR / f"VLM_spectra_corr_map_{args.metric}{'_0-90Hz' if args.wide else ''}.csv"
    results.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    plot_maps(results, args, freq_range)


if __name__ == "__main__":
    main()
