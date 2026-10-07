from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats


RESPONDER_PALETTE = {False: "#785286", True: "#DAE319"}
RANDOM_STATE = 42
PEAK_BANDS = ("delta", "theta", "alpha", "low_beta", "high_beta", "low_gamma", "high_gamma")
VISIT_STYLES_BY_DATE = {
    "2025-06-03": ("day 0", "#440154"),
    "2025-06-05": ("day 1", "#414487"),
    "2025-06-10": ("week 1", "#2A788E"),
    "2025-11-04": ("month 6", "#22A884"),
    "2026-02-19": ("month 9", "#7AD151"),
    "2026-05-12": ("month 12", "#FDE725"),
}


def _plot_psd_overlays(
    fits: list[dict],
    output_dir: Path,
    color_by_madrs: bool = False,
) -> list[Path]:
    madrs_values = np.asarray([fit["madrs"] for fit in fits], dtype=float)
    finite_madrs = madrs_values[np.isfinite(madrs_values)]
    if finite_madrs.size:
        low, high = float(finite_madrs.min()), float(finite_madrs.max())
        if low == high:
            high = low + 1.0
    else:
        low, high = 0.0, 1.0
    madrs_norm = plt.Normalize(low, high)
    madrs_cmap = plt.get_cmap("viridis_r")
    date_cmap = plt.get_cmap("viridis")
    outputs = []

    for hemisphere in ("left", "right"):
        hemi_fits = [fit for fit in fits if fit["hemisphere"] == hemisphere]
        dates = sorted({str(fit["date"]) for fit in hemi_fits})
        date_positions = np.linspace(0.05, 0.95, len(dates)) if len(dates) > 1 else [0.5]
        date_colors = {
            day: date_cmap(float(position))
            for day, position in zip(dates, date_positions)
        }
        fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.0))
        panels = [
            (axes[0, 0], "raw_log_power", (0, 100), "Mean 10-second fit | Raw"),
            (axes[0, 1], "aperiodic_removed_log_power", (0, 100), "Mean 10-second fit | Aperiodic removed"),
            (axes[1, 0], "raw_log_power", (0, 40), "Mean 10-second fit | Raw"),
            (axes[1, 1], "aperiodic_removed_log_power", (0, 40), "Mean 10-second fit | Aperiodic removed"),
        ]
        for axis, value_key, x_range, title in panels:
            for fit in hemi_fits:
                score = fit["madrs"]
                if color_by_madrs:
                    color = madrs_cmap(madrs_norm(score)) if np.isfinite(score) else "0.5"
                else:
                    color = date_colors[str(fit["date"])]
                label = (
                    f"{fit['date']} | MADRS={score:g}"
                    if np.isfinite(score)
                    else f"{fit['date']} | MADRS=NA"
                )
                axis.plot(
                    fit["freqs"],
                    fit[value_key],
                    color=color,
                    linewidth=0.9,
                    alpha=0.9,
                    label=label,
                )
            axis.set_xlim(*x_range)
            axis.set_title(title)
            axis.set_xlabel("Frequency (Hz)")
            axis.set_ylabel("Log10 Power")

        axes[0, 0].legend(fontsize=6, frameon=False, loc="best")
        color_label = "MADRS" if color_by_madrs else "date/condition"
        fig.suptitle(
            f"{hemisphere.title()} hemisphere | Mean of individual daytime fits by visit | color: {color_label}"
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        png_path = output_dir / f"daytime_mean_psd_{hemisphere}.png"
        pdf_path = output_dir / f"daytime_mean_psd_{hemisphere}.pdf"
        fig.savefig(png_path, dpi=200)
        fig.savefig(pdf_path)
        plt.close(fig)
        outputs.extend([png_path, pdf_path])

    return outputs


def _permutation_p(group_no: np.ndarray, group_yes: np.ndarray) -> float:
    if len(group_no) < 2 or len(group_yes) < 2:
        return np.nan
    result = stats.permutation_test(
        (group_no, group_yes),
        statistic=lambda x, y: np.mean(x) - np.mean(y),
        vectorized=False,
        n_resamples=1000,
        alternative="two-sided",
        random_state=RANDOM_STATE,
    )
    return float(result.pvalue)


def _visit_date_sort_key(date_label: str) -> tuple[pd.Timestamp, str]:
    label = str(date_label)
    date_prefix = label.split()[0]
    try:
        parsed = pd.Timestamp(date_prefix)
    except (TypeError, ValueError):
        parsed = pd.Timestamp.max
    return parsed, label


def _build_visit_styles(features: pd.DataFrame, fits: list[dict]) -> dict[tuple[int, str], dict[str, object]]:
    categories = set()
    if {"visit", "date"}.issubset(features.columns):
        visit_frame = features.loc[:, ["visit", "date"]].dropna().drop_duplicates()
        for row in visit_frame.itertuples(index=False):
            categories.add((int(row.visit), str(row.date)))
    for fit in fits:
        categories.add((int(fit["visit"]), str(fit["date"])))

    ordered_categories = sorted(categories, key=lambda category: (category[0], *_visit_date_sort_key(category[1])))
    unmapped = [
        category for category in ordered_categories
        if category[1].split()[0] not in VISIT_STYLES_BY_DATE
    ]
    fallback_cmap = plt.get_cmap("viridis", max(len(unmapped), 1))
    visit_styles = {}
    for index, category in enumerate(ordered_categories):
        date_key = category[1].split()[0]
        if date_key in VISIT_STYLES_BY_DATE:
            label, color = VISIT_STYLES_BY_DATE[date_key]
        else:
            label = f"visit {category[0]}"
            color = fallback_cmap(unmapped.index(category))
        visit_styles[category] = {"label": label, "color": color, "order": index}
    return visit_styles


def _mean_std(values: np.ndarray) -> tuple[float, float]:
    if len(values) == 0:
        return np.nan, np.nan
    return float(np.mean(values)), float(np.std(values))


def _feature_value_frame(
    frame: pd.DataFrame,
    feature: str,
    visit_styles: dict[tuple[int, str], dict[str, object]],
) -> pd.DataFrame:
    columns = ["responder", feature]
    if {"visit", "date"}.issubset(frame.columns):
        columns.extend(["visit", "date"])
    values = frame.loc[:, columns].copy()
    values[feature] = pd.to_numeric(values[feature], errors="coerce")
    values = values.dropna(subset=[feature])
    if values.empty:
        return values

    values["responder"] = values["responder"].astype(bool)
    if {"visit", "date"}.issubset(values.columns):
        values["visit"] = pd.to_numeric(values["visit"], errors="coerce")
        values = values.dropna(subset=["visit", "date"])
        values["visit"] = values["visit"].astype(int)
        values["date"] = values["date"].astype(str)
        values["visit_key"] = list(zip(values["visit"], values["date"]))
        values["visit_label"] = values["visit_key"].map(lambda key: visit_styles[key]["label"])
        values["visit_color"] = values["visit_key"].map(lambda key: visit_styles[key]["color"])
        values["visit_order"] = values["visit_key"].map(lambda key: visit_styles[key]["order"])
    return values


def _collect_feature_stats(
    values: pd.DataFrame,
    feature: str,
    feature_label: str,
    figure_name: str,
    hemisphere: str,
) -> tuple[list[dict], list[dict]]:
    if values.empty:
        return [], []

    summary_rows = []
    comparison_rows = []
    group_no = values.loc[~values["responder"], feature].to_numpy(dtype=float)
    group_yes = values.loc[values["responder"], feature].to_numpy(dtype=float)
    for group_name, group in (("non_responder", group_no), ("responder", group_yes)):
        mean_value, std_value = _mean_std(group)
        summary_rows.append(
            {
                "figure": figure_name,
                "hemisphere": hemisphere,
                "feature": feature,
                "feature_label": feature_label,
                "summary_scope": "response_group",
                "response_group": group_name,
                "visit": pd.NA,
                "date": pd.NA,
                "visit_label": pd.NA,
                "visit_order": pd.NA,
                "n": len(group),
                "mean": mean_value,
                "std": std_value,
            }
        )

    comparison_rows.append(
        {
            "figure": figure_name,
            "hemisphere": hemisphere,
            "feature": feature,
            "feature_label": feature_label,
            "comparison": "non_responder_vs_responder",
            "visit": pd.NA,
            "date": pd.NA,
            "visit_label": pd.NA,
            "visit_order": pd.NA,
            "reference_group": "non_responder",
            "comparison_group": "responder",
            "reference_n": len(group_no),
            "comparison_n": len(group_yes),
            "p_value": _permutation_p(group_no, group_yes),
            "note": pd.NA,
        }
    )

    if {"visit", "date", "visit_label", "visit_order"}.issubset(values.columns):
        baseline = group_no
        grouped = values.groupby(["visit", "date", "visit_label", "visit_order", "responder"], sort=False)
        for (visit, date, visit_label, visit_order, responder_value), visit_frame in grouped:
            visit_values = visit_frame[feature].to_numpy(dtype=float)
            mean_value, std_value = _mean_std(visit_values)
            summary_rows.append(
                {
                    "figure": figure_name,
                    "hemisphere": hemisphere,
                    "feature": feature,
                    "feature_label": feature_label,
                    "summary_scope": "visit",
                    "response_group": "responder" if responder_value else "non_responder",
                    "visit": visit,
                    "date": date,
                    "visit_label": visit_label,
                    "visit_order": visit_order,
                    "n": len(visit_values),
                    "mean": mean_value,
                    "std": std_value,
                }
            )
            comparison_rows.append(
                {
                    "figure": figure_name,
                    "hemisphere": hemisphere,
                    "feature": feature,
                    "feature_label": feature_label,
                    "comparison": "visit_vs_non_responder",
                    "visit": visit,
                    "date": date,
                    "visit_label": visit_label,
                    "visit_order": visit_order,
                    "reference_group": "non_responder_visits",
                    "comparison_group": visit_label,
                    "reference_n": len(baseline),
                    "comparison_n": len(visit_values),
                    "p_value": _permutation_p(baseline, visit_values) if responder_value else np.nan,
                    "note": pd.NA if responder_value else "reference non-responder visit",
                }
            )

    return summary_rows, comparison_rows


def _draw_feature_panel(
    axis,
    frame: pd.DataFrame,
    feature: str,
    title: str,
    figure_name: str,
    hemisphere: str,
    visit_styles: dict[tuple[int, str], dict[str, object]],
    boxplot_grouping: str,
) -> tuple[list[dict], list[dict]]:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    feature_label = title
    values = _feature_value_frame(frame, feature, visit_styles)
    if values.empty:
        axis.set_title(title)
        axis.text(0.5, 0.5, "No fitted peaks", transform=axis.transAxes, ha="center", va="center")
        if boxplot_grouping == "response":
            axis.set_xticks([0, 1], ["NR", "R"])
        axis.set_ylabel("")
        return [], []

    if boxplot_grouping == "visits":
        if "visit_key" not in values.columns:
            raise ValueError("Visit-grouped boxplots require visit and date columns in the feature table")
        visit_keys = sorted(
            values["visit_key"].unique(),
            key=lambda key: visit_styles[key]["order"],
        )
        visit_labels = {
            key: f"{visit_styles[key]['label']}\n{key[1]}"
            for key in visit_keys
        }
        values["visit_plot_label"] = values["visit_key"].map(visit_labels)
        label_order = [visit_labels[key] for key in visit_keys]
        palette = {
            visit_labels[key]: visit_styles[key]["color"]
            for key in visit_keys
        }
        sns.boxplot(
            data=values,
            x="visit_plot_label",
            y=feature,
            order=label_order,
            hue="visit_plot_label",
            hue_order=label_order,
            palette=palette,
            dodge=False,
            boxprops={"alpha": 0.4},
            showfliers=False,
            legend=False,
            ax=axis,
        )
        axis.tick_params(axis="x", labelrotation=45)
        for tick in axis.get_xticklabels():
            tick.set_ha("right")
    elif boxplot_grouping == "response":
        sns.boxplot(
            data=values,
            x="responder",
            y=feature,
            order=[False, True],
            hue="responder",
            hue_order=[False, True],
            palette=RESPONDER_PALETTE,
            dodge=False,
            boxprops={"alpha": 0.4},
            showfliers=False,
            legend=False,
            ax=axis,
        )
    else:
        raise ValueError(f"Unsupported boxplot grouping: {boxplot_grouping}")

    group_no = values.loc[~values["responder"], feature].to_numpy(dtype=float)
    group_yes = values.loc[values["responder"], feature].to_numpy(dtype=float)
    no_mean, no_std = _mean_std(group_no)
    yes_mean, yes_std = _mean_std(group_yes)
    if boxplot_grouping == "response":
        axis.scatter(
            [0, 1],
            [no_mean, yes_mean],
            marker="^",
            color="green",
            edgecolors="black",
            s=20,
            zorder=5,
        )
        p_value = _permutation_p(group_no, group_yes)
    else:
        p_value = np.nan
    if boxplot_grouping == "response" and np.isfinite(p_value):
        whisker_ranges = []
        for group in (group_no, group_yes):
            lower_quartile, upper_quartile = np.quantile(group, [0.25, 0.75])
            interquartile_range = upper_quartile - lower_quartile
            inliers = group[
                (group >= lower_quartile - 1.5 * interquartile_range)
                & (group <= upper_quartile + 1.5 * interquartile_range)
            ]
            whisker_ranges.append((float(inliers.min()), float(inliers.max())))

        visible_min = min(bounds[0] for bounds in whisker_ranges)
        visible_max = max(bounds[1] for bounds in whisker_ranges)
        visible_span = max(visible_max - visible_min, np.finfo(float).eps)
        gap = max(0.035 * visible_span, 0.015)
        bracket_y = visible_max + 0.15 * gap
        bracket_top = bracket_y + 0.8 * gap
        axis.plot(
            [0, 0, 1, 1],
            [bracket_y, bracket_top, bracket_top, bracket_y],
            linewidth=0.8,
            color="black",
        )
        axis.annotate(
            "*" if p_value < 0.05 else "n.s.",
            xy=(0.5, bracket_top),
            xytext=(0, 1),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6,
        )
        axis.set_ylim(visible_min - 0.04 * visible_span, bracket_top + 1.5 * gap)
        title = (
            f"{title}\np={p_value:.3g}"
            f"\nNR {no_mean:.2f} ± {no_std:.2f}"
            f"\nR {yes_mean:.2f} ± {yes_std:.2f}"
        )
    elif boxplot_grouping == "response":
        title = f"{title}\np=NA (n<2 in a group)"

    axis.set_title(title, fontsize=5.5, pad=3)
    axis.set_xlabel("")
    axis.set_ylabel("")
    if boxplot_grouping == "response":
        axis.set_xticks([0, 1], ["NR", "R"])
    axis.tick_params(axis="both", labelsize=6)
    axis.grid(False)
    return _collect_feature_stats(values, feature, feature_label, figure_name, hemisphere)


def _plot_feature_boxplots(
    features: pd.DataFrame,
    output_dir: Path,
    visit_styles: dict[tuple[int, str], dict[str, object]],
    boxplot_grouping: str,
) -> tuple[list[Path], list[dict], list[dict]]:
    metric_suffix = next(
        (
            suffix
            for suffix in ("peak_height", "band_power_sum")
            if all(f"{band}_{suffix}" in features.columns for band in PEAK_BANDS)
        ),
        None,
    )
    if metric_suffix is None:
        raise ValueError("Feature table must contain peak-height or band-power-sum columns for every band")

    metric_label = "Peak height" if metric_suffix == "peak_height" else "Band power sum"
    features_to_plot = [f"{band}_{metric_suffix}" for band in PEAK_BANDS]
    feature_labels = {
        f"{band}_{metric_suffix}": f"{band.replace('_', ' ').title()} {metric_label.lower()}"
        for band in PEAK_BANDS
    }
    features_to_plot.extend(["aperiodic_offset", "aperiodic_exponent"])
    feature_labels.update(
        {
            "aperiodic_offset": "Aperiodic offset",
            "aperiodic_exponent": "Aperiodic exponent",
        }
    )

    fig, axes = plt.subplots(2, len(features_to_plot), figsize=(20.5, 7), squeeze=False)
    summary_rows = []
    comparison_rows = []
    for row, hemisphere in enumerate(("left", "right")):
        hemi_features = features.loc[features["hemisphere"] == hemisphere]
        for column, feature in enumerate(features_to_plot):
            panel_summary_rows, panel_comparison_rows = _draw_feature_panel(
                axes[row, column],
                hemi_features,
                feature,
                f"{hemisphere.title()} | {feature_labels[feature]}",
                f"daytime_specparam_features_by_{boxplot_grouping}",
                hemisphere,
                visit_styles,
                boxplot_grouping=boxplot_grouping,
            )
            summary_rows.extend(panel_summary_rows)
            comparison_rows.extend(panel_comparison_rows)
        axes[row, 0].set_ylabel(f"{metric_label} / parameter", fontsize=7)

    if boxplot_grouping == "response":
        responder_handles = [
            Patch(facecolor=RESPONDER_PALETTE[False], alpha=0.4, label="Non-responder"),
            Patch(facecolor=RESPONDER_PALETTE[True], alpha=0.4, label="Responder"),
        ]
        fig.legend(handles=responder_handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2, frameon=False)
        title = "MADRS response"
        output_stem = "daytime_specparam_features_by_response"
    else:
        visit_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                markersize=5,
                label=f"{style['label']} | {category[1]}",
            )
            for category, style in sorted(visit_styles.items(), key=lambda item: item[1]["order"])
        ]
        fig.legend(
            handles=visit_handles,
            loc="upper center",
            bbox_to_anchor=(0.73, 1.005),
            ncol=min(len(visit_handles), 4),
            frameon=False,
            fontsize=6.5,
        )
        title = "NBU visit"
        output_stem = "daytime_specparam_features_by_visit"
    fig.suptitle(
        f"Daytime Specparam boxplots by {title} | {metric_label.lower()} | all successful 10-second fits",
        y=1.08,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path], summary_rows, comparison_rows


def _plot_paper_figure_4(
    features: pd.DataFrame,
    fits: list[dict],
    output_dir: Path,
    visit_styles: dict[tuple[int, str], dict[str, object]],
    boxplot_grouping: str,
) -> tuple[list[Path], list[dict], list[dict]]:
    metric_suffix = next(
        (
            suffix
            for suffix in ("peak_height", "band_power_sum")
            if f"low_beta_{suffix}" in features.columns
        ),
        None,
    )
    if metric_suffix is None:
        raise ValueError("Feature table must contain low_beta_peak_height or low_beta_band_power_sum")

    fit_categories = sorted(
        {(int(fit["visit"]), str(fit["date"])) for fit in fits},
        key=lambda category: (category[0], *_visit_date_sort_key(category[1])),
    )
    category_colors = {category: visit_styles[category]["color"] for category in fit_categories}
    fit_curves = {}
    for hemisphere in ("left", "right"):
        hemi_fits = [fit for fit in fits if fit["hemisphere"] == hemisphere]
        if not hemi_fits:
            raise ValueError(f"No fitted spectra found for the {hemisphere} hemisphere")
        freqs = np.asarray(hemi_fits[0]["freqs"], dtype=float)
        for fit in hemi_fits:
            fit_freqs = np.asarray(fit["freqs"], dtype=float)
            raw_curve = np.asarray(fit["raw_log_power"], dtype=float)
            removed_curve = np.asarray(fit["aperiodic_removed_log_power"], dtype=float)
            if (
                not np.array_equal(freqs, fit_freqs)
                or raw_curve.shape != freqs.shape
                or removed_curve.shape != freqs.shape
            ):
                raise ValueError(f"Inconsistent fitted frequency grid for the {hemisphere} hemisphere")
        fit_curves[hemisphere] = (hemi_fits, freqs)

    fig, axes = plt.subplots(
        1,
        6,
        figsize=(17.5, 6.2),
        gridspec_kw={"width_ratios": [1.2, 1.2, 1.2, 1.2, 1.15, 1.15]},
    )
    spectra_panels = (
        (axes[0], "left", "raw_log_power", "Left | Raw"),
        (axes[1], "right", "raw_log_power", "Right | Raw"),
        (axes[2], "left", "aperiodic_removed_log_power", "Left | Aperiodic removed"),
        (axes[3], "right", "aperiodic_removed_log_power", "Right | Aperiodic removed"),
    )
    for axis, hemisphere, curve_key, title in spectra_panels:
        hemi_fits, freqs = fit_curves[hemisphere]
        for fit in hemi_fits:
            category = (int(fit["visit"]), str(fit["date"]))
            axis.plot(
                freqs,
                fit[curve_key],
                color=category_colors[category],
                linewidth=1.0,
                alpha=0.9,
            )
        axis.set_xlim(1, 40)
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Log10 Power" if curve_key == "raw_log_power" else "Aperiodic-removed log power")
        axis.set_title(title)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(False)

    low_beta = f"low_beta_{metric_suffix}"
    metric_label = "Peak height" if metric_suffix == "peak_height" else "Band-power sum"
    left_summary_rows, left_comparison_rows = _draw_feature_panel(
        axes[4],
        features.loc[features["hemisphere"] == "left"],
        low_beta,
        "Left | Low beta",
        "paper_figure_4_ephys",
        "left",
        visit_styles,
        boxplot_grouping=boxplot_grouping,
    )
    axes[4].set_ylabel(metric_label, fontsize=6)
    right_summary_rows, right_comparison_rows = _draw_feature_panel(
        axes[5],
        features.loc[features["hemisphere"] == "right"],
        "aperiodic_offset",
        "Right | Aperiodic offset",
        "paper_figure_4_ephys",
        "right",
        visit_styles,
        boxplot_grouping=boxplot_grouping,
    )
    axes[5].set_ylabel("Aperiodic offset", fontsize=6)

    category_labels = {}
    for fit in fits:
        category = (int(fit["visit"]), str(fit["date"]))
        madrs = fit["madrs"]
        madrs_label = f"MADRS={madrs:g}" if np.isfinite(madrs) else "MADRS=NA"
        category_labels[category] = f"{visit_styles[category]['label']}: {category[1]} | {madrs_label}"
    visit_handles = [
        Line2D(
            [0],
            [0],
            color=category_colors[category],
            linewidth=1.2,
            label=category_labels[category],
        )
        for category in fit_categories
    ]
    fig.legend(
        handles=visit_handles,
        title="Visit / period",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=min(len(visit_handles), 3),
        frameon=False,
        fontsize=7,
        title_fontsize=7,
    )
    boxplot_title = "visit" if boxplot_grouping == "visits" else "response"
    fig.suptitle(f"NBU visit-mean spectra and {boxplot_title} features", y=1.17)
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    png_path = output_dir / "paper_figure_4_ephys.png"
    pdf_path = output_dir / "paper_figure_4_ephys.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path], left_summary_rows + right_summary_rows, left_comparison_rows + right_comparison_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot individual daytime Specparam fits and feature summaries.")
    parser.add_argument("--input-dir", type=Path, default=Path("clean/analysis_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("clean/analysis_results/figures"))
    parser.add_argument(
        "--color-by-madrs",
        action="store_true",
        help="Color PSD overlays by MADRS; default assigns unique Viridis colors by date/condition (requires --run-psd).",
    )
    parser.add_argument(
        "--run-psd",
        action="store_true",
        help="Generate PSD plots, including overlays and the spectra panels in paper_figure_4_ephys.",
    )
    parser.add_argument(
        "--boxplot-grouping",
        choices=("visits", "response"),
        default="visits",
        help="Group boxplots by NBU visit (default) or by responder status.",
    )
    args = parser.parse_args()
    if args.color_by_madrs and not args.run_psd:
        parser.error("--color-by-madrs requires --run-psd")

    feature_path = args.input_dir / "daytime_specparam_10s_features.csv"
    curve_path = args.input_dir / "daytime_specparam_fits.pkl"
    features = pd.read_csv(feature_path)
    with curve_path.open("rb") as file:
        fits = pickle.load(file)
    if features.empty or not fits:
        raise ValueError(f"No fit results found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    visit_styles = _build_visit_styles(features, fits)
    outputs = []
    if args.run_psd:
        outputs.extend(_plot_psd_overlays(fits, args.output_dir, color_by_madrs=args.color_by_madrs))
    feature_outputs, feature_summary_rows, feature_comparison_rows = _plot_feature_boxplots(
        features,
        args.output_dir,
        visit_styles,
        boxplot_grouping=args.boxplot_grouping,
    )
    outputs.extend(feature_outputs)
    paper_summary_rows = []
    paper_comparison_rows = []
    if args.run_psd:
        paper_outputs, paper_summary_rows, paper_comparison_rows = _plot_paper_figure_4(
            features,
            fits,
            args.output_dir,
            visit_styles,
            boxplot_grouping=args.boxplot_grouping,
        )
        outputs.extend(paper_outputs)

    summary_path = args.output_dir / "boxplot_summary_stats.csv"
    comparison_path = args.output_dir / "boxplot_permutation_tests.csv"
    summary_frame = pd.DataFrame(feature_summary_rows + paper_summary_rows).sort_values(
        ["figure", "hemisphere", "feature", "summary_scope", "visit_order", "response_group"],
        na_position="last",
    )
    comparison_frame = pd.DataFrame(feature_comparison_rows + paper_comparison_rows).sort_values(
        ["figure", "hemisphere", "feature", "comparison", "visit_order"],
        na_position="last",
    )
    summary_frame.to_csv(summary_path, index=False)
    comparison_frame.to_csv(comparison_path, index=False)
    outputs.extend([summary_path, comparison_path])

    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
