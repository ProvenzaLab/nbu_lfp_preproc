from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_DIR = Path("out_figures_concat_single_channel_tight")
INPUT_PKL = INPUT_DIR / "non_sleep_10s_df_features.pkl"
OUTPUT_DIR = INPUT_DIR / "df_features_boxplots"
MAX_VISIT_LABEL_LEN = 24


def _short_label(label: str, max_len: int = MAX_VISIT_LABEL_LEN) -> str:
    if len(label) <= max_len:
        return label
    return f"{label[: max_len - 1]}~"


def _numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude_cols = {
        "file_name",
        "side",
        "visit_label",
        "nbu_visit_date",
        "exact_datetime",
        "source_channel",
    }
    numeric_cols: list[str] = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
    return numeric_cols


def _visit_order(df: pd.DataFrame) -> list[str]:
    if "visit_label" not in df.columns:
        return ["visit_unknown"]

    visits = df["visit_label"].astype(str).dropna().unique().tolist()
    if not visits:
        return ["visit_unknown"]

    if "exact_datetime" in df.columns:
        dt = pd.to_datetime(df["exact_datetime"], errors="coerce")
        if dt.notna().any():
            ordered = (
                df.assign(_dt=dt)
                .dropna(subset=["_dt"])
                .groupby("visit_label")["_dt"]
                .min()
                .sort_values()
                .index.astype(str)
                .tolist()
            )
            if ordered:
                return ordered

    return sorted(visits)


def _plot_all_features_boxplot(
    df: pd.DataFrame,
    feature_cols: list[str],
    visits: list[str],
    output_path: Path,
) -> None:
    valid_features = []
    for feature in feature_cols:
        vals = pd.to_numeric(df[feature], errors="coerce").dropna()
        if not vals.empty:
            valid_features.append(feature)

    if not valid_features:
        raise ValueError("No non-NaN numeric feature values found for plotting.")

    fig_width = max(11, 0.7 * len(valid_features))
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))

    n_features = len(valid_features)
    n_visits = max(len(visits), 1)
    base_positions = np.arange(1, n_features + 1, dtype=float)
    group_span = 0.82
    box_width = min(0.75 / n_visits, 0.25)
    if n_visits == 1:
        offsets = np.array([0.0])
    else:
        offsets = np.linspace(-group_span / 2, group_span / 2, n_visits)

    cmap = plt.get_cmap("turbo")
    visit_colors = [cmap(i / max(n_visits - 1, 1)) for i in range(n_visits)]

    for visit_idx, visit in enumerate(visits):
        data = []
        positions = []
        for feature_idx, feature in enumerate(valid_features):
            mask = df["visit_label"].astype(str) == visit
            vals = pd.to_numeric(df.loc[mask, feature], errors="coerce").dropna().values
            if len(vals) == 0:
                continue
            data.append(vals)
            positions.append(base_positions[feature_idx] + offsets[visit_idx])

        if not data:
            continue

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(visit_colors[visit_idx])
            patch.set_alpha(0.45)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.1)

    ax.set_xticks(base_positions)
    ax.set_xticklabels([_short_label(v) for v in valid_features], rotation=60, ha="right")

    legend_handles = [
        plt.Line2D([0], [0], color=visit_colors[i], lw=6, alpha=0.8)
        for i in range(n_visits)
    ]
    legend_labels = [_short_label(v) for v in visits]
    ax.legend(
        legend_handles,
        legend_labels,
        title="Visit",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
    )

    ax.set_title("Non-sleep df_features: boxplots by feature (hue=visit)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    if not INPUT_PKL.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_PKL}")

    df = pd.read_pickle(INPUT_PKL)
    if df.empty:
        raise ValueError(f"Input dataframe is empty: {INPUT_PKL}")

    if "visit_label" not in df.columns:
        df = df.copy()
        df["visit_label"] = "visit_unknown"

    if "side" not in df.columns:
        df = df.copy()
        df["side"] = "ALL"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = _numeric_feature_columns(df)
    if not feature_cols:
        raise ValueError("No numeric feature columns found for boxplots.")

    written = 0
    for side in sorted(df["side"].dropna().astype(str).unique().tolist()):
        side_df = df[df["side"].astype(str) == side].copy()
        if side_df.empty:
            continue

        visits = _visit_order(side_df)
        side_slug = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in side.lower())
        out_file = OUTPUT_DIR / f"boxplot_all_features_{side_slug}.pdf"
        _plot_all_features_boxplot(side_df, feature_cols, visits, out_file)
        print(f"Saved all-feature boxplot for {side} to: {out_file}")
        written += 1

    if written == 0:
        raise ValueError("No side-specific boxplots were written.")


if __name__ == "__main__":
    main()
