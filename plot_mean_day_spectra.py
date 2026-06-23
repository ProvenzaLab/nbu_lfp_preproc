from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path("nbu_concat_single_channel")
FILE_GLOB = "spectra_fooof_bandpower_*.pkl"
OUTPUT_DIR = Path("out_figures_concat_single_channel")
OUTPUT_FIG = OUTPUT_DIR / "mean_day_spectra_by_visit.png"

LOCAL_TZ = "America/Chicago"
DAY_START_HOUR = 8
DAY_END_HOUR = 20
LOW_FREQ_MAX_HZ = 30.0

VISIT_FILE_RE = re.compile(
    r"^spectra_fooof_bandpower_(?P<channel>.+?)_visit(?P<visit>\d+)_(?P<start>\d{4}-\d{2}-\d{2})_to_(?P<end>\d{4}-\d{2}-\d{2})$"
)


def parse_visit_metadata(file_stem: str) -> dict[str, object]:
    match = VISIT_FILE_RE.match(file_stem)
    if match is None:
        return {
            "visit_label": "visit_unknown",
            "visit_order": -1,
            "visit_start_date": pd.NaT,
            "side": "UNKNOWN",
        }

    channel = match.group("channel")
    side = channel.split("_")[-1] if "_" in channel else "UNKNOWN"
    return {
        "visit_label": f"visit{int(match.group('visit')):02d}",
        "visit_order": int(match.group("visit")),
        "visit_start_date": pd.Timestamp(match.group("start")),
        "side": side,
    }


def ensure_datetime_index(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if idx.tz is None:
        idx = idx.tz_localize(tz)
    else:
        idx = idx.tz_convert(tz)
    out.index = idx
    return out.sort_index()


def select_daytime(df: pd.DataFrame) -> pd.DataFrame:
    hours = df.index.hour
    mask = (hours >= DAY_START_HOUR) & (hours < DAY_END_HOUR)
    return df.loc[mask]


def extract_numeric_spectrum(df_spec: pd.DataFrame) -> pd.DataFrame:
    spec = df_spec.copy()
    numeric_cols = pd.to_numeric(spec.columns, errors="coerce")
    keep = ~numeric_cols.isna()
    spec = spec.loc[:, keep].copy()
    spec.columns = numeric_cols[keep].astype(float)
    spec = spec.reindex(sorted(spec.columns), axis=1)
    spec = spec.apply(pd.to_numeric, errors="coerce")
    # Log-transform while ignoring non-positive powers.
    return np.log(spec.where(spec > 0))


def visit_color_labels(records: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, str]]:
    if not records:
        return records, {}

    df = pd.DataFrame(
        {
            "visit_label": [r["visit_label"] for r in records],
            "visit_order": [r["visit_order"] for r in records],
            "visit_start_date": [r["visit_start_date"] for r in records],
        }
    ).drop_duplicates()

    df = df.sort_values(["visit_order", "visit_start_date"], kind="mergesort")
    first_date = pd.Timestamp(df["visit_start_date"].min())

    label_map: dict[str, str] = {}
    for _, row in df.iterrows():
        visit = str(row["visit_label"])
        visit_dt = pd.Timestamp(row["visit_start_date"])
        delta_days = int((visit_dt - first_date).days)
        label_map[visit] = f"{visit} (+{delta_days}d)"

    ordered_labels = df["visit_label"].astype(str).tolist()
    rank = {visit: i for i, visit in enumerate(ordered_labels)}
    records_sorted = sorted(
        records,
        key=lambda r: (rank.get(str(r["visit_label"]), 10**9), pd.Timestamp(r["visit_start_date"])),
    )

    return records_sorted, label_map


def main() -> None:
    pkl_files = sorted(BASE_DIR.glob(FILE_GLOB))
    if not pkl_files:
        raise FileNotFoundError(f"No files found at {BASE_DIR}/{FILE_GLOB}")

    per_side: dict[str, list[dict[str, object]]] = {"LEFT": [], "RIGHT": []}

    for pkl_file in pkl_files:
        data = pd.read_pickle(pkl_file)
        if not isinstance(data, dict) or "df_spec" not in data:
            continue

        meta = parse_visit_metadata(pkl_file.stem)
        side = str(meta["side"])
        if side not in per_side:
            continue

        df_spec = data["df_spec"]
        if not isinstance(df_spec, pd.DataFrame) or df_spec.empty:
            continue

        spec = extract_numeric_spectrum(df_spec)
        if spec.empty:
            continue

        spec = ensure_datetime_index(spec, LOCAL_TZ)
        day_spec = select_daytime(spec)
        if day_spec.empty:
            continue

        mean_spectrum = day_spec.mean(axis=0, skipna=True)
        mean_spectrum = mean_spectrum[mean_spectrum.notna()]
        if mean_spectrum.empty:
            continue

        per_side[side].append(
            {
                "visit_label": meta["visit_label"],
                "visit_order": meta["visit_order"],
                "visit_start_date": meta["visit_start_date"],
                "freqs": mean_spectrum.index.to_numpy(dtype=float),
                "power": mean_spectrum.to_numpy(dtype=float),
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey="row")
    side_order = ["LEFT", "RIGHT"]

    for col_idx, side in enumerate(side_order):
        ax_full = axes[0, col_idx]
        ax_low = axes[1, col_idx]
        records = per_side.get(side, [])
        if not records:
            ax_full.set_title(f"{side}: no daytime spectra")
            ax_full.set_xlabel("Frequency (Hz)")
            ax_full.set_ylabel("Mean log spectral power")
            ax_low.set_title(f"{side}: no daytime spectra <= {int(LOW_FREQ_MAX_HZ)} Hz")
            ax_low.set_xlabel("Frequency (Hz)")
            ax_low.set_ylabel("Mean log spectral power")
            continue

        records, label_map = visit_color_labels(records)
        cmap = plt.get_cmap("viridis")

        n = len(records)
        for idx, rec in enumerate(records):
            color = cmap(0.1 + 0.8 * (idx / max(n - 1, 1)))
            visit_label = str(rec["visit_label"])
            plot_label = label_map.get(visit_label, visit_label)
            freqs = rec["freqs"]
            power = rec["power"]

            ax_full.plot(freqs, power, color=color, linewidth=2, alpha=0.95, label=plot_label)

            low_mask = freqs <= LOW_FREQ_MAX_HZ
            if np.any(low_mask):
                ax_low.plot(freqs[low_mask], power[low_mask], color=color, linewidth=2, alpha=0.95, label=plot_label)

        ax_full.set_title(f"{side} hemisphere (full range)")
        ax_full.set_xlabel("Frequency (Hz)")
        ax_full.set_ylabel("Mean log spectral power")
        ax_full.grid(alpha=0.3)
        ax_full.legend(title="Visit", fontsize=8)

        ax_low.set_title(f"{side} hemisphere (<= {int(LOW_FREQ_MAX_HZ)} Hz)")
        ax_low.set_xlabel("Frequency (Hz)")
        ax_low.set_ylabel("Mean log spectral power")
        ax_low.set_xlim(0, LOW_FREQ_MAX_HZ)
        ax_low.grid(alpha=0.3)
        ax_low.legend(title="Visit", fontsize=8)

    fig.suptitle(
        f"Mean daytime log spectra by visit ({DAY_START_HOUR}:00-{DAY_END_HOUR}:00, night excluded)"
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_FIG, dpi=220)
    plt.close(fig)

    print(f"Saved figure: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
