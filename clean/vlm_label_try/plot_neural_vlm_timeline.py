from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
NEURAL_PATH = BASE_DIR.parent / "analysis_results" / "daytime_specparam_10s_features.csv"
OUTPUT_PATH = BASE_DIR / "neural_vlm_timeline"
DATE = "2026-02-19"
TIMEZONE = "US/Central"
CHUNK_SECONDS = 15  # same chunk length as map_neural_vlm.py
BIN = "10s"

BAND_NAMES = ["delta", "theta", "alpha", "low_beta", "high_beta", "low_gamma", "high_gamma"]
NEURAL_FEATURES = [f"{band}_band_power_sum" for band in BAND_NAMES] + [
    "aperiodic_exponent",
    "aperiodic_offset",
]
EXCLUDE_LABELS = {"chunk_path", "Fully_Visible", "Partially_Absent", "Completely_Absent"}


def load_vlm_labels(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    chunk_index = frame["chunk_path"].str.split("_").str[-1].astype(int)
    start = pd.to_datetime(
        frame["chunk_path"].str.split(".").str[0].str[len("TRBD001_"):],
        format="%Y%m%d_%H%M%S",
    )
    frame["time"] = start + pd.to_timedelta(chunk_index * CHUNK_SECONDS, unit="s")
    labels = frame.drop(columns=["chunk_path"]).set_index("time").sort_index()
    labels = labels.resample(BIN).mean().tz_localize(TIMEZONE)
    keep = [column for column in labels.columns if column not in EXCLUDE_LABELS]
    return labels[keep].loc[:, labels[keep].fillna(0).sum() > 0]


def load_neural_features(path: Path) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(path).query("date == @DATE")
    frame["time"] = pd.to_datetime(frame["datetime"], utc=True).dt.tz_convert(TIMEZONE)
    result = {}
    for hemisphere in ("left", "right"):
        hemi = frame.loc[frame["hemisphere"] == hemisphere].set_index("time").sort_index()
        hemi = hemi[NEURAL_FEATURES].resample(BIN).mean()
        zscored = (hemi - hemi.mean()) / hemi.std(ddof=0)
        result[hemisphere] = zscored
    return result


def draw_heatmap(axis, frame: pd.DataFrame, index: pd.DatetimeIndex, **imshow_kwargs):
    data = frame.reindex(index).to_numpy(dtype=float).T
    x_start = mdates.date2num(index[0].tz_localize(None))
    x_end = mdates.date2num(index[-1].tz_localize(None)) + 10 / 86400
    image = axis.imshow(
        np.ma.masked_invalid(data),
        aspect="auto",
        interpolation="nearest",
        extent=(x_start, x_end, data.shape[0] - 0.5, -0.5),
        **imshow_kwargs,
    )
    axis.set_yticks(range(data.shape[0]), frame.columns, fontsize=8)
    axis.set_facecolor("white")
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    return image


def main() -> None:
    lounge = load_vlm_labels(BASE_DIR / "lounge.csv")
    sleep = load_vlm_labels(BASE_DIR / "sleep.csv")
    neural = load_neural_features(NEURAL_PATH)

    frames = [lounge, sleep, neural["left"], neural["right"]]
    # neural rows after the last video label (next-day recording) are not shown
    start = min(lounge.index.min(), sleep.index.min())
    end = max(lounge.index.max(), sleep.index.max())
    index = pd.date_range(start, end, freq=BIN)

    heights = [max(len(frame.columns), 4) for frame in frames]
    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": heights},
        layout="constrained",
    )
    titles = ["Lounge camera | VLM labels", "Sleep camera | VLM labels", "Left | Neural features (z-scored)", "Right | Neural features (z-scored)"]
    images = []
    for axis, frame, title in zip(axes, frames, titles):
        if frame is neural["left"] or frame is neural["right"]:
            kwargs = {"cmap": "coolwarm", "vmin": -3, "vmax": 3}
        else:
            kwargs = {"cmap": "Greys", "vmin": 0, "vmax": 1}
        images.append(draw_heatmap(axis, frame, index, **kwargs))
        axis.set_title(title, fontsize=10, loc="left")

    axes[-1].xaxis_date()
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlabel(f"Time on {DATE} (US/Central)")
    axes[-1].set_xlim(mdates.date2num(index[0].tz_localize(None)), mdates.date2num(index[-1].tz_localize(None)))

    for image, axis_pair, label in (
        (images[0], axes[:2], "Label fraction"),
        (images[2], axes[2:], "z-score"),
    ):
        colorbar = fig.colorbar(image, ax=list(axis_pair), fraction=0.02, pad=0.01, shrink=0.8)
        colorbar.set_label(label, fontsize=9)
        colorbar.ax.tick_params(labelsize=8)

    fig.savefig(OUTPUT_PATH.with_suffix(".pdf"))
    fig.savefig(OUTPUT_PATH.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH.with_suffix('.pdf')}")
    print(f"Saved {OUTPUT_PATH.with_suffix('.png')}")


if __name__ == "__main__":
    main()
