"""Time-aligned VLM labels (lounge, sleep) and left/right power spectrograms.

Run from anywhere:
    python VLM_spectra.py            # 0-40 Hz
    python VLM_spectra.py --wide     # 0-90 Hz
"""
import argparse
import pickle
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
SPECTRA_DIR = BASE_DIR.parent / "nbu_concat_single_channel_TRBD001"
DATE = "2026-02-19"
TIMEZONE = "US/Central"
CHUNK_SECONDS = 15  # same chunk length as map_neural_vlm.py
BIN = pd.Timedelta("10s")
# Same lower bound / log scale / color limits as plot_daily_time_frequency.py
FREQ_RANGE_DEFAULT = (0.1, 40.0)
FREQ_RANGE_WIDE = (0.1, 90.0)
LOG_VMIN, LOG_VMAX = -5.5, 1.0
EXCLUDE_LABELS = {"chunk_path", "Fully_Visible", "Partially_Absent", "Completely_Absent"}
ENVIRONMENTS = ("lounge", "sleep")


def load_vlm_labels(path: Path) -> pd.DataFrame:
    """Return labels on a continuous 10 s grid (US/Central); each chunk covers CHUNK_SECONDS."""
    frame = pd.read_csv(path)
    chunk_index = frame["chunk_path"].str.split("_").str[-1].astype(int)
    start = pd.to_datetime(
        frame["chunk_path"].str.split(".").str[0].str[len("TRBD001_"):],
        format="%Y%m%d_%H%M%S",
    )
    frame["time"] = (start + pd.to_timedelta(chunk_index * CHUNK_SECONDS, unit="s")).dt.tz_localize(TIMEZONE)
    frame = frame.drop(columns=["chunk_path"]).sort_values("time").drop_duplicates("time")

    grid = pd.DataFrame(
        {"time": pd.date_range(frame["time"].min().ceil("10s"), frame["time"].max(), freq=BIN)}
    )
    labels = pd.merge_asof(
        grid,
        frame,
        on="time",
        direction="backward",
        tolerance=pd.Timedelta(seconds=CHUNK_SECONDS),
    ).set_index("time")
    return labels.dropna(how="all")


def label_columns(labels: pd.DataFrame) -> list[str]:
    """Behavior columns that occur at least once (visibility labels excluded)."""
    columns = [column for column in labels.columns if column not in EXCLUDE_LABELS]
    return [column for column in columns if labels[column].fillna(0).sum() > 0]


def load_spectrum(side: str, freq_range: tuple[float, float]) -> pd.DataFrame:
    """Return the precomputed spectrum (time x frequency) on a 10 s US/Central grid."""
    matches = sorted(SPECTRA_DIR.glob(f"spectra_highres_*_{side.upper()}_visit*_{DATE}_to_*.pkl"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {side} spectra file for {DATE}, found {len(matches)}")
    with matches[0].open("rb") as file:
        spectrum = pickle.load(file)["df_spec"].copy()

    index = pd.DatetimeIndex(pd.to_datetime(spectrum.index))
    spectrum.index = index.tz_localize("UTC") if index.tz is None else index
    spectrum.index = spectrum.index.tz_convert(TIMEZONE)
    spectrum.columns = pd.to_numeric(spectrum.columns, errors="coerce")
    keep = (spectrum.columns >= freq_range[0]) & (spectrum.columns <= freq_range[1])
    spectrum = spectrum.loc[:, keep].apply(pd.to_numeric, errors="coerce")
    spectrum.columns = np.round(spectrum.columns.to_numpy(dtype=float), 1)
    return spectrum.resample("10s").mean()


def load_all(freq_range: tuple[float, float]):
    labels = {env: load_vlm_labels(BASE_DIR / f"{env}.csv") for env in ENVIRONMENTS}
    spectra = {side: load_spectrum(side, freq_range) for side in ("left", "right")}
    start = min(frame.index.min() for frame in labels.values())
    end = max(frame.index.max() for frame in labels.values())
    return labels, spectra, pd.date_range(start, end, freq=BIN)


def _time_extent(index: pd.DatetimeIndex) -> tuple[float, float]:
    return (
        mdates.date2num(index[0].tz_localize(None)),
        mdates.date2num(index[-1].tz_localize(None)) + BIN.total_seconds() / 86400,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wide", action="store_true", help="Show 0-90 Hz instead of 0-40 Hz")
    args = parser.parse_args()
    freq_range = FREQ_RANGE_WIDE if args.wide else FREQ_RANGE_DEFAULT

    labels, spectra, index = load_all(freq_range)
    x_start, x_end = _time_extent(index)

    log_power = {side: np.log(frame.where(frame > 0)) for side, frame in spectra.items()}

    behaviors = {env: label_columns(labels[env]) for env in ENVIRONMENTS}
    heights = [len(behaviors["lounge"]), len(behaviors["sleep"]), 14, 14]
    fig, axes = plt.subplots(
        4, 1, figsize=(12, 11), sharex=True, gridspec_kw={"height_ratios": heights}, layout="constrained"
    )

    label_images = []
    for axis, env in zip(axes[:2], ENVIRONMENTS):
        data = labels[env][behaviors[env]].reindex(index).to_numpy(dtype=float).T
        label_images.append(
            axis.imshow(
                np.ma.masked_invalid(data),
                aspect="auto",
                interpolation="nearest",
                cmap="Greys",
                vmin=0,
                vmax=1,
                extent=(x_start, x_end, data.shape[0] - 0.5, -0.5),
            )
        )
        axis.set_yticks(range(len(behaviors[env])), behaviors[env], fontsize=8)
        axis.set_title(f"{env.title()} camera | VLM labels", fontsize=10, loc="left")

    spectrum_images = []
    for axis, side in zip(axes[2:], ("left", "right")):
        frame = log_power[side].reindex(index)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="white")
        spectrum_images.append(
            axis.pcolormesh(
                mdates.date2num(frame.index.tz_localize(None).to_pydatetime()),
                frame.columns.to_numpy(dtype=float),
                np.ma.masked_invalid(frame.to_numpy(dtype=float).T),
                shading="gouraud",
                cmap=cmap,
                vmin=LOG_VMIN,
                vmax=LOG_VMAX,
                rasterized=True,
            )
        )
        axis.set_ylim(*freq_range)
        axis.set_ylabel("Frequency (Hz)", fontsize=9)
        axis.set_title(f"{side.title()} | ln power", fontsize=10, loc="left")

    for axis in axes:
        axis.set_facecolor("white")
        axis.spines[["top", "right"]].set_visible(False)

    axes[-1].xaxis_date()
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].set_xlim(x_start, x_end)
    axes[-1].set_xlabel(f"Time on {DATE} ({TIMEZONE})")

    for image, axis_group, label in (
        (label_images[0], axes[:2], "Label"),
        (spectrum_images[0], axes[2:], "ln power"),
    ):
        colorbar = fig.colorbar(image, ax=list(axis_group), fraction=0.02, pad=0.01, shrink=0.8)
        colorbar.set_label(label, fontsize=9)
        colorbar.ax.tick_params(labelsize=8)

    suffix = "_0-90Hz" if args.wide else ""
    for extension, kwargs in (("pdf", {}), ("png", {"dpi": 200})):
        output = BASE_DIR / f"VLM_spectra{suffix}.{extension}"
        fig.savefig(output, **kwargs)
        print(f"Saved {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
