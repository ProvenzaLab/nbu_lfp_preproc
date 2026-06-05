import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.signal import welch, spectrogram, butter, sosfilt

from utils.utils import find_non_nan_segments

def plot_hourly_psd(df: pd.DataFrame, data_col: str, fs: float = 1.0,
                   nperseg: int = None, freq_range: tuple = None,
                   mode: str = 'raw',
                   **welch_kwargs):
    """
    Compute and plot the Power Spectral Density (PSD) for each hour of data
    on a single figure.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing a 'CT_times' column (pandas DatetimeSeries)
        and a data column to compute the PSD for.
    data_col : str
        Name of the column containing the signal data.
    fs : float, optional
        Sampling frequency in Hz. Default is 1.0.
    nperseg : int, optional
        Length of each segment for Welch's method. Defaults to min(256, len(segment)).
    **welch_kwargs
        Additional keyword arguments passed to scipy.signal.welch.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing all hourly PSDs.
    ax : matplotlib.axes.Axes
        The axes object.
    results : dict
        Dictionary mapping each hour (as a Timestamp) to a tuple (frequencies, psd).
        Frequencies and PSD values are already trimmed to freq_range if provided.

    """
    # --- Validate inputs ---
    if "CT_times" not in df.columns:
        raise ValueError("DataFrame must contain a 'CT_times' column.")
    if data_col not in df.columns:
        raise ValueError(f"Column '{data_col}' not found in DataFrame.")

    # --- Prepare data ---
    df = df.copy()
    df["CT_times"] = pd.to_datetime(df["CT_times"])
    df[data_col] = pd.to_numeric(df[data_col], errors="coerce").fillna(0)

    # Group by calendar hour
    df["_hour"] = df["CT_times"].dt.floor("h")
    grouped = df.groupby("_hour")

    if len(grouped) == 0:
        raise ValueError("No data found after grouping by hour.")

    # --- Plot ---
    fig, axes = plt.subplots(len(grouped), 3, figsize=(15, len(grouped)*5))
    results = {}

    for i, (hour, group) in enumerate(grouped):
        signal = group[data_col].to_numpy()
        # Welch PSD
        seg_len = nperseg if nperseg is not None else min(256, len(signal))
        freqs, psd = welch(signal, fs=fs, nperseg=seg_len, **welch_kwargs)

        # Plot PSD for hour full range and focused at 0-30 Hz
        for j, range in enumerate([(0, 120), (0, 30)]):
            
            mask = (freqs >= range[0]) & (freqs <= range[1])
            freqs, psd = freqs[mask], psd[mask]

            results[hour] = (freqs, psd)
            ax = axes[i, j]
            ax.semilogy(freqs, psd, label=str(hour), linewidth=1.5)
            ax.set_xlabel("Frequency (Hz)", fontsize=12)
            ax.set_ylabel("Power Spectral Density", fontsize=12)
            ax.set_title(f'PSD - {hour}', fontsize=14)
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

        # Plot random 10 sec chunk of data
        ax = axes[i, 2]
        if len(signal) > fs * 10:
            start_idx = fs * 600
            time_chunk = group["CT_times"].iloc[start_idx:start_idx + fs * 10]
            signal_chunk = signal[start_idx:start_idx + fs * 10]

            sos = butter(5, 100, 'low', fs=fs, output='sos')
            signal_chunk = sosfilt(sos, signal_chunk)

            ax.plot(time_chunk, signal_chunk)
            ax.set_xticks(time_chunk[::fs * 2], np.arange(0, 10, 2)) # Label every 2 seconds
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Signal", fontsize=12)
            ax.set_title(f"{mode} plots - {data_col}", fontsize=14)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz='America/Chicago'))

    fig.tight_layout()

    return fig, ax, results

def plot_lfp_spectrogram(lfp_df, ch, fs, nperseg, max_freq, ax):
    segments = find_non_nan_segments(lfp_df[ch].values)
    valid_segments = [(s, e) for s, e in segments if (e - s) >= nperseg]

    if not valid_segments:
        print(f"No valid segments for channel {ch}, skipping.")
        return

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='white')

    for start, end in valid_segments:
        if end - start < nperseg:
            continue

        lfp_values = lfp_df[ch].values[start:end]
        times      = lfp_df['CT_times'].values[start:end]

        f, t_bins, Sxx = spectrogram(lfp_values, fs, nperseg=nperseg, noverlap=nperseg//2)
        freq_mask = f <= max_freq

        Sxx_db = np.clip(10 * np.log10(Sxx[freq_mask]), -50, 0)

        ax.imshow(
            Sxx_db, aspect='auto', origin='lower',
            extent=[times[0], times[-1],
                    f[freq_mask].min(), f[freq_mask].max()],
            cmap=cmap, vmin=-50, vmax=0,  # fix color scale across segments
        )

    ax.set_title(f'LFP [{ch}]')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlim(lfp_df['CT_times'].min(), lfp_df['CT_times'].max())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p', tz='America/Chicago'))

    return ax