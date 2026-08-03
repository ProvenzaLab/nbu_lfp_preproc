import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy.signal import welch, spectrogram, butter, sosfilt

from utils.utils import find_non_nan_segments

def plot_hourly_psd(df: pd.DataFrame, data_col: str, fs: float = 1.0,
                   nperseg: int = None, mode: str = 'raw',
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
            ax = axes[i, j] if len(grouped) > 1 else axes[j]
            ax.semilogy(freqs, psd, label=str(hour), linewidth=1.5)
            ax.set_xlabel("Frequency (Hz)", fontsize=12)
            ax.set_ylabel("Power Spectral Density", fontsize=12)
            ax.set_title(f'PSD - {hour}', fontsize=14)
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)

        # Plot random 10 sec chunk of data
        ax = axes[i, 2] if len(grouped) > 1 else axes[2]
        if len(signal) > fs * 10:
            start_idx = fs * 600 if len(group) > fs * 1200 else fs * 10
            time_chunk = group["CT_times"].iloc[start_idx:start_idx + fs * 10]
            signal_chunk = signal[start_idx:start_idx + fs * 10]

            sos = butter(5, 100, 'low', fs=fs, output='sos')
            signal_chunk = sosfilt(sos, signal_chunk)

            ax.plot(time_chunk, signal_chunk)
            ax.set_xticks(time_chunk[::fs * 2], np.arange(0, 10, 2)) # Label every 2 seconds
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Signal", fontsize=12)
            ax.set_title(f"{mode} plots - {data_col}", fontsize=14)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S %p', tz='America/Chicago'))

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
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %I:%M %p', tz='America/Chicago'))

    return ax

def plot_hour_psd_comparison(hour_dfs_by_method, chs, hour, fs, nperseg=1000):
    """
    Compare the PSD (0-30 Hz) of each channel for a single hour across ECG-cleaning methods.

    Parameters
    ----------
    hour_dfs_by_method : dict[str, pd.DataFrame]
        Mapping of method label (e.g. 'raw', 'template', 'perceive') to the LFP dataframe
        already sliced to just this hour (e.g. via utils.cleaning.group_by_hour). Callers
        should only include methods that actually have data for this hour.
    chs : list[str]
        Channel columns to plot (one row each).
    hour : pd.Timestamp
        The hour being compared, used only for the figure title.
    fs : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Welch segment length, capped to the available samples for that hour.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    methods = [m for m, df in hour_dfs_by_method.items() if df is not None and not df.empty]
    if not methods:
        raise ValueError(f"No data available for hour {hour}")

    n_rows, n_cols = len(chs), len(methods)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for col, method in enumerate(methods):
        hour_df = hour_dfs_by_method[method]

        for row, ch in enumerate(chs):
            ax = axes[row, col]
            signal_vals = pd.to_numeric(hour_df[ch], errors='coerce').dropna().to_numpy()

            if len(signal_vals) < 2:
                ax.set_title(f'{method} - {ch}\n(no data)')
                ax.axis('off')
                continue

            seg_len = min(nperseg, len(signal_vals))
            freqs, psd = welch(signal_vals, fs=fs, nperseg=seg_len)
            mask = (freqs >= 0) & (freqs <= 30)

            ax.semilogy(freqs[mask], psd[mask], linewidth=1.2)
            ax.set_title(f'{method} - {ch}')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('PSD')
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

    fig.suptitle(f'Hourly PSD comparison — {hour}')
    fig.tight_layout()
    return fig

def plot_outlier_window(df, channel, win_start, win_end, fs):
    """
    Plot the time-series signal in a candidate artifact window for a single channel,
    for the user to review during artifact-removal.
    """
    window_df = df.loc[win_start:win_end]

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(window_df['CT_times'], window_df[channel], linewidth=1.0)
    ax.set_title(f'{channel} — candidate artifact ({win_start} to {win_end})')
    ax.set_xlabel('Time')
    ax.set_ylabel('Signal')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz='America/Chicago'))
    fig.tight_layout()

    return fig