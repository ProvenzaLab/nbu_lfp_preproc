import pandas as pd
from scipy import signal
import numpy as np
from specparam import SpectralModel
from tqdm import tqdm
import os
import pickle
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import sys

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "low_beta": (12, 20),
    "high_beta": (20, 30),
    "gamma": (70, 100),
}

def load_nbu_signal(path, channel):
    df_nbu = pd.read_parquet(path)
    return df_nbu[channel], df_nbu

def resample_spectrogram(df_spec, rule="1min"):
    #return df_spec.resample(rule).apply(lambda x: np.nanmean(x, axis=0))
    return df_spec.resample(rule).mean()

def compute_psd_windows(
    timeseries,
    time_index,
    fs=250,
    window_seconds=10,
    step_seconds=0.2,
    only_int_frequencies=True,
    power_method="fft",
):
    window_samples = int(window_seconds * fs)
    step_samples = int(step_seconds * fs)
    if window_samples <= 0:
        raise ValueError("window_seconds must be > 0")
    if step_samples <= 0:
        raise ValueError("step_seconds must be > 0")

    power_method = str(power_method).lower()
    if power_method not in {"welch", "fft"}:
        raise ValueError("power_method must be either 'welch' or 'fft'")

    spectra = []
    times = []
    freqs = None

    for i in range(0, len(timeseries) - window_samples + 1, step_samples):
        if i + window_samples > len(timeseries):
            break

        window_data = timeseries[i : i + window_samples].values
        if power_method == "welch":
            freqs, pxx = signal.welch(
                window_data,
                fs=fs,
                nperseg=window_samples,
                noverlap=0,
            )
        else:
            fft_vals = np.fft.rfft(window_data)
            freqs = np.fft.rfftfreq(window_samples, d=1.0 / fs)
            # Plain FFT-based power estimate.
            pxx = (np.abs(fft_vals) ** 2) / window_samples

        spectra.append(pxx)
        times.append(time_index[i])

    if len(spectra) == 0:
        return pd.DataFrame(index=pd.Index([], dtype="datetime64[ns]")), np.asarray([])

    df_spec = pd.DataFrame(np.asarray(spectra), index=np.asarray(times), columns=freqs)
    if only_int_frequencies:
        df_spec = df_spec.loc[:, np.isclose(df_spec.columns, np.round(df_spec.columns))]
        freqs = df_spec.columns.to_numpy(dtype=float)
    return df_spec, np.asarray(freqs)

def compute_fooof_decomposition(df_spec, freqs, fit_range=(1, 100), integer_hz=True):
    freq_mask = (freqs >= fit_range[0]) & (freqs <= fit_range[1])
    if integer_hz:
        freq_mask &= np.isclose(freqs, np.round(freqs))

    fit_freqs = np.round(freqs[freq_mask]).astype(int) if integer_hz else freqs[freq_mask]

    aperiodic_features = []
    periodic_only_spectra = []
    aperiodic_only_spectra = []
    original_spectra = []
    

    for psd_row in tqdm(df_spec.to_numpy(), desc="Computing FOOOF decomposition"):

        psd_slice = np.asarray(psd_row[freq_mask], dtype=float)
        valid_idx = np.isfinite(psd_slice)

        if valid_idx.sum() < 3:
            aperiodic_features.append((np.nan, np.nan))
            periodic_only_spectra.append(np.full(fit_freqs.shape, np.nan, dtype=float))
            aperiodic_only_spectra.append(np.full(fit_freqs.shape, np.nan, dtype=float))
            original_spectra.append(np.full(fit_freqs.shape, np.nan, dtype=float))
            continue

        fm = SpectralModel(verbose=False)
        try:
            fm.fit(
                fit_freqs[valid_idx],
                np.maximum(psd_slice[valid_idx], np.finfo(float).tiny),
                list(fit_range),
            )
            offset, exponent = fm.results.params.aperiodic.params
            aperiodic_features.append((offset, exponent))

            periodic_spectrum = np.full(fit_freqs.shape, np.nan, dtype=float)
            periodic_spectrum[valid_idx] = fm.results._model.data.power_spectrum - fm.results.model._ap_fit
            aperiodic_only_spectra.append(fm.results.model._ap_fit)
            periodic_only_spectra.append(periodic_spectrum)
            original_spectra.append(psd_slice)
        except Exception:
            aperiodic_features.append((np.nan, np.nan))
            aperiodic_only_spectra.append(np.full(fit_freqs.shape, np.nan, dtype=float))
            periodic_only_spectra.append(np.full(fit_freqs.shape, np.nan, dtype=float))
            original_spectra.append(np.full(fit_freqs.shape, np.nan, dtype=float))

    df_aperiodic_params = pd.DataFrame(
        aperiodic_features,
        index=df_spec.index,
        columns=["offset", "exponent"],
    )
    df_periodic = pd.DataFrame(periodic_only_spectra, index=df_spec.index, columns=fit_freqs)
    df_aperiodic = pd.DataFrame(aperiodic_only_spectra, index=df_spec.index, columns=fit_freqs)
    df_spec_original = pd.DataFrame(original_spectra, index=df_spec.index, columns=fit_freqs)
    return df_aperiodic_params, df_aperiodic, df_periodic, df_spec_original

def compute_bandpower(df_periodic, df_aperiodic, bands):
    df_bandpower = pd.DataFrame(index=df_periodic.index)
    for band_name, (f_low, f_high) in bands.items():
        band_idx = (df_periodic.columns >= f_low) & (df_periodic.columns < f_high)
        df_bandpower[band_name] = df_periodic.loc[:, band_idx].mean(axis=1)

    df_bandpower[["offset", "exponent"]] = df_aperiodic[["offset", "exponent"]]
    return df_bandpower

def add_active_sensing_band(df_bandpower, df_spec, center_freq, half_bandwidth):
    raw_freq_columns = np.asarray(df_spec.columns, dtype=float)
    sensing_values = []

    for timestamp in df_bandpower.index:
        if timestamp not in df_spec.index:
            sensing_values.append(np.nan)
            continue

        band_mask = (raw_freq_columns >= center_freq - half_bandwidth) & (
            raw_freq_columns <= center_freq + half_bandwidth
        )
        if not np.any(band_mask):
            sensing_values.append(np.nan)
            continue

        sensing_values.append(
            np.nanmean(df_spec.loc[timestamp, band_mask].to_numpy(dtype=float))
        )

    df_bandpower = df_bandpower.copy()
    df_bandpower["active_sensing_band"] = sensing_values
    return df_bandpower

def get_files_and_channels(folder_="perceive"):

    path_check = f"NBU_file_path_date_mapping_{folder_}.csv"
    if os.path.exists(path_check):
        df_chs = pd.read_csv(path_check)
        return df_chs

    PATH_DEFAULT = "/mnt/labworlds/Provenza/NBUPipeline"
    subjects = os.listdir(PATH_DEFAULT)
    l_chs_ = []
    for subject in subjects:
        PATH_ = os.path.join(PATH_DEFAULT, subject)
        dates_ = os.listdir(PATH_)
        dates_ = [d for d in dates_ if os.path.isdir(os.path.join(PATH_, d))]

        for date in dates_:
            path_read = os.path.join(PATH_, date, "rawlfp", folder_)
            if not os.path.exists(path_read):
                continue

            files_parq = os.listdir(path_read)
            if len(files_parq) == 0:
                continue
            files_parq = [f for f in files_parq if f.endswith(".parq")]
            df = pd.read_parquet(os.path.join(path_read, files_parq[0]))
            channels = [ch for ch in list(df.columns) if ch not in ["filename", "CT_times", "lead_location", "lead_model", "ipg_hemisphere"]]
            for ch in channels:
                l_chs_.append({
                    "subject": subject,
                    "path": os.path.join(path_read, files_parq[0]),
                    "channel": ch,
                    "date": date,
                })
    df_chs = pd.DataFrame(l_chs_)
    df_chs.to_csv(f"NBU_file_path_date_mapping_{folder_}.csv", index=False)
    return df_chs
    

def run_pipeline(NBU_LFP_PATH, NBU_CHANNEL):
    ts, df_nbu = load_nbu_signal(NBU_LFP_PATH, NBU_CHANNEL)
    #df_spec, freqs = compute_psd_windows(ts, df_nbu.index, fs=250, window_seconds=10, only_int_frequencies=False)
    df_spec, freqs = compute_psd_windows(
        ts,
        df_nbu.index,
        fs=250,
        window_seconds=1,
        step_seconds=0.2,
        only_int_frequencies=False,
        power_method="fft",
    )

    #df_spec_1min = resample_spectrogram(df_spec, rule="1min")

    df_aperiodic_params, df_aperiodic, df_periodic, df_spec_original = compute_fooof_decomposition(
                df_spec,
                freqs,
                fit_range=(1, 100),
                integer_hz=False,
            )

    df_features = compute_bandpower(df_periodic, df_aperiodic_params, BANDS)
    df_features = add_active_sensing_band(
                df_features,
                df_spec,
                center_freq=8.79,
                half_bandwidth=5,
            )

    d_save = {
        "df_spec": df_spec,
        "df_features": df_features,
        "df_aperiodic_params": df_aperiodic_params,
        "df_aperiodic_spectra": df_aperiodic,
        "df_periodic_ap_removed": df_periodic,
        "freqs": freqs
    }
    # save in same directory as input file
    save_dir = os.path.dirname(NBU_LFP_PATH)
    save_path = os.path.join(save_dir, f"spectra_percept_fft_fooof_bandpower_{NBU_CHANNEL}.pkl")

    with open(save_path, "wb") as f:
        pickle.dump(d_save, f)
    


if __name__ == "__main__":

    l_ = get_files_and_channels("raw") # "perceive", "template"

    # for idx, row in l_.iterrows():
    #     NBU_LFP_PATH = row["path"]
    #     NBU_CHANNEL = row["channel"]
    #     run_pipeline(NBU_LFP_PATH, NBU_CHANNEL)

    run_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    run_pipeline(l_.iloc[run_idx]["path"], l_.iloc[run_idx]["channel"])

    # Parallel(n_jobs=-1)(
    #     delayed(run_pipeline)(row["path"], row["channel"])
    #     for idx, row in l_.iterrows()
    # )

    # l_ = get_files_and_channels("raw")

    # Parallel(n_jobs=-1)(
    #     delayed(run_pipeline)(row["path"], row["channxel"])
    #     for idx, row in l_.iterrows()
    # )

    # write nohup output to a file
    # nohup python compute_spectra_fooof_bandpower.py & > nohup_output.log 2>&1 &
