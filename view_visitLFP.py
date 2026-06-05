import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
matplotlib.rcParams['pdf.fonttype'] = 42
from scipy.signal import find_peaks
from scipy.stats import zscore

import os
import glob

from utils.config import FS, LAB_WORLDS
from utils.plotting_utils import plot_hourly_psd, plot_lfp_spectrogram

def plot_visit_details(df, chs, mode):

    rows = len(chs)
    nperseg = 1000

    sp_fig, sp_axs = plt.subplots(rows, 1, figsize=(12, 7*rows), sharex=True)

    for ch, ax in zip(chs, sp_axs):
        ax = plot_lfp_spectrogram(df, ch, fs=FS, nperseg=nperseg, max_freq=30, ax=ax)
        ax.set_title(f'{ch} Spectrogram')
    sp_fig.suptitle(f'{mode} Spectrograms')

    psd_figs = []
    for ch in chs:
        psd_fig, axes, results = plot_hourly_psd(df, ch, fs=FS, nperseg=nperseg, mode=mode)
        psd_fig.suptitle(f'{mode} Hourly PSDs - {ch}', x=0.5, y=1.05)
        psd_figs.append(psd_fig)
    
    return [sp_fig, *psd_figs]

def main(root):
    all_figs = []
    for mode in ['raw', 'perceive', 'template']:

        try:
            lfp_fp = glob.glob(f'{mode}/*LFP.parq', root_dir=root)[0]
            df = pd.read_parquet(os.path.join(root, lfp_fp))
        except:
            print(f'Error loading lfp data for {mode}')
            continue

        chs = [col for col in df.columns if 'LEFT' in col or 'RIGHT' in col]

        """ for ch in chs:

            # Remove large artifacts (mostly toggles, > 5 z-score)
            clean_lfp = zscore(df[ch].values, nan_policy='omit')
            peak_dist = FS * 5 
            peaks, peak_attr = find_peaks(abs(clean_lfp), height=5, distance=peak_dist)

            for peak in peaks:
                start = peak - peak_dist if peak <= peak_dist else 0
                end = peak + peak_dist if peak + peak_dist <= len(clean_lfp) else len(clean_lfp)

                df.iloc[start:end, df.columns.get_loc(ch)] = np.nan """

        figs = plot_visit_details(df, chs, mode)
        all_figs.extend(figs)

    with PdfPages(os.path.join(root, 'visit_preproc_analysis.pdf')) as pdf:
        for fig in all_figs:
            fig.tight_layout()
            pdf.savefig(fig)
        
        pdf.close()

    return

if __name__ == '__main__':
    visits = glob.glob(f'{LAB_WORLDS}/Provenza/NBUPipeline/**/*/')

    for root in visits:
        print(f'Processing visit: {root}')
        main(os.path.join(root, 'rawlfp'))