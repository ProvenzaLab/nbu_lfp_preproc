# Clean analysis workflow

This folder contains the analysis scripts, the already-concatenated visit spectra, and some generated results. The workflow below starts from those visit spectra, so it does not need to read files from the repository's parent folder.

## Files already here

- `nbu_concat_single_channel_TRBD001/`: visit-level spectra pickle files used by the analysis and plotting scripts.
- `Analysis.py`: fits daytime Specparam models and writes feature tables and fit curves.
- `plot_analysis.py`: makes figures from the outputs of `Analysis.py`.
- `plot_daily_time_frequency.py`: makes side-by-side daily/time-frequency plots from the visit spectra.
- `analysis_results/`, `plots/`, and `plots2/`: existing generated outputs; these are not required inputs.

## Files to copy into this folder

To rerun the analysis and plots from the visit-level spectra already in this folder, copy these three input files here:

1. `oura_scores.pkl` — Oura sleep-stage scores. The analysis expects entries keyed like `TRBD001_YYYY-MM-DD`.
2. `trbd001_nbu_sleep_onset.pkl` — sleep-onset intervals; it must contain `nbu_visit`, `sleep_start`, and `sleep_end` columns.
3. `TRBD Programming Videos(TRBD001).csv` — MADRS scores; it must contain `Date` and `MADRS` columns.

These files are not currently in `clean/`. The CSV and pickle files contain study data; use the appropriate authorized copies.

You do **not** need to copy `NBU_file_path_date_mapping.csv` or `TRBD001_all_stim_params.parq` for the workflow below. The visit-level spectra required by it are already in `nbu_concat_single_channel_TRBD001/`.

## Run order

Run commands from this folder, using a Python environment with the project's dependencies installed (`numpy`, `pandas`, `specparam`, `joblib`, `matplotlib`, `seaborn`, and `scipy`):

```bash
cd clean

# 1. Fit daytime spectra; writes analysis_results/
python Analysis.py \
  --input-dir nbu_concat_single_channel_TRBD001 \
  --oura oura_scores.pkl \
  --sleep-onset trbd001_nbu_sleep_onset.pkl \
  --madrs "TRBD Programming Videos(TRBD001).csv" \
  --output-dir analysis_results

# 2. Plot feature summaries grouped by visit (the default); writes analysis_results/figures/
python plot_analysis.py \
  --input-dir analysis_results \
  --output-dir analysis_results/figures

# Optional: instead group boxplots by non-responder vs. responder
python plot_analysis.py \
  --input-dir analysis_results \
  --output-dir analysis_results/figures \
  --boxplot-grouping response

# Optional: also generate PSD overlays and paper_figure_4_ephys
python plot_analysis.py \
  --input-dir analysis_results \
  --output-dir analysis_results/figures \
  --run-psd

# 3. Optional: make daily/time-frequency plots; writes plots2/
python plot_daily_time_frequency.py \
  --input-dir nbu_concat_single_channel_TRBD001 \
  --output-dir plots2 \
  --oura oura_scores.pkl \
  --sleep-onset trbd001_nbu_sleep_onset.pkl
```

Step 3 is independent of steps 1–2; it only needs the visit spectra and the two sleep-data files. The output folders are created automatically.

### Choosing the band feature

In `Analysis.py`, set `USE_BAND_POWER_SUM` near the top of the file before running:

- `False` (default): calculate fitted peak heights, as in the existing results.
- `True`: set the Specparam peak count to zero and calculate each band's sum of aperiodic-corrected power instead.

The feature columns are named with the selected metric (`*_peak_height` or `*_band_power_sum`). `plot_analysis.py` detects either format automatically. Rerun `Analysis.py` before plotting to regenerate matching results.

Boxplots are grouped by NBU visit by default, with one box per visit/date category and all successful 10-second fits in that category. Use `--boxplot-grouping response` to instead show non-responder and responder boxes with permutation-test statistics. These groupings are mutually exclusive. PSD figures are skipped by default; pass `--run-psd` to generate standalone PSD overlays and `paper_figure_4_ephys`. That figure shows the per-visit mean spectra, color-coded by visit/date period, in this order: left raw, right raw, left aperiodic-removed, and right aperiodic-removed. Its feature panels use the selected boxplot grouping.

## Recomputing the visit spectra

The commands above start from the visit pickle files already in `nbu_concat_single_channel_TRBD001/`. Rebuilding those files is a separate, earlier pipeline stage and is **not self-contained in this folder yet**:

- `compute_spectra_fooof_bandpower.py` needs the original cleaned LFP Parquet files and a channel/path mapping. The current mapping contains paths outside `clean/`, and the script uses a machine-specific absolute path to locate that mapping.
- `concat_nbu_ephys_singlech.py` needs the per-channel `spectra_highres_spectra_*.pkl` files and `TRBD001_all_stim_params.parq`. Its source-data directory is also hard-coded to a path outside `clean/`.

The per-channel source spectra and original LFP Parquet files are not included here. Copying only the mapping or stimulation Parquet file is therefore insufficient to rebuild the visit spectra. To make that earlier stage portable, the source data must also be copied into `clean/`, the mapping updated to point to those local copies, and the two scripts' machine-specific paths changed to local paths. Until then, use the visit-level spectra already provided above.
