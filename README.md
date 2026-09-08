# NBU LFP Preprocessing

Tools for compiling, cleaning, and feature-extracting Percept LFP (local field
potential) recordings from patients' NeuroBehavioral Unit (NBU) inpatient
visits. The centerpiece is a local Streamlit app that walks a user through
comparing ECG-cleaning methods, removing movement/ECG artifacts, and saving a
clean dataset ready for spectral analysis.

## What's in this repo

- **`app.py`** — the interactive preprocessing platform (Streamlit web app).
  For a single patient/visit window, it:
  1. Loads LFP data under all three ECG-cleaning methods (`raw`, `template`,
     `perceive`).
  2. Steps through each hour of available data, comparing the PSD across the
     three methods, and lets the user pick the cleanest one for that hour.
  3. Stitches the selected hourly chunks into one dataset.
  4. Flags high-amplitude artifacts (MAD-based peak detection) for manual
     keep/remove review, and auto-removes further candidates that match a
     saved artifact template via cross-correlation.
  5. Shows a summary of the clean data (full spectrogram + z-scored power
     over time per channel).
  6. Saves the clean dataset and metadata, then runs the spectral/FOOOF/
     bandpower pipeline on it.
- **`compile_timedomain.py`** — parses raw Percept `.json` session files
  (via [`perceptparser`](https://github.com/ProvenzaLab/perceptparser)) into
  a single time-indexed dataframe per patient/visit/ECG-method, merges in
  stimulation settings, corrects tablet time drift, and writes the result to
  a `.parq` file. Can be run standalone to batch-compile all NBU visits for a
  study, or called on demand by the app.
- **`compute_spectra_fooof_bandpower.py`** — computes windowed PSDs
  (Welch's method), decomposes them into aperiodic/periodic components with
  [`specparam`](https://github.com/fooof-tools/fooof) (FOOOF), and derives
  per-band power features (delta/theta/alpha/beta/gamma + an "active
  sensing" band). Runs once per channel and is invoked automatically by the
  app after cleaning.
- **`view_visitLFP.py`** — standalone script for generating spectrogram/PSD
  summary plots for a saved visit dataset, without going through the app.
- **`utils/config.py`** — shared constants: network drive paths (data lake,
  lab worlds, project worlds), study ID prefixes, sampling rate, timezone,
  and helpers for locating NBU visit dates and behavioral events.
- **`utils/cleaning.py`** — core data-loading, hour-grouping, outlier
  detection/removal, template-matching, and save logic used by the app.
- **`utils/plotting_utils.py`** — matplotlib figure builders used by the app
  and by `view_visitLFP.py`.
- **`utils/utils.py`** — misc helpers (e.g. time-drift correction, NaN
  segment finding).

## Prerequisites

- Windows with access to the lab's mapped network drives:
  - `Z:` — data lake (raw Percept session files, behavioral events)
  - `Y:` — lab worlds (pipeline output: compiled/cleaned LFP data)
  - `X:` — project worlds (per-patient time-drift correction files)
- Python **3.13+**
- [`uv`](https://docs.astral.sh/uv/) for dependency management (a `uv.lock`
  is checked in)
- Access to the private [`perceptparser`](https://github.com/ProvenzaLab/perceptparser)
  repo (pulled automatically by `uv` as a git dependency — make sure your
  git credentials for GitHub are set up)

## Setup

1. **Clone the repo** (if you haven't already) and open a terminal in the
   project root:
   ```powershell
   git clone <repo-url>
   cd nbu_lfp_preproc
   ```
2. **Confirm the network drives are mapped** to `Z:`, `Y:`, and `X:` as
   described above. Update the paths in `utils/config.py` if your lab's
   drive letters differ.
3. **Install `uv`** if you don't have it (see the [uv install docs](https://docs.astral.sh/uv/getting-started/installation/)).
4. **Install dependencies**:
   ```powershell
   uv sync
   ```
   This creates a `.venv` and installs everything pinned in `uv.lock`,
   including `perceptparser` from its GitHub source.
5. **Launch the preprocessing app**:
   ```powershell
   uv run streamlit run app.py
   ```
   This opens the app in your browser at `http://localhost:8501`.

## Using the app

1. Enter a **patient ID** (e.g. `P015`, `AA006`, `TRBD001`) and the visit's
   **start/end dates**, then click **Load Data**. If a method's data hasn't
   been compiled yet, it's parsed from the raw session files and cached to
   `Y:\Provenza\NBUPipeline\<patient>\<start date>\rawlfp\<method>\`.
2. For each available hour, compare the PSD across ECG-cleaning methods and
   pick the cleanest one, then click **Next** (or **Back** to revise a
   previous hour).
3. Review each flagged high-amplitude candidate window and click **Keep** or
   **Remove**. Removing a candidate saves its waveform as a template; later
   candidates on the same channel that correlate above the sidebar
   threshold are auto-removed without further manual review.
4. Check the summary spectrogram and z-scored power plots for the cleaned
   data.
5. Click **Save clean dataset and run spectral/FOOOF pipeline** to write the
   clean `.parq` file and selection metadata to
   `Y:\Provenza\NBUPipeline\<patient>\<start date>\rawlfp\clean\`, and run
   the spectral/FOOOF/bandpower pipeline for each channel (saved alongside
   as `spectra_fooof_bandpower_<channel>.pkl`).

Use **Start over** in the sidebar to reset and process another visit, or
**Shut down app** to stop the server.

## Batch compilation

To compile raw time-domain data for many visits at once (outside the app),
edit the patient filter in `compile_timedomain.py`'s `__main__` block and
run:
```powershell
uv run python compile_timedomain.py
```
