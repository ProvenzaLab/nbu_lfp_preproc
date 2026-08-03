"""
NBU LFP Interactive Preprocessing Platform
===========================================

A local web app that walks a user through preprocessing and artifact-cleaning
Percept LFP data for a single patient/visit window:

  1. Load LFP data for the patient/date window under all three ECG-cleaning
     methods (raw, template, perceive).
  2. Step through each hour of available data, comparing the PSD across the
     three methods, and pick the cleanest one for that hour.
  3. Stitch the selected hourly chunks into one clean dataset.
  4. Flag high-amplitude artifacts (>10 SD peaks) and let the user approve or
     reject removal of the 10s window around each.
  5. Save the cleaned dataset and run the spectral/FOOOF/bandpower pipeline
     on it.

Run with:  streamlit run app.py   (opens http://localhost:8501)

Deployment options considered for this platform:
  - Streamlit (used here): minimal boilerplate, built-in session state,
    good fit for a linear step-by-step wizard.
  - Plotly Dash: more control over layout/callbacks, richer interactive
    plots, more boilerplate for the same wizard flow.
  - Panel (HoloViz): similar tradeoffs to Streamlit, pairs well with
    HoloViews/Bokeh if finer plot interactivity is wanted later.
  - Flask/FastAPI + custom HTML/JS: full control, but the wizard state
    machine has to be hand-built; only worth it for a multi-user/remote
    deployment.
"""

import streamlit as st
import matplotlib.pyplot as plt

from utils.config import STUDY_IDS, FS
from utils.cleaning import (
    load_all_ecg_methods,
    group_by_hour,
    get_available_hours,
    stitch_selected_hours,
    detect_outlier_peaks,
    remove_outlier_window,
    save_clean_dataset,
)
from utils.plotting_utils import plot_hour_psd_comparison, plot_outlier_window
import compute_spectra_fooof_bandpower as fooof_pipeline

st.set_page_config(page_title="NBU LFP Preprocessing Platform", layout="wide")


def validate_patient_id(pt):
    return bool(pt) and pt[:-3] in STUDY_IDS.keys() and pt[-3:] != '000'


def init_state():
    defaults = {
        'stage': 'load',
        'pt': '',
        'window': None,
        'hourly': None,
        'chs': None,
        'hours': None,
        'hour_idx': 0,
        'hour_selections': {},
        'clean_df': None,
        'outlier_candidates': None,
        'outlier_idx': 0,
        'removed_outliers': [],
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)


def reset_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()


def finish_hour_review():
    clean_df = stitch_selected_hours(st.session_state.hourly, st.session_state.hour_selections)
    st.session_state.clean_df = clean_df

    candidates = detect_outlier_peaks(clean_df, st.session_state.chs, FS, sd_threshold=10, pad_seconds=5)
    st.session_state.outlier_candidates = candidates
    st.session_state.outlier_idx = 0
    st.session_state.removed_outliers = []
    st.session_state.stage = 'outliers' if candidates else 'save'


def go_next_hour(selected_method):
    hour = st.session_state.hours[st.session_state.hour_idx]
    st.session_state.hour_selections[hour] = selected_method
    st.session_state.hour_idx += 1
    if st.session_state.hour_idx >= len(st.session_state.hours):
        finish_hour_review()


def go_back_hour():
    if st.session_state.hour_idx > 0:
        st.session_state.hour_idx -= 1


def handle_outlier(remove):
    idx = st.session_state.outlier_idx
    candidate = st.session_state.outlier_candidates[idx]
    if remove:
        remove_outlier_window(st.session_state.clean_df, candidate['channel'], candidate['win_start'], candidate['win_end'])
        st.session_state.removed_outliers.append(candidate)
    st.session_state.outlier_idx += 1
    if st.session_state.outlier_idx >= len(st.session_state.outlier_candidates):
        st.session_state.stage = 'save'


init_state()
st.title("NBU LFP Preprocessing Platform")
st.sidebar.button("Start over", on_click=reset_state)

if st.session_state.stage == 'load':
    st.header("1. Load visit data")

    pt = st.text_input("Patient ID (e.g. P015, AA006, TRBD001)", value=st.session_state.pt)
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start date")
    end_date = col2.date_input("End date")

    if st.button("Load Data"):
        if not validate_patient_id(pt):
            st.error(f"Unrecognized patient ID prefix. Must start with one of {list(STUDY_IDS.keys())}.")
        else:
            window = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            with st.spinner("Loading raw / template / perceive LFP data..."):
                
                dfs = load_all_ecg_methods(pt, window)

            if not dfs:
                st.error("No LFP data found for any ECG method in that window.")
            else:
                chs = [c for c in next(iter(dfs.values())).columns if 'LEFT' in c or 'RIGHT' in c]
                with st.spinner("Indexing data by hour..."):
                    hourly = group_by_hour(dfs)
                hours = get_available_hours(hourly)

                if not hours:
                    st.error("No hourly data available to review.")
                else:
                    st.session_state.pt = pt
                    st.session_state.window = window
                    st.session_state.hourly = hourly
                    st.session_state.chs = chs
                    st.session_state.hours = hours
                    st.session_state.hour_idx = 0
                    st.session_state.hour_selections = {}
                    st.session_state.stage = 'hours'
                    st.rerun()

elif st.session_state.stage == 'hours':
    hours = st.session_state.hours
    idx = st.session_state.hour_idx
    hour = hours[idx]

    st.header("2. Compare hourly PSDs")
    st.progress((idx) / len(hours))
    st.subheader(f"Hour {idx + 1} of {len(hours)}: {hour}")

    available_methods = [m for m, h in st.session_state.hourly.items() if hour in h]
    hour_dfs = {m: st.session_state.hourly[m][hour] for m in available_methods}

    with st.spinner("Rendering PSD comparison..."):
        fig = plot_hour_psd_comparison(hour_dfs, st.session_state.chs, hour, FS)
    st.pyplot(fig)
    plt.close(fig)

    default_method = st.session_state.hour_selections.get(hour, available_methods[0])
    selected_method = st.radio(
        "Cleanest method for this hour (applied to all channels)",
        available_methods,
        index=available_methods.index(default_method) if default_method in available_methods else 0,
        key=f"method_radio_{idx}",
    )

    col1, col2 = st.columns(2)
    if col1.button("Back", disabled=(idx == 0), key=f"back_{idx}"):
        go_back_hour()
        st.rerun()
    if col2.button("Next", key=f"next_{idx}"):
        go_next_hour(selected_method)
        st.rerun()

elif st.session_state.stage == 'outliers':
    candidates = st.session_state.outlier_candidates
    idx = st.session_state.outlier_idx
    candidate = candidates[idx]

    st.header("3. Review candidate artifacts (>10 SD peaks)")
    st.progress(idx / len(candidates))
    st.subheader(f"Candidate {idx + 1} of {len(candidates)}")
    st.write(f"Channel **{candidate['channel']}**, peak at **{candidate['peak_time']}**")

    with st.spinner("Rendering candidate window..."):
        fig = plot_outlier_window(
            st.session_state.clean_df, candidate['channel'], candidate['win_start'], candidate['win_end'], FS
        )
    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)
    if col1.button("Keep (no artifact)", key=f"keep_{idx}"):
        handle_outlier(False)
        st.rerun()
    if col2.button("Remove (artifact)", key=f"remove_{idx}"):
        handle_outlier(True)
        st.rerun()

elif st.session_state.stage == 'save':
    st.header("4. Save clean dataset & compute features")
    st.write(f"Hours reviewed: {len(st.session_state.hour_selections)}")
    st.write(f"Artifacts removed: {len(st.session_state.removed_outliers)}")

    if st.button("Save clean dataset and run spectral/FOOOF pipeline"):
        with st.spinner("Saving clean dataset..."):
            parq_path, meta_path = save_clean_dataset(
                st.session_state.clean_df,
                st.session_state.pt,
                st.session_state.window,
                st.session_state.hour_selections,
                st.session_state.removed_outliers,
            )

        progress_bar = st.progress(0.0)
        chs = st.session_state.chs
        for i, ch in enumerate(chs):
            with st.spinner(f"Running FOOOF/bandpower pipeline for {ch}..."):
                fooof_pipeline.run_pipeline(str(parq_path), ch)
            progress_bar.progress((i + 1) / len(chs))

        st.session_state.stage = 'done'
        st.session_state.parq_path = parq_path
        st.session_state.meta_path = meta_path
        st.rerun()

elif st.session_state.stage == 'done':
    st.header("Done")
    st.success("Clean dataset saved and spectral/FOOOF pipeline complete.")
    st.write(f"Clean dataset: `{st.session_state.parq_path}`")
    st.write(f"Selection metadata: `{st.session_state.meta_path}`")
    st.write(f"Feature files saved alongside the clean dataset as `spectra_fooof_bandpower_<channel>.pkl`.")
