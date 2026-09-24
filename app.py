"""
NBU LFP Interactive Preprocessing Platform
===========================================

A local web app that walks a user through preprocessing and artifact-cleaning
Percept LFP data for a single patient/visit window:

  1. Load the raw (non-ECG-cleaned) LFP data for the patient/date window.
  2. Step through each hour of raw data (PSDs + the most complete 30s snippet
     for every channel) and label it clean, unclean, or remove it.
  3. For unclean hours, parse just that hour's session files with the template
     and perceive ECG-cleaning methods, compare them against raw, and pick the
     cleanest one (or remove the hour).
  4. Stitch the selected hourly chunks into one clean dataset.
  5. Flag high-amplitude artifacts (MAD peaks) and let the user approve or
     reject removal of the 10s window around each. Use template matching to remove
     similar artifacts automatically.
  6. Review a summary of the clean data: a full spectrogram for each channel.
  7. Save the cleaned dataset and run the spectral/FOOOF/bandpower pipeline
     on it.

Run with:  streamlit run app.py   (opens http://localhost:8501)

"""

import os

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils.config import STUDY_IDS, FS, get_events
from utils.cleaning import (
    ALT_ECG_METHODS,
    load_raw,
    group_by_hour,
    load_hour_methods,
    find_most_complete_window,
    stitch_selected_hours,
    detect_outlier_peaks,
    remove_outlier_window,
    extract_template,
    match_template,
    save_clean_dataset,
    save_marked_artifacts,
)
from utils.plotting_utils import (
    plot_hour_psd_comparison,
    plot_hour_raw_trace_comparison,
    plot_outlier_window,
    plot_full_spectrogram,
    plot_zscored_power_over_time,
)
import compute_spectra_fooof_bandpower as fooof_pipeline

st.set_page_config(page_title="NBU LFP Preprocessing Platform", layout="wide")

REMOVE_HOUR = 'removed'  # radio sentinel in the compare step: exclude this hour from the clean dataset
# Raw-hour review uses a wide view; the ECG-method comparison zooms in where ECG
# artifacts (low-frequency harmonics, individual heartbeats) are easiest to see.
SNIPPET_SECONDS = 30
PSD_MAX_FREQ = 100  # Hz
COMPARE_SNIPPET_SECONDS = 5
COMPARE_PSD_MAX_FREQ = 40  # Hz


def validate_patient_id(pt):
    return bool(pt) and pt[:-3] in STUDY_IDS.keys() and pt[-3:] != '000'


def init_state():
    defaults = {
        'stage': 'load',
        'pt': '',
        'window': None,
        'raw_hourly': None,
        'file_map': None,
        'method_hours': {},
        'stream_cache': {},
        'chs': None,
        'hours': None,
        'hour_idx': 0,
        'hour_step': 'label',
        'hour_selections': {},
        'clean_df': None,
        'outlier_candidates': None,
        'outlier_idx': 0,
        'removed_outliers': [],
        'artifact_templates': {},
        'auto_removed': [],
        'marked_artifacts': {},
        'events_df': pd.DataFrame(),
    }
    for key, val in defaults.items():
        st.session_state.setdefault(key, val)


def reset_state():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_state()


def finish_hour_review():
    clean_df = stitch_selected_hours(
        st.session_state.raw_hourly, st.session_state.method_hours, st.session_state.hour_selections
    )
    if clean_df.empty:
        st.session_state.hour_idx = len(st.session_state.hours) - 1
        st.session_state.hour_error = "Every hour was removed — keep at least one hour to continue."
        return
    st.session_state.clean_df = clean_df

    candidates = detect_outlier_peaks(clean_df, st.session_state.chs, FS, pad_seconds=5)
    st.session_state.outlier_candidates = candidates
    st.session_state.outlier_idx = 0
    st.session_state.removed_outliers = []
    st.session_state.artifact_templates = {}
    st.session_state.auto_removed = []
    st.session_state.marked_artifacts = {}
    st.session_state.events_df = get_events(st.session_state.pt, st.session_state.window)
    st.session_state.stage = 'outliers' if candidates else 'summary'


def select_hour(label, method):
    """Record the current hour's selection and move on to the next hour.

    label: 'clean' | 'unclean' | 'removed'; method: the dataset to keep, or None if removed.
    """
    hour = st.session_state.hours[st.session_state.hour_idx]
    st.session_state.hour_selections[hour] = {'label': label, 'method': method}
    st.session_state.hour_step = 'label'
    st.session_state.hour_idx += 1
    if st.session_state.hour_idx >= len(st.session_state.hours):
        finish_hour_review()


def go_back_hour():
    if st.session_state.hour_step == 'compare':
        st.session_state.hour_step = 'label'
    elif st.session_state.hour_idx > 0:
        st.session_state.hour_idx -= 1


def describe_selection(sel):
    if sel['label'] == 'removed':
        return "removed"
    return f"{sel['label']} (kept {sel['method']})"


def render_hour_plots(hour_dfs, chs, hour, seconds=SNIPPET_SECONDS, max_freq=PSD_MAX_FREQ):
    """PSDs (0-max_freq Hz) + the most complete `seconds`-long snippet (all channels
    valid) for each dataset in hour_dfs, side by side so both fit on screen at once."""
    # Only channels recorded this hour (chs spans every sensing config in the visit)
    chs = [ch for ch in chs if hour_dfs['raw'][ch].notna().any()] or chs
    psd_col, trace_col = st.columns(2)

    with st.spinner("Rendering PSD comparison..."):
        fig = plot_hour_psd_comparison(hour_dfs, chs, hour, FS, max_freq=max_freq)
    psd_col.pyplot(fig)
    plt.close(fig)

    with st.spinner("Rendering time-domain snippet..."):
        t0 = find_most_complete_window(hour_dfs['raw'], chs, seconds=seconds)
        fig = plot_hour_raw_trace_comparison(hour_dfs, chs, t0, FS, seconds=seconds)
    trace_col.pyplot(fig)
    plt.close(fig)


def handle_outlier(remove):
    idx = st.session_state.outlier_idx
    candidate = st.session_state.outlier_candidates[idx]
    if remove:
        template = extract_template(st.session_state.clean_df, candidate['channel'], candidate['peak_time'], FS)
        if template is not None:
            st.session_state.artifact_templates.setdefault(candidate['channel'], []).append(template)

        remove_outlier_window(st.session_state.clean_df, candidate['channel'], candidate['win_start'], candidate['win_end'])
        candidate['method'] = 'manual'
        st.session_state.removed_outliers.append(candidate)
    st.session_state.outlier_idx += 1
    if st.session_state.outlier_idx >= len(st.session_state.outlier_candidates):
        st.session_state.stage = 'summary'


def handle_mark_artifact(note):
    """Flag the current candidate as an important artifact (e.g. an injected
    sync marker) with a note, without affecting the keep/remove decision."""
    idx = st.session_state.outlier_idx
    candidate = st.session_state.outlier_candidates[idx]
    st.session_state.marked_artifacts[idx] = {**candidate, 'note': note}


init_state()
st.title("NBU LFP Preprocessing Platform")
st.sidebar.button("Start over", on_click=reset_state)

st.sidebar.divider()
if st.sidebar.button("Shut down app"):
    st.sidebar.warning("Shutting down — close this browser tab.")
    os._exit(0)

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
            with st.spinner("Loading raw LFP data..."):
                raw_df, file_map = load_raw(pt, window)

            if raw_df is None or raw_df.empty:
                st.error("No raw LFP data found in that window.")
            else:
                chs = [c for c in raw_df.columns if 'LEFT' in c or 'RIGHT' in c]
                with st.spinner("Indexing data by hour..."):
                    raw_hourly = group_by_hour(raw_df)
                hours = sorted(raw_hourly)

                if not hours:
                    st.error("No hourly data available to review.")
                else:
                    st.session_state.pt = pt
                    st.session_state.window = window
                    st.session_state.raw_hourly = raw_hourly
                    st.session_state.file_map = file_map
                    st.session_state.method_hours = {}
                    st.session_state.stream_cache = {}
                    st.session_state.chs = chs
                    st.session_state.hours = hours
                    st.session_state.hour_idx = 0
                    st.session_state.hour_step = 'label'
                    st.session_state.hour_selections = {}
                    st.session_state.stage = 'hours'
                    st.rerun()

elif st.session_state.stage == 'hours':
    hours = st.session_state.hours
    idx = st.session_state.hour_idx
    hour = hours[idx]
    chs = st.session_state.chs
    raw_hour_df = st.session_state.raw_hourly[hour]
    prev = st.session_state.hour_selections.get(hour)
    step = st.session_state.hour_step

    st.header("2. Review hourly data")
    st.progress(idx / len(hours))
    st.subheader(f"Hour {idx + 1} of {len(hours)}: {hour}")

    hour_error = st.session_state.pop('hour_error', None)
    if hour_error:
        st.error(hour_error)
    if prev:
        st.caption(f"Previously marked: {describe_selection(prev)}")

    if step == 'label':
        render_hour_plots({'raw': raw_hour_df}, chs, hour)

        st.write("Is the raw data for this hour clean?")
        col1, col2, col3, col4 = st.columns(4)
        if col1.button("Back", disabled=(idx == 0), key=f"back_label_{idx}"):
            go_back_hour()
            st.rerun()
        if col2.button("Clean (keep raw)", key=f"clean_{idx}"):
            select_hour('clean', 'raw')
            st.rerun()
        if col3.button("Unclean (compare ECG methods)", key=f"unclean_{idx}"):
            st.session_state.hour_step = 'compare'
            st.rerun()
        if col4.button("Remove hour", key=f"remove_hour_{idx}"):
            select_hour('removed', None)
            st.rerun()

    else:
        if hour not in st.session_state.method_hours:
            with st.spinner(f"Parsing this hour with {' / '.join(ALT_ECG_METHODS)} ECG cleaning..."):
                st.session_state.method_hours[hour] = load_hour_methods(
                    st.session_state.pt, hour, raw_hour_df, st.session_state.file_map, st.session_state.stream_cache
                )
        hour_dfs = {'raw': raw_hour_df, **st.session_state.method_hours[hour]}

        missing = [m for m in ALT_ECG_METHODS if m not in hour_dfs]
        if missing:
            st.warning(f"No {' / '.join(missing)} data could be loaded for this hour.")

        render_hour_plots(hour_dfs, chs, hour,
                          seconds=COMPARE_SNIPPET_SECONDS, max_freq=COMPARE_PSD_MAX_FREQ)

        radio_options = list(hour_dfs) + [REMOVE_HOUR]
        if prev is None or prev['label'] == 'clean':
            default = radio_options[0]
        else:
            default = prev['method'] or REMOVE_HOUR
        selected = st.radio(
            "Dataset to keep for this hour (applied to all channels)",
            radio_options,
            index=radio_options.index(default) if default in radio_options else 0,
            format_func=lambda m: "Remove this hour (exclude from dataset)" if m == REMOVE_HOUR else m,
            key=f"method_radio_{idx}",
        )

        col1, col2 = st.columns(2)
        if col1.button("Back", key=f"back_compare_{idx}"):
            go_back_hour()
            st.rerun()
        if col2.button("Next", key=f"next_{idx}"):
            if selected == REMOVE_HOUR:
                select_hour('removed', None)
            else:
                select_hour('unclean', selected)
            st.rerun()

elif st.session_state.stage == 'outliers':
    threshold = st.sidebar.slider(
        "Template match threshold", min_value=0.5, max_value=1.0, value=0.85, step=0.01,
        help="Candidates whose waveform shape matches a saved artifact template at or above "
             "this correlation are auto-removed without a manual click.",
    )

    candidates = st.session_state.outlier_candidates

    # Auto-advance past any not-yet-reviewed candidate that matches a template
    # saved from an earlier manual "Remove" on the same channel.
    while st.session_state.outlier_idx < len(candidates):
        idx = st.session_state.outlier_idx
        candidate = candidates[idx]
        window = extract_template(st.session_state.clean_df, candidate['channel'], candidate['peak_time'], FS)
        score = match_template(window, st.session_state.artifact_templates.get(candidate['channel'], []))

        if score < threshold:
            break

        remove_outlier_window(st.session_state.clean_df, candidate['channel'], candidate['win_start'], candidate['win_end'])
        candidate['method'] = 'template_match'
        st.session_state.removed_outliers.append(candidate)
        st.session_state.auto_removed.append({**candidate, 'match_score': round(score, 3)})
        st.session_state.outlier_idx += 1

    idx = st.session_state.outlier_idx

    st.header("3. Review candidate artifacts (>10 SD peaks)")

    events_df = st.session_state.events_df

    if st.session_state.auto_removed:
        st.caption(f"{len(st.session_state.auto_removed)} candidate(s) auto-removed via template match so far.")

    if idx >= len(candidates):
        st.subheader("All candidates reviewed")
        if st.session_state.auto_removed:
            st.write("Auto-removed via template match:")
            st.dataframe(pd.DataFrame(st.session_state.auto_removed)[['channel', 'peak_time', 'match_score']])
        if st.session_state.marked_artifacts:
            st.write(f"Marked as important ({len(st.session_state.marked_artifacts)}):")
            st.dataframe(pd.DataFrame(st.session_state.marked_artifacts.values())[['channel', 'peak_time', 'note']])
        if st.button("Continue"):
            st.session_state.stage = 'summary'
            st.rerun()
    else:
        candidate = candidates[idx]
        st.progress(idx / len(candidates))
        st.subheader(f"Candidate {idx + 1} of {len(candidates)}")
        st.write(f"Channel **{candidate['channel']}**, peak at **{candidate['peak_time']}**")

        with st.spinner("Rendering candidate window..."):
            fig = plot_outlier_window(
                st.session_state.clean_df, candidate['channel'], candidate['win_start'], candidate['win_end'], events_df, FS
            )
        st.pyplot(fig)
        plt.close(fig)

        marked = st.session_state.marked_artifacts.get(idx)
        note = st.text_input(
            "Note (e.g. sync marker ID)",
            value=marked['note'] if marked else '',
            key=f"note_{idx}",
            help="Use this to flag artifacts intentionally injected into the data "
                 "stream (e.g. for later time-syncing) so they can be found again.",
        )

        col1, col2, col3 = st.columns(3)
        if col1.button("Keep (no artifact)", key=f"keep_{idx}"):
            handle_outlier(False)
            st.rerun()
        if col2.button("Remove (artifact)", key=f"remove_{idx}"):
            handle_outlier(True)
            st.rerun()
        if col3.button("Mark as important", key=f"mark_{idx}"):
            handle_mark_artifact(note)
            st.rerun()

        if marked:
            st.success(f"Marked as important: \"{marked['note']}\"")

elif st.session_state.stage == 'summary':
    st.header("4. Review clean data summary")
    st.write(f"Hours reviewed: {len(st.session_state.hour_selections)}")
    st.write(f"Artifacts removed: {len(st.session_state.removed_outliers)}")
    st.write(f"Artifacts marked as important: {len(st.session_state.marked_artifacts)}")

    with st.spinner("Rendering spectrogram of clean data..."):
        spec_fig = plot_full_spectrogram(st.session_state.clean_df, st.session_state.chs, FS)
    st.pyplot(spec_fig)
    plt.close(spec_fig)

    if st.button("Continue to save"):
        st.session_state.stage = 'save'
        st.rerun()

elif st.session_state.stage == 'save':
    st.header("5. Save clean dataset & compute features")
    st.write(f"Hours reviewed: {len(st.session_state.hour_selections)}")
    st.write(f"Artifacts removed: {len(st.session_state.removed_outliers)}")
    st.write(f"Artifacts marked as important: {len(st.session_state.marked_artifacts)}")

    if st.button("Save clean dataset and run spectral/FOOOF pipeline"):
        with st.spinner("Saving clean dataset..."):
            parq_path, meta_path = save_clean_dataset(
                st.session_state.clean_df,
                st.session_state.pt,
                st.session_state.window,
                st.session_state.hour_selections,
                st.session_state.removed_outliers,
            )
            marked_path = save_marked_artifacts(
                list(st.session_state.marked_artifacts.values()),
                st.session_state.pt,
                st.session_state.window,
            )

        progress_bar = st.progress(0.0)
        chs = st.session_state.chs
        for i, ch in enumerate(chs):
            with st.spinner(f"Running FOOOF/bandpower pipeline for {ch}..."):
                fooof_pipeline.run_pipeline(parq_path, ch)
            progress_bar.progress((i + 1) / len(chs))

        st.session_state.stage = 'done'
        st.session_state.parq_path = parq_path
        st.session_state.meta_path = meta_path
        st.session_state.marked_path = marked_path
        st.rerun()

elif st.session_state.stage == 'done':
    st.header("Done")
    st.success("Clean dataset saved and spectral/FOOOF pipeline complete.")
    st.write(f"Clean dataset: `{st.session_state.parq_path}`")
    st.write(f"Selection metadata: `{st.session_state.meta_path}`")
    if st.session_state.get('marked_path'):
        st.write(f"Marked artifacts: `{st.session_state.marked_path}`")
    st.write(f"Feature files saved alongside the clean dataset as `spectra_fooof_bandpower_<channel>.pkl`.")
