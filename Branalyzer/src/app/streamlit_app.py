from __future__ import annotations

import tempfile
import os
from data_ingest import load_eegbci_subject, load_user_edf, extract_annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ml_pipeline import run_pipeline
import preprocessing as prep


# Page config
st.set_page_config(page_title="Branalyzer", layout="wide")
st.title("Branalyzer")
st.subheader("EEG Model Benchmarking Dashboard")


# Plotly waveform helper
def _downsample(data: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return data
    return data[:, ::factor]


# scrolls through the recording with play/pause
def plot_waveforms_plotly(
    raw,
    picks: list[str],
    t_start: float,
    duration: float,
    ds_factor: int,
    title: str,
    scroll_step: float = 0.5,
    scroll_seconds: float = 30.0,
    frame_duration_ms: int = 200,
):
    if raw is None:
        return None

    sfreq = float(raw.info["sfreq"])
    total_time = raw.n_times / sfreq

    ch_names = list(raw.ch_names)
    pick_idx = [ch_names.index(ch) for ch in picks if ch in ch_names]
    if not pick_idx:
        return None

    # all data at once, slice per frame
    all_data, all_times = raw[pick_idx, :]
    all_data = _downsample(all_data, ds_factor)
    all_times = all_times[::ds_factor]
    ds_sfreq = sfreq / ds_factor

    # y offsets between channels
    global_scale = np.nanstd(all_data) if np.nanstd(all_data) > 0 else 1.0
    offsets = np.arange(len(pick_idx))[::-1] * global_scale * 4

    def _window(t0):
        i0 = max(0, int(t0 * ds_sfreq))
        i1 = min(len(all_times), int((t0 + duration) * ds_sfreq))
        if i1 <= i0:
            i1 = i0 + 1
        return all_data[:, i0:i1], all_times[i0:i1]

    # initial traces
    win_data, win_times = _window(t_start)
    fig = go.Figure()
    for i, idx in enumerate(pick_idx):
        fig.add_trace(
            go.Scatter(
                x=win_times,
                y=win_data[i] + offsets[i],
                mode="lines",
                name=ch_names[idx],
                hovertemplate="t=%{x:.3f}s<br>amp=%{y:.3f}<extra></extra>",
            )
        )

    # fixed y range
    y_min = float(np.min(all_data) + offsets[-1]) if len(offsets) else 0
    y_max = float(np.max(all_data) + offsets[0]) if len(offsets) else 1
    y_pad = (y_max - y_min) * 0.05

    # each frame shifts the window forward
    end_t = min(t_start + scroll_seconds, total_time - duration)
    frame_starts = np.arange(t_start, end_t, scroll_step)

    frames = []
    slider_steps = []
    for fi, fs in enumerate(frame_starts):
        fd, ft = _window(fs)
        frame_data = []
        for i in range(len(pick_idx)):
            frame_data.append(go.Scatter(x=ft, y=fd[i] + offsets[i]))

        frames.append(go.Frame(
            data=frame_data,
            name=str(fi),
            layout=go.Layout(
                xaxis=dict(range=[float(fs), float(fs + duration)]),
            ),
        ))
        slider_steps.append({
            "args": [[str(fi)], {"frame": {"duration": frame_duration_ms, "redraw": True}, "mode": "immediate"}],
            "label": f"{fs:.1f}s",
            "method": "animate",
        })

    fig.frames = frames

    fig.update_layout(
        title=title,
        height=500,
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h"),
        xaxis=dict(title="Time (s)", range=[t_start, t_start + duration]),
        yaxis=dict(range=[y_min - y_pad, y_max + y_pad]),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 1.0,
                "y": 1.18,
                "xanchor": "right",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_duration_ms, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "steps": slider_steps,
                "x": 0.0,
                "len": 1.0,
                "xanchor": "left",
                "y": -0.02,
                "yanchor": "top",
                "currentvalue": {"prefix": "Window start: ", "visible": True},
                "transition": {"duration": 0},
            }
        ] if slider_steps else [],
    )
    return fig

# load raw + avg ref
@st.cache_resource
def load_and_ref(subject: int):
    raw = prep.load_raw_db(subject)
    raw = prep.set_average_reference(raw)
    return raw

# ICA (reused by signal viewer and epoch pipeline)
@st.cache_resource
def run_ica_cached(subject: int):
    raw = load_and_ref(subject)
    return prep.run_ica_auto(raw.copy())

# epochs
@st.cache_resource
def get_epochs(subject: int):
    ica_clean = run_ica_cached(subject)
    selected = prep.select_channels(ica_clean.copy(), prep.MOTOR_CHANNELS)
    filtered = prep.bandpass_mu_beta(selected, l_freq=8.0, h_freq=30.0)
    return prep.make_epochs(filtered, tmin=0.0, tmax=4.0)

@st.cache_resource
def build_signal_variants(subject: int):
    """
    Returns:
      raw_view      -> avg ref + selected channels (for display only)
      filtered_view -> avg ref + channel selection + 1-78 Hz bandpass
      ica_view      -> avg ref + auto ICA artifact removal + channel selection + 8-30 Hz bandpass
    """
    raw = load_and_ref(subject)

    # Raw 
    raw_view = raw.copy()
    raw_view = prep.select_channels(raw_view, prep.MOTOR_CHANNELS)

    # Filtered 
    filtered = raw.copy()
    filtered = prep.highlowpass_for_ica(filtered, l_freq=1.0, requested_h_freq=100.0)
    filtered = prep.select_channels(filtered, prep.MOTOR_CHANNELS)
    filtered = prep.bandpass_mu_beta(filtered)

    # ICA-cleaned (reuses ICA above)
    ica_clean = run_ica_cached(subject)
    ica_view = ica_clean.copy()
    ica_view = prep.select_channels(ica_view, prep.MOTOR_CHANNELS)
    ica_view = prep.bandpass_mu_beta(ica_view)

    return raw_view, filtered, ica_view

@st.cache_resource
def build_signal_variants_from_raw(_raw, cache_key:int):
    raw = _raw.copy()

    # Raw
    raw_view = prep.set_average_reference(raw)

    # Filtered
    filtered = raw.copy()
    filtered = prep.highlowpass_for_ica(filtered, l_freq=1.0, requested_h_freq=100.0)
    filtered = prep.bandpass_mu_beta(filtered)

    # ICA-cleaned
    ica_clean = prep.run_ica_auto(raw.copy())
    ica_view = ica_clean.copy()
    ica_view = prep.bandpass_mu_beta(ica_view)

    return raw_view, filtered, ica_view

@st.cache_resource
def load_physionet(subject: int):
    return load_eegbci_subject(subject, runs=[4, 8, 12])

@st.cache_resource
def load_user_data(file_keys: tuple[str, ...], _file_paths: list[str]):
    # When temp file is created in file_upload_helper, the path is changed on each run (i.e., a slider change)
    # Because Streamlit is caching these inputs, it is different each time and re-loads the files on each interaction.
    # To avoid this, we can use the file keys (e.g. "S001R01.edf" from PhysioNet) as the cache key.
    # Underscoring _file_paths to indiciate we don't want the file path to be the cache key, just the file name essentially.
    return load_user_edf(list(_file_paths))

@st.cache_resource
def run_preprocessing(_raw, cache_key: int):
    return prep.preprocessing(_raw)



def file_upload_helper(uploaded_files) -> list[str]:
    """ Streamlit's file_uploader returns in-memory bytes, not disk paths,
        so we need to write them to disk first, so our preprocessing pipeline can read them.
        https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader

        Input is a list of uploaded_files and returns a list of file paths to those temporary files."""
    temp = tempfile.mkdtemp()
    paths = []
    for file in uploaded_files:
        path = os.path.join(temp, file.name)
        with open(path, "wb") as output:
            output.write(file.getbuffer())
        paths.append(path)
    return paths

# Sidebar controls
with st.sidebar:
    st.header("Controls")

    source = st.selectbox("Data Source", options=["Upload EDF", "PhysioNet EEGBCI"], index=0)
    if source == "PhysioNet EEGBCI":
        subject = st.number_input("PhysioNet Subject", min_value=1, max_value=109, value=1, step=1)
    else:
        uploaded_files = st.file_uploader("Upload EDF file(s)", type=["edf", "edf+"], accept_multiple_files=True)

    st.divider()
    st.subheader("Waveform Window")
    t_start = st.number_input("Start time (s)", min_value=0.0, value=0.0, step=1.0)
    duration = st.slider("Duration (s)", min_value=2, max_value=30, value=10, step=1)
    ds_factor = st.select_slider("Downsample factor", options=[1, 2, 3, 4, 5, 6, 8, 10, 12], value=6)

    st.caption("Tip: higher downsample = faster plots.")

    st.divider()
    st.subheader("Animation")
    scroll_step = st.select_slider("Scroll step (s)", options=[0.25, 0.5, 1.0, 2.0], value=0.5)
    scroll_seconds = st.slider("Scroll range (s)", min_value=10, max_value=120, value=30, step=10)
    frame_speed = st.select_slider("Frame speed (ms)", options=[50, 100, 150, 200, 300, 500], value=200)

    st.divider()
    st.subheader("Models")
    run_lda = st.checkbox("LDA", value=True)
    run_svm = st.checkbox("SVM", value=True)
    run_rf = st.checkbox("Random Forest", value=True)

# Load data depending on data source
if source == "PhysioNet EEGBCI":
    raw = load_physionet(int(subject))
else:
    if not uploaded_files:
        st.warning("Please upload at least one EDF file to proceed.")
        st.stop()
    file_paths = file_upload_helper(uploaded_files)
    file_keys = tuple(f.name for f in uploaded_files)
    raw = load_user_data(file_keys, file_paths)

    st.sidebar.divider()
    st.sidebar.subheader("File Info")
    st.sidebar.write(f"Channels: {len(raw.ch_names)}")
    st.sidebar.write(f"Sampling Frequency: {raw.info['sfreq']} Hz")
    st.sidebar.write(f"Duration: {raw.n_times / raw.info['sfreq']:.2f} seconds")

    annotations = extract_annotations(raw)
    if annotations:
        st.sidebar.write("Annotations found:")
        for desc, count in annotations.items():
            st.sidebar.write(f"  {desc}: {count}")
    else:
        st.sidebar.write("No annotations found in the uploaded data.")
cache_key = int(subject) if source == "PhysioNet EEGBCI" else hash(file_keys)

# Preprocessing
with st.status("Preprocessing EEG data…", expanded=False) as status:
    cache_key = int(subject) if source == "PhysioNet EEGBCI" else hash(file_keys)
    epochs = run_preprocessing(raw, cache_key)
    status.update(label="Preprocessing complete", state="complete")

if not (run_lda or run_svm or run_rf):
    st.warning("Select at least one model in the sidebar.")
    st.stop()

# keyed on subject + model toggles so animation changes don't retrain
@st.cache_resource
def run_models(_epochs, cache_key: int, lda: bool, svm: bool, rf: bool):
    return run_pipeline(_epochs, run_lda=lda, run_svm=svm, run_rf=rf)



# Benchmarking pipeline
with st.status("Running models…", expanded=False) as status:
    pipeline_out = run_models(epochs, cache_key, run_lda, run_svm, run_rf)
    status.update(label="Models complete", state="complete")

results = pipeline_out.results


# Signal Inspection
st.divider()
st.header("Signal Inspection")
st.write("EEG channel viewer (Raw / Filtered / ICA-Cleaned).")

try:
    with st.status("Loading EEG + building signal variants…", expanded=False) as status:
        if source == "PhysioNet EEGBCI":
            raw_view, filt_view, ica_view = build_signal_variants(int(subject))
        else:
            raw_view, filt_view, ica_view = build_signal_variants_from_raw(raw, cache_key)
        status.update(label="Signals ready", state="complete")

    all_channels = list(raw_view.ch_names)
    default_channels = all_channels[: min(7, len(all_channels))]
    picks = st.multiselect("Channels", options=all_channels, default=default_channels)

    tab_raw, tab_filt, tab_ica = st.tabs(["Raw", "Filtered", "ICA-Cleaned"])

    with tab_raw:
        fig = plot_waveforms_plotly(
            raw_view, picks, t_start, float(duration), int(ds_factor),
            "Raw (avg ref + selected channels)",
            scroll_step=float(scroll_step),
            scroll_seconds=float(scroll_seconds),
            frame_duration_ms=int(frame_speed),
        )
        if fig is None:
            st.warning("Nothing to plot (check channel selection).")
        else:
            st.plotly_chart(fig, use_container_width=True)

    with tab_filt:
        fig = plot_waveforms_plotly(
            filt_view, picks, t_start, float(duration), int(ds_factor),
            "Filtered (1 Hz highpass + 8-30 Hz bandpass)",
            scroll_step=float(scroll_step),
            scroll_seconds=float(scroll_seconds),
            frame_duration_ms=int(frame_speed),
        )
        if fig is None:
            st.warning("Nothing to plot (check channel selection).")
        else:
            st.plotly_chart(fig, use_container_width=True)

    with tab_ica:
        fig = plot_waveforms_plotly(
            ica_view, picks, t_start, float(duration), int(ds_factor),
            "ICA-Cleaned",
            scroll_step=float(scroll_step),
            scroll_seconds=float(scroll_seconds),
            frame_duration_ms=int(frame_speed),
        )
        if fig is None:
            st.warning("Nothing to plot (check channel selection).")
        else:
            st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error("Could not load or preprocess EEG signals.")
    st.code(str(e))



# Model Benchmarking
st.divider()
st.header("Model Benchmarking")

best_performer = max(results, key=lambda r: r.accuracy)

cols = st.columns(4)
with cols[0]:
    st.metric("Best Model", best_performer.name)
with cols[1]:
    st.metric("Accuracy", f"{best_performer.accuracy:.4f}")
with cols[2]:
    st.metric("F1 Score", f"{best_performer.f1_score:.4f}")
with cols[3]:
    st.metric("Std Dev", f"{best_performer.std_dev:.4f}")

rows = []
for r in results:
    rows.append(
        {
            "Model": r.name + " + CSP",
            "Accuracy": r.accuracy,
            "F1 Score": r.f1_score,
            "Std Dev": r.std_dev,
            "Inference Time (s)": r.inference_time_s,
        }
    )

# Removed sorting (.sort_values("Accuracy", ascending=False)) to keep same order as columns.
df = pd.DataFrame(rows).round({"Accuracy": 4, "F1 Score": 4, "Std Dev": 4, "Inference Time (s)": 8})
st.dataframe(df, use_container_width=True, hide_index=True)

# Share color scale across confusion matrices
max_cm_value = max((r.confusion_matrix.max() if r.confusion_matrix is not None else 0 for r in results), default=0)

st.subheader("Confusion Matrix")
cols = st.columns(len(results))
for col, r in zip(cols, results):
    with col:
        cm = r.confusion_matrix
        if cm is not None:
            n_classes = cm.shape[0]
            class_labels = ["Left", "Right"][:n_classes] if n_classes <= 2 else ["Rest", "Left", "Right"][:n_classes]
            fig_cm = go.Figure(
                go.Heatmap(
                    z=cm,
                    x=class_labels,
                    y=class_labels,
                    colorscale="teal",
                    text=cm,
                    texttemplate="%{text}",
                    showscale=False,
                    zmin=0,
                    zmax=max_cm_value,
                )
            )
            fig_cm.update_layout(
                title=f"{r.name} — Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                height=280,
                width=280,
                margin=dict(l=20, r=20, t=45, b=20),
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info(f"{r.name}: No confusion matrix available.")



# Event Log
st.divider()
st.header("Event Log")
st.write("Ground truth vs. prediction timeline. Rows highlighted in red indicate epochs where the model disagrees with the ground truth label.")

_LABEL_NAMES_RAW = {1: "Rest", 2: "Left Fist", 3: "Right Fist"}
_LABEL_NAMES_BINARY = {0: "Left Fist", 1: "Right Fist"}

def _highlight_disagreements(row):
    """Red background for rows where prediction != ground truth."""
    color = "background-color: #4c1616" if row["Match"] == "❌" else ""
    return [color] * len(row)

for r in results:
    if r.predictions is None:
        st.info(f"{r.name}: No predictions available yet.")
        continue

    st.subheader(f"{r.name} — Event Log")
    preds = r.predictions
    truth = r.ground_truth

    pred_label_map = _LABEL_NAMES_BINARY if set(np.unique(preds).tolist()) <= {0, 1} else _LABEL_NAMES_RAW
    truth_label_map = _LABEL_NAMES_BINARY if (truth is not None and set(np.unique(truth).tolist()) <= {0, 1}) else _LABEL_NAMES_RAW

    rows_log = []
    for i, pred in enumerate(preds):
        gt = truth[i] if truth is not None else None
        match = "✅" if (gt is not None and int(pred) == int(gt)) else ("❌" if gt is not None else "—")
        rows_log.append({
            "Epoch #": i + 1,
            "Ground Truth": truth_label_map.get(int(gt), str(gt)) if gt is not None else "—",
            "Predicted": pred_label_map.get(int(pred), str(pred)),
            "Match": match,
        })

    df_log = pd.DataFrame(rows_log)
    n_errors = (df_log["Match"] == "❌").sum()
    n_total = len(df_log)
    st.caption(f"{n_errors} disagreements out of {n_total} epochs ({100 * n_errors / max(n_total, 1):.1f}% error rate)")

    styled = df_log.style.apply(_highlight_disagreements, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True, height=300)

with st.expander("Pipeline Notes"):
    st.write(pipeline_out.notes)
    st.write({"Subjects": pipeline_out.n_subjects, "Epochs": pipeline_out.n_epochs})