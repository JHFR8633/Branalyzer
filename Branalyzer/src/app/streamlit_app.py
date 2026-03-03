from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline import run_pipeline

# Option A: Streamlit calls preprocessing directly for waveforms
import preprocessing as prep


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Branalyzer", layout="wide")
st.title("Branalyzer")
st.subheader("EEG Model Benchmarking Dashboard")


# -----------------------------
# Plotly waveform helper
# -----------------------------
def _downsample(data: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return data
    return data[:, ::factor]


def plot_waveforms_plotly(raw, picks: list[str], t_start: float, duration: float, ds_factor: int, title: str):
    if raw is None:
        return None

    sfreq = float(raw.info["sfreq"])
    start = max(0, int(t_start * sfreq))
    stop = max(start + 1, int((t_start + duration) * sfreq))

    ch_names = list(raw.ch_names)
    pick_idx = [ch_names.index(ch) for ch in picks if ch in ch_names]
    if not pick_idx:
        return None

    data, times = raw[pick_idx, start:stop]  # (n_ch, n_times)
    data = _downsample(data, ds_factor)
    times = times[::ds_factor]

    scale = np.nanstd(data) if np.nanstd(data) > 0 else 1.0
    offsets = np.arange(len(pick_idx))[::-1] * scale * 4

    fig = go.Figure()
    for i, idx in enumerate(pick_idx):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=data[i] + offsets[i],
                mode="lines",
                name=ch_names[idx],
                hovertemplate="t=%{x:.3f}<br>amp=%{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        height=460,
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(orientation="h"),
    )
    return fig


# -----------------------------
# Cache signal builds (avoids re-downloading/recomputing on every rerun)
# -----------------------------
@st.cache_resource
def build_signal_variants(subject: int):
    """
    Returns:
      raw_view     -> raw with avg ref + channel selection (good for viewing)
      filtered_view-> highpass + bandpass (Mu/Beta)
      ica_view     -> ICA-cleaned (currently same as filtered because ICA isn't implemented)
    """
    raw = prep.load_raw_db(subject)
    raw = prep.set_average_reference(raw)
    raw = prep.select_channels(raw, prep.MOTOR_CHANNELS)

    # Raw tab should show "raw after selection/reference" (still basically raw)
    raw_view = raw.copy()

    # Filtered tab
    filtered = raw.copy()
    filtered = prep.highpass_filter(filtered)      # 1 Hz highpass for ICA step
    filtered = prep.bandpass_filter(filtered)      # 8–30 Hz Mu/Beta

    # ICA-cleaned tab (right now ICA is not implemented → will be same as filtered)
    ica_clean = filtered.copy()
    ica_clean = prep.ica_artifact_removal(ica_clean)

    return raw_view, filtered, ica_clean


# -----------------------------
# Sidebar controls
# -----------------------------
with st.sidebar:
    st.header("Controls")

    subject = st.number_input("PhysioNet Subject", min_value=1, max_value=109, value=1, step=1)

    st.divider()
    st.subheader("Waveform Window")
    t_start = st.number_input("Start time (s)", min_value=0.0, value=0.0, step=1.0)
    duration = st.slider("Duration (s)", min_value=2, max_value=30, value=10, step=1)
    ds_factor = st.select_slider("Downsample factor", options=[1, 2, 3, 4, 5, 6, 8, 10, 12], value=6)

    st.caption("Tip: higher downsample = faster plots.")


# -----------------------------
# Benchmarking pipeline (backend)
# -----------------------------
with st.status("Running pipeline…", expanded=False) as status:
    pipeline_out = run_pipeline(int(subject))  # backend metrics (currently stub) (added now)
    status.update(label="Pipeline complete", state="complete")

results = pipeline_out.results
r0 = results[0] 
#r0, r1, r2 = results[:3]
#change back to :3 later when we have more models
# -----------------------------
# Signal Inspection (frontend)
# -----------------------------
st.divider()
st.header("Signal Inspection")
st.write("EEG channel viewer (Raw / Filtered / ICA-Cleaned).")

try:
    with st.status("Loading EEG + building signal variants…", expanded=False) as status:
        raw_view, filt_view, ica_view = build_signal_variants(int(subject))
        status.update(label="Signals ready", state="complete")

    # channel multiselect defaults
    all_channels = list(raw_view.ch_names)
    default_channels = all_channels[: min(7, len(all_channels))]  # match MOTOR_CHANNELS size-ish
    picks = st.multiselect("Channels", options=all_channels, default=default_channels)

    tab_raw, tab_filt, tab_ica = st.tabs(["Raw", "Filtered", "ICA-Cleaned"])

    with tab_raw:
        fig = plot_waveforms_plotly(raw_view, picks, t_start, float(duration), int(ds_factor), "Raw (avg ref + selected channels)")
        if fig is None:
            st.warning("Nothing to plot (check channel selection).")
        else:
            st.plotly_chart(fig, use_container_width=True)

    with tab_filt:
        fig = plot_waveforms_plotly(filt_view, picks, t_start, float(duration), int(ds_factor), "Filtered (1Hz highpass + 8–30Hz bandpass)")
        if fig is None:
            st.warning("Nothing to plot (check channel selection).")
        else:
            st.plotly_chart(fig, use_container_width=True)

    with tab_ica:
        st.caption("Note: ICA artifact removal is currently marked NOT IMPLEMENTED in preprocessing.py, so this may match the filtered signal for now.")
        fig = plot_waveforms_plotly(ica_view, picks, t_start, float(duration), int(ds_factor), "ICA-Cleaned (currently same as filtered until ICA implemented)")
        if fig is None:
            st.warning("Nothing to plot (check channel selection).")
        else:
            st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error("Could not load or preprocess EEG signals.")
    st.code(str(e))


# -----------------------------
# Model Benchmarking (backend output)
# -----------------------------
st.divider()
st.header("Model Benchmarking")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Accuracy", f"{r0.accuracy:.2f}")
with col2:
    st.metric("F1 Score", f"{r0.f1_score:.2f}") #will need changed
with col3:
    st.metric("Std Dev", f"{r0.std_dev:.2f}")

rows = []
for r in results:
    rows.append(
        {
            "Model": r.name,
            "Accuracy": r.accuracy,
            "F1 Score": r.f1_score,
            "Std Dev": r.std_dev,
            "Inference Time (s)": r.inference_time_s,
        }
    )

df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
st.dataframe(df, use_container_width=True, hide_index=True)

st.subheader("Confusion Matrix")
st.info("Will display once ModelResult.confusion_matrix is produced in the pipeline.")


# -----------------------------
# Event Log (future hook)
# -----------------------------
st.divider()
st.header("Event Log")
st.write("Ground truth vs. prediction timeline.")
st.info("Will display once ModelResult.predictions + ground truth are wired into the pipeline.")

with st.expander("Pipeline Notes"):
    st.write(pipeline_out.notes)
    st.write({"Subjects": pipeline_out.n_subjects, "Epochs": pipeline_out.n_epochs})