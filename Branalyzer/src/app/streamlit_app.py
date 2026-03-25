from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ml_pipeline import run_pipeline
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

    data, times = raw[pick_idx, start:stop]
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
# Shared caching layer
# -----------------------------
# The key idea: load raw + run ICA ONCE per subject, then reuse
# that work for both the signal viewer AND the model pipeline.
# Previously these were independent, meaning ICA ran twice.

@st.cache_resource
def load_and_ref(subject: int) -> "mne.io.Raw":
    """Load raw EEGBCI data + set average reference. Cached per subject."""
    raw = prep.load_raw_db(subject)
    raw = prep.set_average_reference(raw)
    return raw


@st.cache_resource
def run_ica_cached(subject: int) -> "mne.io.Raw":
    """Run ICA artifact removal on avg-ref raw. Cached per subject."""
    raw = load_and_ref(subject)
    return prep.run_ica_auto(raw.copy())


@st.cache_resource
def get_epochs(subject: int) -> "mne.Epochs":
    """
    Full preprocessing -> epochs, reusing the cached ICA result.
    This is the expensive step that no longer reruns when you
    change model code or toggle model checkboxes.
    """
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

    NOTE: Reuses load_and_ref() and run_ica_cached() so ICA is
    never computed twice for the same subject.
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

    # ICA-cleaned (reuses cached ICA)
    ica_clean = run_ica_cached(subject)
    ica_view = ica_clean.copy()
    ica_view = prep.select_channels(ica_view, prep.MOTOR_CHANNELS)
    ica_view = prep.bandpass_mu_beta(ica_view)

    return raw_view, filtered, ica_view


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

    st.divider()
    st.subheader("Models")
    run_lda = st.checkbox("LDA", value=True)
    run_svm = st.checkbox("SVM", value=True)
    run_rf = st.checkbox("Random Forest", value=True)


# -----------------------------
# Preprocessing (cached — won't rerun on model changes)
# -----------------------------
with st.status("Preprocessing EEG data…", expanded=False) as status:
    epochs = get_epochs(int(subject))
    status.update(label="Preprocessing complete (cached)", state="complete")

# -----------------------------
# Benchmarking pipeline (only reruns models)
# -----------------------------
if not (run_lda or run_svm or run_rf):
    st.warning("Select at least one model in the sidebar.")
    st.stop()

with st.status("Running models…", expanded=False) as status:
    pipeline_out = run_pipeline(epochs, int(subject), run_lda=run_lda, run_svm=run_svm, run_rf=run_rf)
    status.update(label="Models complete", state="complete")

results = pipeline_out.results


# -----------------------------
# Signal Inspection
# -----------------------------
st.divider()
st.header("Signal Inspection")
st.write("EEG channel viewer (Raw / Filtered / ICA-Cleaned).")

try:
    with st.status("Loading EEG + building signal variants…", expanded=False) as status:
        raw_view, filt_view, ica_view = build_signal_variants(int(subject))
        status.update(label="Signals ready", state="complete")

    all_channels = list(raw_view.ch_names)
    default_channels = all_channels[: min(7, len(all_channels))]
    picks = st.multiselect("Channels", options=all_channels, default=default_channels)

    tab_raw, tab_filt, tab_ica = st.tabs(["Raw", "Filtered", "ICA-Cleaned"])

    with tab_raw:
        fig = plot_waveforms_plotly(
            raw_view,
            picks,
            t_start,
            float(duration),
            int(ds_factor),
            "Raw (avg ref + selected channels)",
        )
        if fig is None:
            st.warning("Nothing to plot (check channel selection).")
        else:
            st.plotly_chart(fig, use_container_width=True)

    with tab_filt:
        fig = plot_waveforms_plotly(
            filt_view,
            picks,
            t_start,
            float(duration),
            int(ds_factor),
            "Filtered (1 Hz highpass + 8-30 Hz bandpass)",
        )
        if fig is None:
            st.warning("Nothing to plot (check channel selection).")
        else:
            st.plotly_chart(fig, use_container_width=True)

    with tab_ica:
        fig = plot_waveforms_plotly(
            ica_view,
            picks,
            t_start,
            float(duration),
            int(ds_factor),
            "ICA-Cleaned",
        )
        if fig is None:
            st.warning("Nothing to plot (check channel selection).")
        else:
            st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error("Could not load or preprocess EEG signals.")
    st.code(str(e))


# -----------------------------
# Model Benchmarking
# -----------------------------
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


# -----------------------------
# Event Log
# -----------------------------
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