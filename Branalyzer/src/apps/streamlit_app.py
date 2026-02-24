import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Branalyzer", layout="wide")

st.title("Branalyzer")
st.caption("Modular EEG Analysis Toolkit — Streamlit Dashboard")

# Sidebar controls (placeholders)
st.sidebar.header("Controls")
view_mode = st.sidebar.selectbox("Signal View", ["Raw", "Filtered (8–30Hz)", "ICA-Cleaned"])
subject = st.sidebar.number_input("Subject ID", min_value=1, max_value=109, value=1, step=1)
epoch_len = st.sidebar.selectbox("Epoch length (sec)", [2, 4])

# Layout
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("Signal Viewer (placeholder)")
    # Fake signal so UI isn't blank while backend is being built
    t = np.linspace(0, 2, 500)
    y = np.sin(2 * np.pi * 10 * t) + 0.2 * np.random.randn(len(t))
    df = pd.DataFrame({"t": t, "amplitude": y})
    fig = px.line(df, x="t", y="amplitude", title=f"Mode: {view_mode} | Subject: {subject}")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Model Benchmarking (placeholder)")
    metrics = pd.DataFrame(
        [
            {"Model": "LDA", "Accuracy": 0.71, "F1": 0.69, "Std": 0.05, "Time (ms)": 12},
            {"Model": "SVM", "Accuracy": 0.74, "F1": 0.72, "Std": 0.04, "Time (ms)": 35},
            {"Model": "RF",  "Accuracy": 0.70, "F1": 0.68, "Std": 0.06, "Time (ms)": 18},
        ]
    )
    st.dataframe(metrics, use_container_width=True)

st.divider()

st.subheader("Event Timeline (placeholder)")
timeline = pd.DataFrame({
    "epoch": list(range(1, 21)),
    "ground_truth": np.random.choice(["Left", "Right", "Rest"], 20),
    "prediction": np.random.choice(["Left", "Right", "Rest"], 20),
})
st.dataframe(timeline, use_container_width=True)
st.info("Next step: replace placeholder data with real outputs from preprocessing + models.")

