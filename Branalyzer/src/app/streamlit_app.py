import streamlit as st
from pipeline import run_pipeline

st.set_page_config(
    page_title="Branalyzer",
    layout="wide"
)

st.title("Branalyzer")
st.subheader("EEG Model Benchmarking Dashboard")

pipeline_out = run_pipeline(None)
r0, r1, r2 = pipeline_out.results

st.divider()

st.header("Signal Inspection")
st.write("EEG channel viewer (Raw / Filtered / ICA-Cleaned).")

st.divider()

st.header("Model Benchmarking")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accuracy", f"{r0.accuracy:.2f}", help=r0.name)

with col2:
    st.metric("F1 Score", f"{r1.f1_score:.2f}", help=r1.name)

with col3:
    st.metric("Std Dev", f"{r2.std_dev:.2f}", help=r2.name)

st.write("Confusion matrix will be displayed here.")

st.divider()

st.header("Event Log")
st.write("Ground truth vs. prediction timeline.")
