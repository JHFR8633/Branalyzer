import streamlit as st
from preprocessing import preprocessing
import logging
import threading
import queue
import time


# ============================================================
# GLOBAL THREAD-SAFE LOG QUEUE
# (Accessible by both main thread and background thread)
# ============================================================

LOG_QUEUE = queue.Queue()


# ============================================================
# STREAMLIT PAGE CONFIG (MUST BE FIRST ST CALL)
# ============================================================

st.set_page_config(
    page_title="Branalyzer",
    layout="wide"
)

st.title("Branalyzer")
st.subheader("EEG Model Benchmarking Dashboard")


# ============================================================
# SESSION STATE INITIALIZATION (UI STATE ONLY)
# ============================================================

if "pipeline_thread" not in st.session_state:
    st.session_state.pipeline_thread = None

if "logs" not in st.session_state:
    st.session_state.logs = []

if "logger_initialized" not in st.session_state:

    # Custom logging handler that writes to GLOBAL queue
    class StreamlitLogHandler(logging.Handler):
        def emit(self, record):
            log_entry = self.format(record)
            LOG_QUEUE.put(log_entry)  # ✅ safe (no session_state)

    handler = StreamlitLogHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger = logging.getLogger()
    logger.handlers = []
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    st.session_state.logger_initialized = True

def run_preprocessing():
    """
    Runs preprocessing in background thread.
    Logging automatically streams into LOG_QUEUE.
    """
    preprocessing(subject=1)

st.divider()
st.header("Preprocessing Control")

col1, col2 = st.columns(2)

with col1:
    if st.button("Start Preprocessing", type="primary"):
        if (
            st.session_state.pipeline_thread is None
            or not st.session_state.pipeline_thread.is_alive()
        ):
            # Clear old logs
            st.session_state.logs = []

            # Start background thread
            thread = threading.Thread(target=run_preprocessing)
            thread.start()

            st.session_state.pipeline_thread = thread

with col2:
    if st.button("Clear Logs"):
        st.session_state.logs = []

st.divider()
st.header("Live Preprocessing Log")

log_placeholder = st.empty()

# Pull new logs from GLOBAL queue
while not LOG_QUEUE.empty():
    st.session_state.logs.append(LOG_QUEUE.get())

# Display logs
log_placeholder.text("\n".join(st.session_state.logs))

if (
    st.session_state.pipeline_thread is not None
    and st.session_state.pipeline_thread.is_alive()
):
    time.sleep(0.2)
    st.rerun()

st.divider()
st.header("Signal Inspection")
st.write("EEG channel viewer (Raw / Filtered / ICA-Cleaned).")

st.divider()
st.header("Model Benchmarking")

if (
    st.session_state.pipeline_thread is not None
    and not st.session_state.pipeline_thread.is_alive()
):
    st.success("Preprocessing complete. Model benchmarking is now available.")
else:
    st.info("Run preprocessing first to enable model benchmarking.")

st.divider()
st.header("Event Log")
st.write("Ground truth vs. prediction timeline.")