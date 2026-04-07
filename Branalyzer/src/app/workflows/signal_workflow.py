from __future__ import annotations

import streamlit as st

from ml_pipeline import run_pipeline
import preprocessing as prep


@st.cache_resource
def load_and_ref(subject: int):
    """Load a subject's raw EEG and apply average reference once per cached subject."""
    raw = prep.load_raw_db(subject)
    raw = prep.set_average_reference(raw)
    return raw


@st.cache_resource
def run_ica_cached(subject: int):
    """Run ICA for a subject using the cached referenced raw signal."""
    raw = load_and_ref(subject)
    return prep.run_ica_auto(raw.copy())


@st.cache_resource
def get_epochs(subject: int):
    """Build cached motor-imagery epochs from the ICA-cleaned subject data."""
    ica_clean = run_ica_cached(subject)
    selected = prep.select_channels(ica_clean.copy(), prep.MOTOR_CHANNELS)
    filtered = prep.bandpass_mu_beta(selected, l_freq=8.0, h_freq=30.0)
    return prep.make_epochs(filtered, tmin=0.0, tmax=4.0)


@st.cache_resource
def build_signal_variants(subject: int):
    """Build cached raw, filtered, and ICA-cleaned views for signal inspection."""
    raw = load_and_ref(subject)

    raw_view = prep.select_channels(raw.copy(), prep.MOTOR_CHANNELS)

    filtered_view = raw.copy()
    filtered_view = prep.highlowpass_for_ica(filtered_view, l_freq=1.0, requested_h_freq=100.0)
    filtered_view = prep.select_channels(filtered_view, prep.MOTOR_CHANNELS)
    filtered_view = prep.bandpass_mu_beta(filtered_view)

    ica_view = run_ica_cached(subject)
    ica_view = prep.select_channels(ica_view.copy(), prep.MOTOR_CHANNELS)
    ica_view = prep.bandpass_mu_beta(ica_view)

    return raw_view, filtered_view, ica_view


@st.cache_resource
def run_models(_epochs, subject: int, lda: bool, svm: bool, rf: bool):
    """Run the selected model pipeline with Streamlit resource caching enabled."""
    return run_pipeline(_epochs, subject, run_lda=lda, run_svm=svm, run_rf=rf)


def load_subject_data(subject: int):
    """Load the raw inspection view and channel metadata for one subject."""
    raw_view = prep.select_channels(load_and_ref(subject).copy(), prep.MOTOR_CHANNELS)
    available_channels = list(raw_view.ch_names)
    default_channels = available_channels[: min(7, len(available_channels))]
    return {
        "raw_view": raw_view,
        "available_channels": available_channels,
        "default_channels": default_channels,
    }


def run_preprocessing_stage(subject: int):
    """Run the preprocessing stage and return all artifacts needed by the app UI."""
    filtered_view, ica_view = None, None
    raw_view, filtered_view, ica_view = build_signal_variants(subject)
    epochs = get_epochs(subject)
    return {
        "raw_view": raw_view,
        "filtered_view": filtered_view,
        "ica_view": ica_view,
        "epochs": epochs,
    }


def run_model_stage(epochs, subject: int, lda: bool, svm: bool, rf: bool):
    """Run the selected models and package the pipeline outputs for session state."""
    pipeline_out = run_models(epochs, subject, lda, svm, rf)
    return {
        "pipeline_out": pipeline_out,
        "results": pipeline_out.results,
        "last_model_flags": {
            "run_lda": lda,
            "run_svm": svm,
            "run_rf": rf,
        },
    }
