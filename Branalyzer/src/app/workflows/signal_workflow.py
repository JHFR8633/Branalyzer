from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

import data_ingest as ingest
from ml_pipeline import run_pipeline
import preprocessing as prep


PHYSIONET_SOURCE = "PhysioNet EEGBCI"
UPLOAD_SOURCE = "Upload EDF"


def _apply_average_reference(raw):
    """Return a copy of the raw signal with average reference applied."""
    referenced = raw.copy()
    referenced.set_eeg_reference("average")
    return referenced


def _default_channels(raw_view) -> list[str]:
    """Return the default channel selection used by the Streamlit viewer."""
    channel_names = list(raw_view.ch_names)
    return channel_names[: min(7, len(channel_names))]


def _write_uploaded_files(file_specs: tuple[tuple[str, bytes], ...]) -> list[str]:
    """Persist uploaded EDF bytes to temp files so MNE can read them from disk."""
    temp_dir = Path(tempfile.mkdtemp(prefix="branalyzer_upload_"))
    file_paths: list[str] = []
    for file_name, file_bytes in file_specs:
        file_path = temp_dir / file_name
        file_path.write_bytes(file_bytes)
        file_paths.append(str(file_path))
    return file_paths


def _ensure_preprocessing_backend() -> None:
    """Raise a clear error if the legacy preprocessing helpers are unavailable."""
    required_names = [
        "run_ica_auto",
        "select_channels",
        "bandpass_mu_beta",
        "highlowpass_for_ica",
        "make_epochs",
        "MOTOR_CHANNELS",
    ]
    missing = [name for name in required_names if not hasattr(prep, name)]
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(
            "The current preprocessing backend is missing required helpers: "
            f"{missing_text}. Loading raw data still works, but preprocessing "
            "and models need the updated preprocessing module before they can run."
        )


def _preprocessing_backend_available() -> bool:
    """Return whether the current preprocessing module exposes the required helpers."""
    try:
        _ensure_preprocessing_backend()
    except RuntimeError:
        return False
    return True


@st.cache_resource
def load_physionet_raw(subject: int, extended: bool = False):
    runs = [4, 6, 8, 10, 12, 14] if extended else [4, 8, 12]
    print(f"Loading subject {subject} with runs: {runs}")
    raw = ingest.load_eegbci_subject(subject, runs=runs)
    return _apply_average_reference(raw)


@st.cache_resource
def load_uploaded_raw(file_specs: tuple[tuple[str, bytes], ...]):
    """Load uploaded EDF files after materializing them to temp files."""
    file_paths = _write_uploaded_files(file_specs)
    raw = ingest.load_user_edf(file_paths)
    return _apply_average_reference(raw)


@st.cache_resource
def run_ica_cached(subject: int, extended: bool = False):
    _ensure_preprocessing_backend()
    raw = load_physionet_raw(subject, extended)
    return prep.run_ica_auto(raw.copy())


@st.cache_resource
def get_epochs(subject: int, extended: bool = False):
    _ensure_preprocessing_backend()
    ica_clean = run_ica_cached(subject, extended)
    selected = prep.select_channels(ica_clean.copy(), prep.MOTOR_CHANNELS)
    filtered = prep.bandpass_mu_beta(selected, l_freq=8.0, h_freq=30.0)
    return prep.make_epochs(filtered, tmin=0.0, tmax=4.0)


@st.cache_resource
def build_signal_variants(subject: int, extended: bool = False):
    _ensure_preprocessing_backend()
    raw = load_physionet_raw(subject, extended)

    raw_view = prep.select_channels(raw.copy(), prep.MOTOR_CHANNELS)

    filtered_view = raw.copy()
    filtered_view = prep.highlowpass_for_ica(filtered_view, l_freq=1.0, requested_h_freq=100.0)
    filtered_view = prep.select_channels(filtered_view, prep.MOTOR_CHANNELS)
    filtered_view = prep.bandpass_mu_beta(filtered_view)

    ica_view = run_ica_cached(subject, extended)
    ica_view = prep.select_channels(ica_view.copy(), prep.MOTOR_CHANNELS)
    ica_view = prep.bandpass_mu_beta(ica_view)

    return raw_view, filtered_view, ica_view


@st.cache_resource
def build_signal_variants_for_upload(file_specs: tuple[tuple[str, bytes], ...]):
    """Build raw, filtered, and ICA-cleaned signal views for uploaded EDF data."""
    _ensure_preprocessing_backend()
    raw = load_uploaded_raw(file_specs)

    raw_view = raw.copy()

    filtered_view = raw.copy()
    filtered_view = prep.highlowpass_for_ica(filtered_view, l_freq=1.0, requested_h_freq=100.0)
    filtered_view = prep.bandpass_mu_beta(filtered_view)

    ica_view = prep.run_ica_auto(raw.copy())
    ica_view = prep.bandpass_mu_beta(ica_view)

    return raw_view, filtered_view, ica_view


@st.cache_resource
def get_epochs_for_upload(file_specs: tuple[tuple[str, bytes], ...]):
    """Build epochs for uploaded EDF data using the shared preprocessing backend."""
    _ensure_preprocessing_backend()
    ica_clean = prep.run_ica_auto(load_uploaded_raw(file_specs).copy())
    filtered = prep.bandpass_mu_beta(ica_clean, l_freq=8.0, h_freq=30.0)
    return prep.make_epochs(filtered, tmin=0.0, tmax=4.0)


@st.cache_resource
def run_models(_epochs, subject_label: str, lda: bool, svm: bool, rf: bool, extended: bool = False):
    return run_pipeline(_epochs, subject=subject_label, run_lda=lda, run_svm=svm, run_rf=rf)


def load_subject_data(subject: int, extended_runs: bool = False):
    raw_view = load_physionet_raw(subject, extended_runs).copy()
    available_channels = list(raw_view.ch_names)
    return {
        "raw_view": raw_view,
        "available_channels": available_channels,
        "default_channels": _default_channels(raw_view),
        "loaded_source": PHYSIONET_SOURCE,
        "loaded_data_label": f"PhysioNet subject {subject}",
        "can_preprocess_loaded_data": _preprocessing_backend_available(),
        "loaded_file_names": [],
        "loaded_file_specs": None,
        "loaded_file_info": None,
        "loaded_annotations": {},
    }


def load_uploaded_data(file_specs: tuple[tuple[str, bytes], ...]):
    """Load uploaded EDF data for raw inspection and surface file metadata."""
    raw_view = load_uploaded_raw(file_specs).copy()
    available_channels = list(raw_view.ch_names)
    file_names = [file_name for file_name, _ in file_specs]
    duration_seconds = raw_view.n_times / float(raw_view.info["sfreq"])
    annotations = ingest.extract_annotations(raw_view)
    return {
        "raw_view": raw_view,
        "available_channels": available_channels,
        "default_channels": _default_channels(raw_view),
        "loaded_source": UPLOAD_SOURCE,
        "loaded_data_label": f"Uploaded EDF ({len(file_names)} file{'s' if len(file_names) != 1 else ''})",
        "can_preprocess_loaded_data": _preprocessing_backend_available(),
        "loaded_file_names": file_names,
        "loaded_file_specs": file_specs,
        "loaded_file_info": {
            "Channels": len(raw_view.ch_names),
            "Sampling Frequency": f"{float(raw_view.info['sfreq']):.2f} Hz",
            "Duration": f"{duration_seconds:.2f} s",
        },
        "loaded_annotations": annotations,
    }


def run_preprocessing_stage(subject: int | None = None, file_specs: tuple[tuple[str, bytes], ...] | None = None, extended_runs: bool = False):
    if file_specs is not None:
        raw_view, filtered_view, ica_view = build_signal_variants_for_upload(file_specs)
        epochs = get_epochs_for_upload(file_specs)
        data_label = f"Uploaded EDF ({len(file_specs)} file{'s' if len(file_specs) != 1 else ''})"
    elif subject is not None:
        raw_view, filtered_view, ica_view = build_signal_variants(subject, extended_runs)
        epochs = get_epochs(subject, extended_runs)
        data_label = f"PhysioNet subject {subject}"
    else:
        raise ValueError("run_preprocessing_stage requires either a subject or uploaded file specs.")

    return {
        "raw_view": raw_view,
        "filtered_view": filtered_view,
        "ica_view": ica_view,
        "epochs": epochs,
        "preprocessed_data_label": data_label,
    }


def run_model_stage(epochs, subject_label: str, lda: bool, svm: bool, rf: bool, extended: bool = False):
    pipeline_out = run_models(epochs, subject_label, lda, svm, rf, extended)
    return {
        "pipeline_out": pipeline_out,
        "results": pipeline_out.results,
        "modeled_data_label": str(subject_label),
        "last_model_flags": {
            "run_lda": lda,
            "run_svm": svm,
            "run_rf": rf,
        },
    }
