from __future__ import annotations

import streamlit as st

from ui.results import render_event_log, render_model_benchmarking
from ui.signal_inspection import render_signal_inspection
from ui.workflow_controls import render_load_data_dialog, render_stage_controls
from workflows.session_state import (
    clear_model_state,
    clear_preprocessing_state,
    initialize_app_state,
    mark_subject_warning,
    reset_for_new_subject,
)
from workflows.signal_workflow import (
    load_subject_data,
    run_model_stage,
    run_preprocessing_stage,
)


st.set_page_config(page_title="Branalyzer", layout="wide", initial_sidebar_state="collapsed")
st.title("Branalyzer")
st.subheader("EEG Model Benchmarking Dashboard")


_MODEL_OPTIONS = ["LDA", "SVM", "Random Forest"]


def handle_load_data(subject: int) -> None:
    """Load raw data for the chosen subject and reset downstream workflow state.

    This updates Streamlit session state for the loaded subject, raw signal views,
    and waveform navigation controls.
    """
    st.session_state["selected_subject"] = subject
    st.session_state["subject_modal_value"] = subject
    st.session_state["subject_modal_open"] = False
    st.session_state["pending_load_subject"] = None

    if mark_subject_warning(subject):
        reset_for_new_subject()

    loaded = load_subject_data(subject)
    st.session_state["loaded_subject"] = subject
    st.session_state["raw_view"] = loaded["raw_view"]
    st.session_state["available_channels"] = loaded["available_channels"]
    st.session_state["default_channels"] = loaded["default_channels"]
    st.session_state["data_loaded"] = True
    st.session_state["subject_warning_pending"] = False
    st.session_state["waveform_window_start"] = 0.0
    st.session_state["waveform_jump_to"] = 0
    st.session_state["waveform_jump_to_input"] = 0
    st.session_state["waveform_sync_jump_input"] = True
    clear_preprocessing_state()


def handle_run_preprocessing(subject: int) -> None:
    """Run preprocessing for the currently loaded subject and store derived artifacts.

    This requires loaded raw data, updates preprocessing-related session state,
    clears model results, and triggers a rerun after completion.
    """
    if mark_subject_warning(subject):
        reset_for_new_subject()

    if not st.session_state["data_loaded"] or st.session_state["loaded_subject"] != subject:
        st.warning("Load data for the selected subject before running preprocessing.")
        return

    with st.status("Preprocessing EEG data...", expanded=False) as status:
        preprocessed = run_preprocessing_stage(subject)
        status.update(label="Preprocessing complete", state="complete")

    st.session_state["loaded_subject"] = subject
    st.session_state["preprocessed_subject"] = subject
    st.session_state["raw_view"] = preprocessed["raw_view"]
    st.session_state["filtered_view"] = preprocessed["filtered_view"]
    st.session_state["ica_view"] = preprocessed["ica_view"]
    st.session_state["epochs"] = preprocessed["epochs"]
    st.session_state["data_loaded"] = True
    st.session_state["preprocessing_ready"] = True
    st.session_state["subject_warning_pending"] = False
    clear_model_state()
    st.rerun()


def handle_run_models(subject: int, run_lda: bool, run_svm: bool, run_rf: bool) -> None:
    """Run the selected models against the cached preprocessed epochs.

    The function validates model selection and preprocessing state, stores the
    pipeline outputs in session state, and reruns the app when finished.
    """
    if not (run_lda or run_svm or run_rf):
        st.warning("Select at least one model before running the pipeline.")
        return

    if not st.session_state["preprocessing_ready"] or st.session_state["preprocessed_subject"] != subject:
        st.warning("Run preprocessing for the selected subject before running models.")
        return

    with st.status("Running models...", expanded=False) as status:
        model_output = run_model_stage(st.session_state["epochs"], subject, run_lda, run_svm, run_rf)
        status.update(label="Models complete", state="complete")

    st.session_state["modeled_subject"] = subject
    st.session_state["pipeline_out"] = model_output["pipeline_out"]
    st.session_state["results"] = model_output["results"]
    st.session_state["last_model_flags"] = model_output["last_model_flags"]
    st.session_state["models_ready"] = True
    st.session_state["subject_warning_pending"] = False
    st.rerun()


def main() -> None:
    """Render the Streamlit dashboard and wire UI controls to workflow handlers.

    The page reads from session state on each rerun and delegates section rendering
    to the UI modules.
    """
    initialize_app_state()
    selected_subject = int(st.session_state["selected_subject"])
    dashboard_section = st.container()

    if st.session_state["subject_modal_open"]:
        render_load_data_dialog()

    if mark_subject_warning(selected_subject):
        st.info(
            f"Subject {selected_subject} is selected, but the currently loaded results belong to "
            f"subject {st.session_state['loaded_subject']}. Click a workflow button to switch."
        )

    st.divider()
    st.header("Model Settings")

    available_channels = st.session_state["available_channels"]
    default_channels = st.session_state["default_channels"]
    picks = st.multiselect(
        "Channels",
        options=available_channels,
        default=default_channels,
        disabled=not st.session_state["data_loaded"],
    )
    selected_models = st.multiselect(
        "Models",
        options=_MODEL_OPTIONS,
        default=_MODEL_OPTIONS,
        disabled=not st.session_state["data_loaded"],
        key="model_selection_multiselect",
    )
    run_lda = "LDA" in selected_models
    run_svm = "SVM" in selected_models
    run_rf = "Random Forest" in selected_models

    with dashboard_section:
        render_stage_controls(
            selected_subject,
            run_lda,
            run_svm,
            run_rf,
            on_load_data=handle_load_data,
            on_run_preprocessing=handle_run_preprocessing,
            on_run_models=handle_run_models,
        )

    render_signal_inspection(
        picks=picks,
    )
    render_model_benchmarking()
    render_event_log()


if __name__ == "__main__":
    main()
