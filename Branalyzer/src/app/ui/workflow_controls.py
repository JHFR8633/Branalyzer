from __future__ import annotations

from typing import Callable

import streamlit as st


@st.dialog("Load PhysioNet Data")
def render_load_data_dialog() -> None:
    """Render the subject picker dialog used by the load-data workflow stage.

    Confirming the form stores the requested subject in session state and triggers
    a rerun so the main page can execute the actual load step.
    """
    current_subject = int(st.session_state["subject_modal_value"])

    with st.form("load_data_form"):
        subject_value = st.number_input(
            "PhysioNet Subject",
            min_value=1,
            max_value=109,
            value=current_subject,
            step=1,
        )
        confirm_col, cancel_col = st.columns(2)
        with confirm_col:
            confirm = st.form_submit_button("Confirm", use_container_width=True)
        with cancel_col:
            cancel = st.form_submit_button("Cancel", use_container_width=True)

    if confirm:
        st.session_state["selected_subject"] = int(subject_value)
        st.session_state["subject_modal_value"] = int(subject_value)
        st.session_state["subject_modal_open"] = False
        st.session_state["pending_load_subject"] = int(subject_value)
        st.rerun()
        return
    if cancel:
        st.session_state["subject_modal_open"] = False
        st.rerun()
        return


def render_stage_controls(
    subject: int,
    run_lda: bool,
    run_svm: bool,
    run_rf: bool,
    on_load_data: Callable[[int], None],
    on_run_preprocessing: Callable[[int], None],
    on_run_models: Callable[[int, bool, bool, bool], None],
) -> None:
    """Render the three workflow buttons and their stage-specific status blocks.

    This section expects session state to already contain the current workflow
    flags and uses the provided callbacks to trigger each stage.
    """
    if st.session_state["subject_warning_pending"]:
        st.warning(
            "The selected subject differs from the currently loaded data. "
            "The next stage action will reset dependent results for the new subject."
        )

    can_preprocess = st.session_state["data_loaded"] and st.session_state["loaded_subject"] == subject
    can_run_models = st.session_state["preprocessing_ready"] and st.session_state["preprocessed_subject"] == subject

    col_load, col_preprocess, col_models = st.columns(3)
    with col_load:
        if st.button("Load Data", use_container_width=True):
            st.session_state["subject_modal_value"] = subject
            st.session_state["subject_modal_open"] = True
            st.rerun()
        pending_load_subject = st.session_state["pending_load_subject"]
        if pending_load_subject is not None:
            with st.status("Loading EEG data...", expanded=False) as status:
                on_load_data(int(pending_load_subject))
                status.update(
                    label=f"Data loaded for subject {st.session_state['loaded_subject']}",
                    state="complete",
                )
            st.rerun()
        render_stage_status(
            is_complete=st.session_state["data_loaded"] and st.session_state["loaded_subject"] is not None,
            loading_label="Loading EEG data...",
            complete_label=(
                f"Data loaded for subject {st.session_state['loaded_subject']}"
                if st.session_state["loaded_subject"] is not None
                else ""
            ),
        )
    with col_preprocess:
        if st.button("Run Preprocessing", disabled=not can_preprocess, use_container_width=True):
            on_run_preprocessing(subject)
        render_stage_status(
            is_complete=st.session_state["preprocessing_ready"] and st.session_state["preprocessed_subject"] is not None,
            loading_label="Preprocessing EEG data...",
            complete_label=(
                f"Preprocessing complete for subject {st.session_state['preprocessed_subject']}"
                if st.session_state["preprocessed_subject"] is not None
                else ""
            ),
        )
    with col_models:
        if st.button("Run Models", disabled=not can_run_models, use_container_width=True):
            on_run_models(subject, run_lda, run_svm, run_rf)
        model_flags = st.session_state.get("last_model_flags") or {}
        selected_models = [
            model_name
            for model_name, enabled in (
                ("LDA", model_flags.get("run_lda")),
                ("SVM", model_flags.get("run_svm")),
                ("Random Forest", model_flags.get("run_rf")),
            )
            if enabled
        ]
        model_summary = ", ".join(selected_models) if selected_models else "selected models"
        render_stage_status(
            is_complete=st.session_state["models_ready"] and st.session_state["modeled_subject"] is not None,
            loading_label="Running models...",
            complete_label=(
                f"Models complete for subject {st.session_state['modeled_subject']} ({model_summary})"
                if st.session_state["modeled_subject"] is not None
                else ""
            ),
        )


def render_stage_status(is_complete: bool, loading_label: str, complete_label: str) -> None:
    """Render a persistent status placeholder for a workflow stage.

    Completed stages reuse the same status UI footprint as their loading state so
    the layout stays stable between reruns.
    """
    if is_complete and complete_label:
        with st.status(loading_label, expanded=False, state="complete") as status:
            status.update(label=complete_label, state="complete")
    else:
        st.write("")
