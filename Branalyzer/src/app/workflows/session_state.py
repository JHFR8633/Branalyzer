from __future__ import annotations

import streamlit as st


_DEFAULT_STATE = {
    "selected_source": "Upload EDF",
    "selected_subject": 1,
    "subject_modal_open": False,
    "subject_modal_source": "PhysioNet EEGBCI",
    "subject_modal_value": 1,
    "pending_load_request": None,
    "waveform_window_start": 0.0,
    "waveform_step_size": 1,
    "waveform_jump_to": 0,
    "waveform_jump_to_input": 0,
    "waveform_sync_jump_input": False,
    "waveform_ds_factor": 6,
    "waveform_window_size": 10,
    "waveform_frame_speed": 200,
    "waveform_autoplay": False,
    "loaded_subject": None,
    "loaded_source": None,
    "loaded_data_label": None,
    "preprocessed_subject": None,
    "preprocessed_data_label": None,
    "modeled_subject": None,
    "modeled_data_label": None,
    "raw_view": None,
    "available_channels": [],
    "default_channels": [],
    "loaded_file_names": [],
    "loaded_file_specs": None,
    "loaded_file_info": None,
    "loaded_annotations": {},
    "filtered_view": None,
    "ica_view": None,
    "epochs": None,
    "pipeline_out": None,
    "results": None,
    "last_model_flags": None,
    "data_loaded": False,
    "can_preprocess_loaded_data": False,
    "preprocessing_ready": False,
    "models_ready": False,
    "subject_warning_pending": False,
    "user_annotations": {},
    "user_assignments": {},
    "user_event_map": None,
    "user_code_map": None,
    "annotation_mapping_complete": False,
    "user_display_names": {},
    "_prev_class_a": None,
    "_prev_class_b": None,
    "class_a_display_name": None,
    "class_b_display_name": None,
    "anno_class_a": "—",
    "anno_class_b": "—",
    "anno_rest": "Rest",
}


def initialize_app_state() -> None:
    """Seed Streamlit session state with the app's default workflow and UI values."""
    for key, value in _DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


def subject_has_changed(selected_subject: int) -> bool:
    """Return whether the selected subject differs from the currently loaded subject."""
    if st.session_state.get("selected_source") != "PhysioNet EEGBCI":
        return False
    if st.session_state.get("loaded_source") != "PhysioNet EEGBCI":
        return False
    loaded_subject = st.session_state.get("loaded_subject")
    return loaded_subject is not None and loaded_subject != selected_subject


def clear_model_state() -> None:
    """Clear model outputs and readiness flags while leaving earlier stages intact."""
    st.session_state["modeled_subject"] = None
    st.session_state["pipeline_out"] = None
    st.session_state["results"] = None
    st.session_state["last_model_flags"] = None
    st.session_state["modeled_data_label"] = None
    st.session_state["models_ready"] = False


def clear_preprocessing_state() -> None:
    """Clear preprocessing artifacts and model state for the active session."""
    st.session_state["preprocessed_subject"] = None
    st.session_state["filtered_view"] = None
    st.session_state["ica_view"] = None
    st.session_state["epochs"] = None
    st.session_state["preprocessed_data_label"] = None
    st.session_state["preprocessing_ready"] = False
    clear_model_state()


def reset_for_new_subject() -> None:
    """Reset loaded data and all downstream state before switching subjects."""
    st.session_state["loaded_subject"] = None
    st.session_state["loaded_source"] = None
    st.session_state["loaded_data_label"] = None
    st.session_state["raw_view"] = None
    st.session_state["available_channels"] = []
    st.session_state["default_channels"] = []
    st.session_state["loaded_file_names"] = []
    st.session_state["loaded_file_specs"] = None
    st.session_state["loaded_file_info"] = None
    st.session_state["loaded_annotations"] = {}
    st.session_state["data_loaded"] = False
    st.session_state["can_preprocess_loaded_data"] = False
    st.session_state["subject_warning_pending"] = False
    clear_model_state()
    clear_preprocessing_state()
    clear_user_annotations()


def mark_subject_warning(selected_subject: int) -> bool:
    """Update and return the subject-change warning flag in session state."""
    changed = subject_has_changed(selected_subject)
    st.session_state["subject_warning_pending"] = changed
    return changed

def clear_user_annotations() -> None:
    """Clear user annotations and related state."""
    st.session_state["user_annotations"] = {}
    st.session_state["user_assignments"] = {}
    st.session_state["user_event_map"] = None
    st.session_state["user_code_map"] = None
    st.session_state["annotation_mapping_complete"] = False
    st.session_state["user_display_names"] = {}
    st.session_state["_prev_class_a"] = None
    st.session_state["_prev_class_b"] = None
    for key in ("class_a_display_name", "class_b_display_name", "rest_display_name",
                "anno_class_a", "anno_class_b", "anno_rest"):
        st.session_state.pop(key, None)