# Workflow Modules

This page documents the workflow helpers in `Branalyzer/src/app/workflows/`. These modules connect the UI to data loading, preprocessing, and modeling, and keep state consistent across reruns.

## `session_state.py`

Owns the default session-state layout and reset helpers.

`initialize_app_state()`
Seeds Streamlit session state with default values.

`subject_has_changed(selected_subject)`
Checks whether the selected PhysioNet subject differs from the currently loaded subject.

`clear_model_state()`
Clears model outputs while leaving earlier stage artifacts intact.

`clear_preprocessing_state()`
Clears preprocessing artifacts and also clears model outputs.

`reset_for_new_subject()`
Resets loaded data and all downstream state before switching subjects.

`mark_subject_warning(selected_subject)`
Updates and returns the subject-change warning flag.

## `signal_workflow.py`

Owns the non-UI workflow logic and caching for load, preprocess, and model stages.

Caching:
- Uses `@st.cache_resource` for expensive data loads and preprocessing steps.
- Cached functions return MNE `Raw` objects, signal variants, or `Epochs` objects.

`load_physionet_raw(subject)`
Loads PhysioNet EEGBCI data and applies average reference.

`load_uploaded_raw(file_specs)`
Loads uploaded EDF files and applies average reference.

`build_signal_variants(subject)`
Builds raw, filtered, and ICA-cleaned signal views for PhysioNet data.

`build_signal_variants_for_upload(file_specs)`
Builds raw, filtered, and ICA-cleaned views for uploaded EDF data.

`get_epochs(subject)`
Builds epochs for a PhysioNet subject using the preprocessing backend.

`get_epochs_for_upload(file_specs)`
Builds epochs for uploaded EDF data using the preprocessing backend.

`run_models(_epochs, subject_label, lda, svm, rf)`
Runs the selected model pipeline with Streamlit caching enabled.

`load_subject_data(subject)`
Packages the raw data and UI metadata for a PhysioNet subject.

Return shape (dict):
- `raw_view`, `available_channels`, `default_channels`
- `loaded_source`, `loaded_data_label`
- `can_preprocess_loaded_data`
- `loaded_file_names`, `loaded_file_specs`, `loaded_file_info`, `loaded_annotations`

`load_uploaded_data(file_specs)`
Packages raw data and UI metadata for uploaded EDF files, including simple file info and annotations.

`run_preprocessing_stage(subject=None, file_specs=None)`
Runs preprocessing for either a PhysioNet subject or uploaded EDF data and returns the artifacts needed by the UI.

Return shape (dict):
- `raw_view`, `filtered_view`, `ica_view`, `epochs`
- `preprocessed_data_label`

`run_model_stage(epochs, subject_label, lda, svm, rf)`
Runs the selected models and packages results for session state.

Return shape (dict):
- `pipeline_out`, `results`
- `modeled_data_label`
- `last_model_flags`
