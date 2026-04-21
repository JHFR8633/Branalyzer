# Streamlit App Entrypoint

This page documents the top-level Streamlit app in `Branalyzer/src/app/streamlit_app.py`. It owns page configuration, app state initialization, stage handlers, and section rendering.

## Key Responsibilities

- Configure the Streamlit page.
- Initialize session state defaults.
- Handle the three workflow stages: load, preprocess, model.
- Coordinate UI sections and pass callbacks into UI modules.

## Inputs and State

- Entry UI state comes from `st.session_state` seeded by `initialize_app_state()`.
- Load requests are passed as a dict with `source` and either `subject` or `files`.
- Model selection is read from a Streamlit multiselect and converted to boolean flags.

## Outputs

- Updates `st.session_state` with raw/preprocessed/model artifacts.
- Triggers `st.rerun()` after long-running stages to refresh UI state.

## Main Functions

`handle_load_data(request)`
Loads PhysioNet EEGBCI data or uploaded EDF data, stores raw signal views and metadata in session state, resets downstream workflow state, and prepares waveform navigation defaults.

Key fields set in session state:
- `loaded_source`, `loaded_subject`, `loaded_data_label`
- `raw_view`, `available_channels`, `default_channels`
- `loaded_file_names`, `loaded_file_specs`, `loaded_file_info`, `loaded_annotations`
- `waveform_window_start`, `waveform_jump_to`, `waveform_jump_to_input`

`handle_run_preprocessing(subject)`
Validates that data is loaded, runs preprocessing through the workflow layer, stores derived artifacts in session state, clears model outputs, and reruns the app.

`handle_run_models(subject, run_lda, run_svm, run_rf)`
Validates preprocessing readiness and model selection, runs the selected models through the workflow layer, stores results in session state, and reruns the app.

`main()`
Renders the dashboard layout, wires up workflow controls, and renders the signal inspection and results sections.
