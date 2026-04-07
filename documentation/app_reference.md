# Branalyzer App Reference

## Purpose

This document describes the current Streamlit application in `src/app/`, with a focus on:

- the app entrypoint
- the UI rendering modules
- the workflow and session-state modules
- the waveform plotting pipeline

The content is based on the code comments and docstrings currently in the project, so it is meant to stay close to the implementation rather than act as a marketing-style overview.

## Application Structure

The Streamlit dashboard is split into three main layers:

1. `streamlit_app.py`
   The top-level entrypoint that wires the page together.
2. `ui/`
   Rendering helpers for workflow controls, signal inspection, benchmarking, and waveform figures.
3. `workflows/`
   Session-state helpers and stage-oriented workflow functions for loading data, preprocessing, and running models.

## High-Level Flow

The app is organized around three explicit stages:

1. `Load Data`
   Loads EEG data from either PhysioNet EEGBCI or uploaded EDF files.
2. `Run Preprocessing`
   Builds filtered views, ICA-cleaned views, and epochs.
3. `Run Models`
   Runs the selected classification models against the prepared epochs.

The app uses `st.session_state` to keep stage outputs available across reruns so changing plot controls does not reload data or retrain models.

## `streamlit_app.py`

### Role

`streamlit_app.py` is the dashboard entrypoint. It sets page-level Streamlit configuration, defines the stage handlers, initializes app state, and delegates section rendering to the UI modules.

### Main Functions

#### `handle_load_data(request)`

Loads raw data for the chosen source and resets downstream workflow state.

Responsibilities:

- reads the requested source from the modal payload
- loads PhysioNet data or uploaded EDF data through the workflow layer
- stores raw signal views and channel metadata in `st.session_state`
- resets waveform navigation state
- clears preprocessing and model outputs because they belong to a previous load stage

#### `handle_run_preprocessing(subject)`

Runs preprocessing for the currently loaded dataset and stores the derived artifacts.

Responsibilities:

- validates that data has already been loaded
- routes preprocessing through the workflow layer
- stores `raw_view`, `filtered_view`, `ica_view`, and `epochs`
- marks preprocessing as ready
- clears model results because the underlying epochs may have changed
- reruns the app after the stage completes

#### `handle_run_models(subject, run_lda, run_svm, run_rf)`

Runs the selected models against the cached preprocessed epochs.

Responsibilities:

- validates that at least one model is selected
- validates that preprocessing has already been run
- routes model execution through the workflow layer
- stores pipeline output, model results, and the last selected model flags
- reruns the app after the stage completes

#### `main()`

Renders the main dashboard and connects the UI controls to the workflow handlers.

Responsibilities:

- initializes default session state
- opens the load-data modal when requested
- renders channel and model selectors
- renders the workflow controls
- renders signal inspection, benchmarking, and event log sections

## `ui/workflow_controls.py`

### Role

This module renders the top-level workflow controls that drive the app stages.

### Functions

#### `render_load_data_dialog()`

Renders the modal dialog used by the load stage.

Current behavior:

- lets the user choose a data source
- shows a PhysioNet subject input for the PhysioNet source
- shows a file uploader for the upload source
- stores the selected request in session state for later processing by the main app flow

#### `render_stage_controls(...)`

Renders the `Load Data`, `Run Preprocessing`, and `Run Models` buttons together with their status blocks.

Responsibilities:

- computes which stages are currently enabled
- shows the subject-change warning when relevant
- processes the pending load request
- keeps each stage’s status indicator in a stable on-page position

#### `render_stage_status(is_complete, loading_label, complete_label)`

Renders a persistent status placeholder for one stage.

This keeps the completed stage visually aligned with the loading state so the layout does not jump.

## `ui/signal_inspection.py`

### Role

This module renders the signal inspection area of the app. It is intentionally isolated behind a Streamlit fragment so waveform interactions can rerun locally instead of redrawing the entire page.

### Functions

#### `render_signal_inspection(picks)`

Renders the signal inspection section.

Responsibilities:

- checks whether raw data has been loaded
- renders plot controls for downsampling, frame speed, and window size
- displays raw, filtered, and ICA-cleaned tabs when preprocessing is ready
- displays only the raw view when preprocessing has not been run yet

#### `render_waveform_navigation(duration)`

Renders the waveform navigation controls below the plots.

Responsibilities:

- clamps the current window start to the recording length
- keeps step size and jump-to values synchronized with session state
- handles left and right stepping
- handles manual jump-to updates

#### `maybe_autoplay_waveform(duration, frame_speed)`

Advances the waveform window while autoplay is active.

Responsibilities:

- checks whether autoplay is enabled
- computes the next valid window start
- updates session state
- reruns only the fragment

#### `render_waveform_plot(...)`

Renders one waveform plot block for the supplied signal view.

Responsibilities:

- renders the title row and optional subtitle
- builds the Plotly figure with `plot_waveforms_plotly`
- shows a warning when the current channel selection cannot be plotted

#### `render_waveform_title_row(title, plot_key, subtitle=None)`

Renders the waveform title area above each plot.

Responsibilities:

- displays the title and optional subtitle
- shows the shared play or pause button
- toggles waveform autoplay through session state

## `ui/results.py`

### Role

This module renders the model results area after the model stage has completed.

### Functions

#### `render_model_benchmarking()`

Renders the model benchmarking section.

Responsibilities:

- shows top-line metrics for the best performer
- displays a comparison table for all model results
- renders one confusion matrix per model

#### `render_confusion_matrix(result, max_cm_value)`

Renders a single confusion matrix heatmap.

Using a shared maximum value helps keep the color scale comparable across models.

#### `render_event_log()`

Renders the combined event log across all selected models.

Current table shape:

- `Epoch #`
- one prediction column per model
- `Ground Truth`

Each model prediction cell is colored:

- green for a match
- red for a mismatch

#### `build_event_log_comparison_table(results)`

Builds the combined event log table and a lookup used for cell-level styling.

This keeps display values and styling logic separate.

#### `highlight_model_prediction_cells(row, match_lookup)`

Applies per-cell background color to the model prediction columns in the event log.

## `ui/waveform.py`

### Role

This module contains the composition-based waveform plotting pipeline used by the signal inspection section.

The goal of the module is to keep waveform preparation, window slicing, and Plotly trace construction modular enough to support future overlays such as model predictions.

### Core Data Structures

#### `WaveformPlotConfig`

Stores the caller-facing plotting options for one waveform view.

Fields include:

- selected channels
- current window start
- window duration
- downsample factor
- title

#### `PreparedWaveformData`

Stores the normalized waveform data and derived values needed for plotting.

Fields include:

- selected channel names and indices
- downsampled data and times
- effective sample rate
- total recording time
- per-channel display offsets
- global y-axis bounds

#### `WindowSlice`

Represents a single visible time window extracted from the prepared data.

### Overlay Interface

#### `WaveformOverlay`

Defines the protocol for pluggable trace builders that use the shared waveform data pipeline.

This is the main extension seam for future overlays.

#### `WaveformLineOverlay`

Implements the current stacked waveform line view.

Responsibilities:

- builds one Plotly line trace per selected channel
- offsets each channel vertically for readability
- includes raw amplitude and display offset in hover data

### Builder

#### `WaveformFigureBuilder`

Assembles the final Plotly figure from prepared data and registered overlays.

Key methods:

- `build_figure()`
- `build_initial_window()`
- `build_initial_traces(window_slice)`
- `build_layout(initial_window)`

### Helper Functions

#### `_downsample(data, factor)`

Returns every nth sample when downsampling is enabled.

#### `normalize_ds_factor(ds_factor)`

Clamps the downsample factor to a valid positive integer.

#### `compute_channel_offsets(data)`

Builds vertical offsets so multiple channels can be shown without overlap.

#### `compute_y_range(data, offsets)`

Computes a fixed y-axis range for the full prepared waveform view.

#### `prepare_waveform_data(raw, config)`

Selects channels, downsamples data, and derives plot-ready waveform state.

#### `slice_window(prepared, start_time, duration)`

Extracts the visible time window from the prepared waveform data.

#### `plot_waveforms_plotly(...)`

Compatibility wrapper used by the UI layer.

This is the main public entrypoint for waveform plotting in the app.

## `workflows/session_state.py`

### Role

This module defines the app’s default session-state structure and the helper functions that manage stage resets and subject warnings.

### Stored State Categories

The default state includes:

- load modal state
- selected source and subject
- waveform navigation state
- loaded, preprocessed, and modeled stage outputs
- uploaded file metadata
- readiness flags for each stage

### Functions

#### `initialize_app_state()`

Seeds Streamlit session state with the default app values.

#### `subject_has_changed(selected_subject)`

Returns whether the selected PhysioNet subject differs from the currently loaded PhysioNet subject.

This check is intentionally limited to the PhysioNet path.

#### `clear_model_state()`

Clears model outputs while keeping earlier stages intact.

#### `clear_preprocessing_state()`

Clears preprocessing artifacts and also clears model state.

#### `reset_for_new_subject()`

Resets loaded data and all downstream state before switching subjects.

#### `mark_subject_warning(selected_subject)`

Updates and returns the subject-change warning flag.

## `workflows/signal_workflow.py`

### Role

This module contains the non-UI workflow logic for loading signals, preparing signal variants, building epochs, and running the model pipeline.

It is the main seam between Streamlit state and the EEG processing backend.

### Constants

#### `PHYSIONET_SOURCE`

String label for the PhysioNet EEGBCI source.

#### `UPLOAD_SOURCE`

String label for the uploaded EDF source.

### Internal Helpers

#### `_apply_average_reference(raw)`

Returns a copy of the raw signal with average reference applied.

#### `_default_channels(raw_view)`

Returns the initial channel selection used by the Streamlit viewer.

#### `_write_uploaded_files(file_specs)`

Writes uploaded EDF bytes to temp files so MNE can load them from disk.

#### `_ensure_preprocessing_backend()`

Checks whether the preprocessing module exposes the helper functions the current workflow expects.

If not, it raises a clear runtime error instead of failing later in the pipeline.

#### `_preprocessing_backend_available()`

Returns a boolean form of the backend availability check for use in stage gating.

### Cached Load Functions

#### `load_physionet_raw(subject)`

Loads a PhysioNet subject and applies average reference.

#### `load_uploaded_raw(file_specs)`

Loads uploaded EDF files and applies average reference.

### Cached Preprocessing Functions

#### `run_ica_cached(subject)`

Runs ICA for a PhysioNet subject.

#### `get_epochs(subject)`

Builds epochs for a PhysioNet subject.

#### `build_signal_variants(subject)`

Builds raw, filtered, and ICA-cleaned views for PhysioNet data.

#### `build_signal_variants_for_upload(file_specs)`

Builds raw, filtered, and ICA-cleaned views for uploaded EDF data.

#### `get_epochs_for_upload(file_specs)`

Builds epochs for uploaded EDF data.

### Cached Model Function

#### `run_models(_epochs, subject_label, lda, svm, rf)`

Runs the selected model pipeline with Streamlit resource caching.

### Stage Wrappers

#### `load_subject_data(subject)`

Packages the raw data and UI metadata needed after loading a PhysioNet subject.

#### `load_uploaded_data(file_specs)`

Packages the raw data and UI metadata needed after loading uploaded EDF data.

This includes:

- raw signal view
- available and default channels
- file names
- simple file info
- extracted annotations

#### `run_preprocessing_stage(subject=None, file_specs=None)`

Runs preprocessing for either a PhysioNet subject or uploaded EDF data.

It returns the artifacts the app needs for signal inspection and modeling.

#### `run_model_stage(epochs, subject_label, lda, svm, rf)`

Runs the selected models and packages the results for `st.session_state`.

## Current Caveat: Preprocessing Backend Drift

The Streamlit app and workflow layer currently expect the preprocessing backend to expose helpers such as:

- `run_ica_auto`
- `select_channels`
- `bandpass_mu_beta`
- `highlowpass_for_ica`
- `make_epochs`
- `MOTOR_CHANNELS`

If `preprocessing.py` no longer provides those helpers, then:

- raw loading can still work
- preprocessing may fail at runtime
- model execution may fail at runtime

This means the next maintenance step should be reconciling `preprocessing.py` with the workflow layer before treating upload preprocessing support as fully stable.

## Recommended Next Documentation Pass

A good next step would be to add:

- a preprocessing backend reference
- a model-pipeline reference for `ml_pipeline.py`, `lda.py`, `svm.py`, `rf.py`, and `csp_pipeline.py`
- a troubleshooting guide for common Streamlit state and preprocessing issues
