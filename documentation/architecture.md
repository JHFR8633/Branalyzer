# System Architecture

Branalyzer is a Streamlit application organized around a simple three-stage workflow: load data, preprocess signals, and run models. The UI delegates each stage to workflow helpers that manage session state and caching.

## Major Components

- **Streamlit UI**: Renders the app, controls the workflow stages, and visualizes results.
- **Workflow Layer**: Coordinates loading, preprocessing, and model execution while caching intermediate artifacts.
- **Preprocessing Module**: Handles average reference, ICA-based artifact removal, filtering, channel selection, and epoching.
- **Modeling Pipeline**: Runs CSP feature extraction with classic ML classifiers and reports benchmark metrics.

## Dashboard Features

- Load-data dialog for selecting PhysioNet subjects or EDF uploads.
- Stage-based workflow controls with progress indicators.
- Multi-tab waveform inspection (raw, filtered, ICA-cleaned).
- Model benchmarking table with confusion matrices.
- Event log that compares predictions against ground truth per epoch.

## Data Sources

- **PhysioNet EEGBCI**: Built-in subject selection and download through MNE.
- **User Uploads (EDF)**: Local file uploads for custom EEG recordings.
