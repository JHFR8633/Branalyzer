# Branalyzer Documentation

This folder contains Markdown documentation for the current Streamlit application and its supporting modules.

## Current Notes

- The current Streamlit app is staged around three explicit actions:
  `Load Data`, `Run Preprocessing`, and `Run Models`.
- The app supports two load sources:
  `PhysioNet EEGBCI` and `Upload EDF`.
- Raw inspection works for both sources.
- Preprocessing and model execution depend on the preprocessing backend exposing the helpers expected by the workflow layer.
- The checked-in preprocessing backend should be reconciled before treating upload preprocessing support as fully stable.
