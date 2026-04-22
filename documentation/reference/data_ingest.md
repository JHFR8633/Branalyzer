# Data Ingestion Helpers

This page documents the data ingestion utilities in `Branalyzer/src/app/data_ingest.py`.

## Responsibilities

- Load PhysioNet EEGBCI runs with MNE and standardize channel metadata.
- Load user-provided EDF files and filter to EEG-only channels.
- Normalize channel naming quirks across datasets.
- Extract simple annotation counts for UI display.

## Key Functions

`load_eegbci_subject(subject, runs, data_path="./data", preload=True)`
Downloads or loads PhysioNet EEGBCI EDF files, standardizes channel names, concatenates runs, and applies the standard 1005 montage.

Inputs:
- `subject`: EEGBCI subject id.
- `runs`: list of run ids to concatenate.
- `data_path`: local cache path for downloaded EDF files.

Output:
- MNE `Raw` containing concatenated EEG recordings.

`load_user_edf(edf_path, preload=True)`
Loads one or more EDF files into MNE, keeps EEG channels only, normalizes channel names, and applies a montage when possible.

Inputs:
- `edf_path`: string path or list of string paths.
- `preload`: whether to preload samples into memory.

Output:
- MNE `Raw` of EEG-only channels, concatenated if multiple files are provided.

`standardize_channel_names(ch_name)`
Normalizes common EEG channel name formats (strips prefixes/suffixes, normalizes case).

`extract_annotations(raw)`
Returns a dict of annotation description counts from a `Raw` instance.
