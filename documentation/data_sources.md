# Dataset Sources

## PhysioNet EEGBCI

The primary built-in dataset is PhysioNet EEGBCI (EEG Motor Movement/Imagery). The app loads subject runs through MNE and standardizes channel metadata for consistent processing.

Typical usage:
- Select a subject id (1-109).
- Runs 4, 8, and 12 are used for motor imagery by default.

## User-Provided EDF Files

Users can upload one or more EDF/EDF+ files directly in the app.

Behavior:
- EDF files are loaded with MNE.
- Non-EEG channels are dropped.
- Channel names are normalized and a montage is applied when possible.

Notes:
- Uploaded data can be inspected immediately.
- Preprocessing and modeling follow the same pipeline when supported.
