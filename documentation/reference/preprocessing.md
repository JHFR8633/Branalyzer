# Preprocessing Helpers

This page documents the EEG preprocessing helpers in `Branalyzer/src/app/preprocessing.py` and their expected behavior as used by the workflow layer.

## Responsibilities

- Apply average reference.
- Run ICA and remove common artifacts using ICLabel.
- Select motor-related channels.
- Apply mu/beta bandpass filtering.
- Create labeled epochs for classification.

## Core Constants

- `MOTOR_CHANNELS`: default motor-related channels (C3, Cz, C4, F3, F4, P3, P4).
- `IMAGERY_RUNS`: PhysioNet run ids for imagery tasks (4, 8, 12).
- `EVENT_IDS`: mapping of PhysioNet annotations to event ids.
- `LABEL_MAP`: semantic labels (`rest`, `left_fist`, `right_fist`) to event ids.

## Key Functions

`load_raw_db(subject)`
Loads and concatenates PhysioNet EEGBCI imagery runs for a subject.

`set_average_reference(raw)`
Applies average reference to the raw signal.

`select_channels(raw, channels=MOTOR_CHANNELS)`
Selects a default motor-related channel subset.

`highlowpass_for_ica(raw, l_freq=1.0, requested_h_freq=100.0)`
Prepares a signal copy for ICA by high/low-pass filtering with Nyquist safety.

`run_ica_auto(raw, n_components=None, random_state=25)`
Runs ICA using Picard and removes components labeled as artifacts by ICLabel.

Artifact classes removed:
- muscle artifact
- eye blink
- heart beat
- line noise
- channel noise

`bandpass_mu_beta(raw, l_freq=8.0, h_freq=30.0)`
Filters the signal in the mu/beta band.

`make_epochs(raw, tmin=0.0, tmax=4.0, event_ids=EVENT_IDS, label_map=LABEL_MAP)`
Builds labeled epochs for downstream classification.

`preprocessing(raw, channels=MOTOR_CHANNELS, event_ids=EVENT_IDS, label_map=LABEL_MAP)`
Runs the full preprocessing chain and returns epochs.

## Technical Notes

- ICA preprocessing applies a 1-100 Hz filter capped at Nyquist to match ICLabel expectations.
- The pipeline resamples to 128 Hz if the source sampling rate is higher.
- Epoching uses `mne.events_from_annotations`; unknown annotations log an error and yield empty events.
