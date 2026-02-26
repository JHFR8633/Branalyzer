import mne
import numpy as np
from mne.datasets import eegbci
from mne.preprocessing import ICA
import logging

# Constants specific to the PhysioNet EEGBCI dataset.
MOTOR_CHANNELS: list[str] = ["C3", "Cz", "C4", "F3", "F4", "P3", "P4"]
IMAGERY_RUNS: list[int] = [4, 8, 12]
# T0 -> Rest / T1 -> Left Fist / T2 -> Right Fist. These are used later in events_from_annotations(), starting from 1.
EVENT_IDS: dict[str, int] = {"rest": 1, "left_fist": 2, "right_fist": 3}
DATA_PATH: str = "./data" # Path to store the PhysioNet EEGMMIDB data for now, until we implement user data ingestion.

# Constants for preprocessing.
TARGET_SAMPLING_RATE: int = 128
BANDPASS_LOW: float = 8.0
BANDPASS_HIGH: float = 30.0
ICA_HIGHPASS: float = 1.0

# Logging setup.
logging.basicConfig(level=logging.INFO, format = "%(levelname)s: %(message)s")


"""
TODO: Our preprocessing pipeline order has changed a bit and now follows:
1. Load raw data
2. Set average reference across all channels (new step I discovered after snooping around for preprocessing order)
3. Resampling   (TODO: This step may be removed. I am still considering whether standardizing the sampling rate is necessary. It would require:
                1. finding supporting research about this step being implemented / 2. testing whether high sampling rates (i.e. 500Hz) causes issue with performance due to more data.)
4. Highpass filtering (1Hz) for ICA (also new step I discovered; this is common practice before ICA)
5. ICA artifact removal
6. Bandpass filtering (8-30Hz) for Mu and Beta frequency bands
7. Epoching

NOTE: In practice, for ICA: We will apply the 1Hz highpass filtering -> make a copy of the data -> fit ICA on the copy -> apply those ICA component exclusions to the original data -> then bandpass filter it.
        This is because the ica.fit() function can modify the data directly, and we just want the weights generated from it to separate the components in the original data.
"""


def load_raw(data_path: str) -> mne.io.Raw:
    """ FUTURE IMPLEMENTATION: Loads the raw EEG data from the user-provided .edf file."""
    logging.info(f"Loading raw imagery run data from user-provided file... [NOT IMPLEMENTED]")
    raise NotImplementedError("User data ingestion is not yet implemented.")


def load_raw_db(subject: int) -> mne.io.Raw:
    """ Loads the raw EEG data for a given subject from the PhysioNet EEGMMIDB dataset."""
    logging.info(f"Loading raw imagery run data from PhysioNet EEGMMIDB subject {subject}...")
    
    raw_files = eegbci.load_data(subject, IMAGERY_RUNS, path=DATA_PATH)
    # eegbci.load_data() returns a list of file paths, one each to each run (4, 8, 12), so we then have to append them all together into a single continuous 'Raw' object.

    raws_list = []
    for file in raw_files:
        raw = mne.io.read_raw_edf(file, preload=True) # preload=True loads the data into memory, which is necessary for concatenation and later processing.
        eegbci.standardize(raw) # renames channels to standard names (e.g. our C3, C4, Cz, etc.) and sets channel type (EEG in our case)
        raws_list.append(raw)

    raw = mne.concatenate_raws(raws_list) # concatenates the list of Raw objects into a single Raw object.
    # Creates a standardized 10-05 montage (electrode layout). 10-05 is a superset of the international 10-20 system, and will cover denser systems (i.e. >64 channels).
    montage_assignment = mne.channels.make_standard_montage("standard_1005") 
    raw.set_montage(montage_assignment)
    logging.info(f"Finished loading and concatenating raw data for subject {subject}.")

    return raw


def set_average_reference(raw: mne.io.Raw) -> mne.io.Raw:
    """ Applies an average reference across every channel before selection and ICA. This practice is common and was discovered on this thread:
    https://mne.discourse.group/t/order-to-pre-process-data/4761 """
    logging.info("Setting average reference across all channels...")
    raw.set_eeg_reference("average", projection=False) # projection=False applies the reference to the data immediately
    return raw


def select_channels(raw: mne.io.Raw, channels_to_select: list[str] = MOTOR_CHANNELS) -> mne.io.Raw:
    """ Selects only the motor imagery-relevant channels from the raw data for subsequent processing steps. This is common practice to reduce dimensionality and focus on relevant features."""
    logging.info(f"Selecting motor imagery-relevant channels: {channels_to_select}...")
    # TODO: In future implementation, we will need checks for whether the data has the channels, and perhaps a complex selection process if it doesn't match (i.e. user selects their own channels?)
    raw.pick_channels(channels_to_select)
    return raw


def resample(raw: mne.io.Raw, target_rate: int = TARGET_SAMPLING_RATE) -> mne.io.Raw:
    logging.info(f"Resampling data to {target_rate}... [NOT IMPLEMENTED]")
    return raw


def highpass_filter(raw: mne.io.Raw, low_freq: float = ICA_HIGHPASS) -> mne.io.Raw:
    logging.info(f"Applying 1Hz highpass filter for ICA decomposition...")
    raw.filter(l_freq=low_freq, h_freq=None)
    return raw


def ica_artifact_removal(raw: mne.io.Raw) -> mne.io.Raw:
    logging.info(f"Removing muscle artifacts (ocular, muscular) via ICA... [NOT IMPLEMENTED]")

    # NOTE: In practice, for ICA: We will apply the 1Hz highpass filtering -> make a copy of the data -> fit ICA on the copy -> apply those ICA component exclusions to the original data -> then bandpass filter it.
    # This is because the ica.fit() function can modify the data directly, and we just want the weights generated from it to separate the components in the original data.
    return raw


def bandpass_filter(raw: mne.io.Raw, low_freq: float = BANDPASS_LOW, high_freq: float = BANDPASS_HIGH) -> mne.io.Raw:
    logging.info(f"Applying {low_freq}-{high_freq}Hz bandpass filter for Mu and Beta frequency bands...")
    raw.filter(l_freq=low_freq, h_freq=high_freq)
    return raw


def epoching(raw: mne.io.Raw) -> mne.io.Raw: # TODO: This will have to be changed to -> mne.Epochs once implemented.
    logging.info(f"Epoching concatenated, cleaned data into epochs for motor imagery classification model training... [NOT IMPLEMENTED]")
    return raw


def preprocessing(subject: int) -> mne.io.Raw: # TODO: This will have to be changed to -> mne.Epochs once implemented.
    """ Runs our preprocessing pipeline on the raw data for a given subject and returns the preprocessed epochs. 
    ORDER: Load data -> Avg. Reference -> Channel selection -> Resample (maybe) -> Highpass Filter -> ICA (on copy of data, then apply component exclusions to original data) -> Bandpass Filter -> Epoching"""

    logging.info(f"Starting preprocessing pipeline for subject {subject}...")

    # EXAMPLE PIPELINE IMPLEMENTATION
    raw = load_raw_db(subject)
    raw = set_average_reference(raw)
    raw = select_channels(raw, MOTOR_CHANNELS)
    raw = resample(raw)
    raw = highpass_filter(raw)
    raw = ica_artifact_removal(raw)
    raw = bandpass_filter(raw)
    epochs = epoching(raw) # TODO: Until epoching is implemented, this just returns the Raw object.

    logging.info(f"Preprocessing complete for subject {subject}.")
    return epochs


if __name__ == "__main__":
    epochs = preprocessing(subject=1)
    print((raw.get_data(picks = 0)).shape)
    print(epochs.info) # TODO: Again, this is a Raw object until epoching is implemented.