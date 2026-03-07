import logging
import mne
from mne.datasets import eegbci
from mne.preprocessing import ICA

# Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Domain constants
MOTOR_CHANNELS = ["C3", "Cz", "C4", "F3", "F4", "P3", "P4"]

# EEGBCI loading constants
IMAGERY_RUNS = [4, 8, 12]
DATA_PATH = "./data"

# EEGBCI annotation names -> internal ids
EVENT_IDS = {"T0": 1, "T1": 2, "T2": 3}

# Semantic labels -> same ids
LABEL_MAP = {"rest": 1, "left_fist": 2, "right_fist": 3}


def load_raw(data_path: str) -> mne.io.Raw:
    """Future implementation for user-provided EDF loading."""
    logging.info("Loading raw imagery run data from user-provided file... [NOT IMPLEMENTED]")
    raise NotImplementedError("User data ingestion is not yet implemented.")


def load_raw_db(subject: int) -> mne.io.Raw:
    """Load and concatenate PhysioNet EEGBCI imagery runs for a subject."""
    logging.info(f"Loading raw imagery run data from PhysioNet EEGMMIDB subject {subject}...")

    raw_files = eegbci.load_data(
        subject,
        IMAGERY_RUNS,
        path=DATA_PATH,
        update_path=True,
    )

    raws_list = []
    for file in raw_files:
        raw = mne.io.read_raw_edf(file, preload=True)
        eegbci.standardize(raw)
        raws_list.append(raw)

    raw = mne.concatenate_raws(raws_list)
    montage_assignment = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage_assignment)

    logging.info(f"Finished loading and concatenating raw data for subject {subject}.")
    return raw


def set_average_reference(raw: mne.io.Raw) -> mne.io.Raw:
    raw.set_eeg_reference("average", projection=False)
    return raw


def select_channels(raw: mne.io.Raw, channels: list[str] = MOTOR_CHANNELS) -> mne.io.Raw:
    raw.pick_channels(channels)
    return raw


def highpass_for_ica(raw: mne.io.Raw, l_freq: float = 1.0) -> mne.io.Raw:
    raw.filter(l_freq=l_freq, h_freq=None)
    return raw


def run_ica(raw: mne.io.Raw, n_components: int = 7, random_state: int = 25) -> mne.io.Raw:
    """Fit ICA on a copy, then apply it to the original raw."""
    copy = raw.copy()
    picks = mne.pick_types(copy.info, eeg=True, eog=True, exclude="bads")

    ica = ICA(n_components=n_components, max_iter="auto", random_state=random_state)
    ica.fit(copy, picks=picks)
    ica.apply(raw)
    return raw


def bandpass_mu_beta(raw: mne.io.Raw, l_freq: float = 8.0, h_freq: float = 30.0) -> mne.io.Raw:
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    return raw


def make_epochs(raw: mne.io.Raw, tmin: float = 0.0, tmax: float = 4.0) -> mne.Epochs:
    events, _ = mne.events_from_annotations(raw, event_id=EVENT_IDS)
    epochs = mne.Epochs(
        raw,
        events,
        event_id=LABEL_MAP,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
    )
    return epochs


def preprocessing(raw: mne.io.Raw) -> mne.Epochs:
    """
    Raw -> cleaned + epoched data.
    No downloading or filesystem assumptions here.
    """
    raw = set_average_reference(raw)
    raw = select_channels(raw)
    raw = highpass_for_ica(raw, l_freq=1.0)
    raw = run_ica(raw, n_components=len(MOTOR_CHANNELS))
    raw = bandpass_mu_beta(raw, l_freq=8.0, h_freq=30.0)
    epochs = make_epochs(raw, tmin=0.0, tmax=4.0)
    return epochs