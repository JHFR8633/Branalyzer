import logging
import mne
from mne.datasets import eegbci # Old import from PhysioNet, but kept for reference in the future.
from mne.preprocessing import ICA
from mne_icalabel import label_components

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
    raw.pick(channels)
    return raw


def highlowpass_for_ica(raw: mne.io.Raw, l_freq: float = 1.0, requested_h_freq: float = 100.0) -> mne.io.Raw:
    """
    According to the ICA_Label documentation used in run_ica_auto, the ML model used to automatically classify channels for ICA was trained on data with 1Hz-100Hz.
    Therefore, before using ICA_Label, we need to use a maximum of 100Hz as a high-pass filter.
    However, we have to consider the Nyquist frequency of the data (i.e. half of the sampling frequency).
    If the requested high frequency (100Hz as default) exceeds the Nyquist frequency, we will have to cap the high frequency to that Nyquist frequency instead.
    For example: If the input data is 160Hz (such as our PhysioNet data), Nyquist is 80Hz, so we will lose information between 80-100Hz. 
                 So, we cap it to the 80Hz Nyquist frequency instead (but just on the copy of the data for ICA).
    
    https://github.com/mne-tools/mne-icalabel/blob/main/mne_icalabel/iclabel/label_components.py
    """
    nyquist = raw.info['sfreq'] / 2
    ica_h_freq = min(requested_h_freq, nyquist - 2.0) # 2 Hz buffer against the Nyquist filter is necessary as MNE needs a 'transition band' for filtering.

    if(ica_h_freq < requested_h_freq):
        logging.info(f"Requested h_freq of {requested_h_freq} exceeds Nyquist frequency of {nyquist}. Setting h_freq to {ica_h_freq} for ICA preprocessing.")
    else:
        logging.info(f"Using requested h_freq of {requested_h_freq} for ICA preprocessing.")

    raw.filter(l_freq=l_freq, h_freq=ica_h_freq)
    return raw


def run_ica_auto(raw: mne.io.Raw, n_components: int | None = None, random_state: int = 25) -> mne.io.Raw:
    

    
    if raw.info['sfreq'] > 128:
        logging.info(f"Resampling from {raw.info['sfreq']} Hz to 128 Hz before ICA.")
        raw = raw.copy().resample(128)

    if n_components is None:
        good_channels = mne.pick_types(raw.info, eeg=True, exclude="bads")
        n_components = min(len(good_channels) - 1, 20)
        logging.info(f"Automatically setting n_components to {n_components} (capped at 20).")
    else:
        logging.info(f"Manually set n_components as {n_components}.")

    hipass_filtered = highlowpass_for_ica(raw.copy(), l_freq=1.0, requested_h_freq=100.0)

    # picard is faster than infomax
    ica = ICA(n_components=n_components, method="picard", max_iter="auto", random_state=random_state)
    ica.fit(hipass_filtered)
    
    labels = label_components(hipass_filtered, ica, method="iclabel")

    labeled_components = labels["labels"]

    exclusions = [i for i, label in enumerate(labeled_components) if label in ["muscle artifact", "eye blink", "heart beat", "line noise", "channel noise"]]
    # Labels found via: https://mne.tools/mne-icalabel/0.6/generated/api/mne_icalabel.iclabel.iclabel_label_components.html

    logging.info(f"ICA_Label identified and will exclude: {exclusions}")

    ica.exclude = exclusions
    ica.apply(raw)

    return raw


def bandpass_mu_beta(raw: mne.io.Raw, l_freq: float = 8.0, h_freq: float = 30.0) -> mne.io.Raw:
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    return raw


def make_epochs(
    raw: mne.io.Raw, 
    tmin: float = 0.0, tmax: float = 4.0,
    event_ids: dict[str, int] = EVENT_IDS, 
    label_map: dict[str, int] = LABEL_MAP
    ) -> mne.Epochs:

    # Try block for unknown annotations.
    try:
        events, _ = mne.events_from_annotations(raw, event_id=event_ids)
    except ValueError as e:
        logging.error(f"Error extracting events from annotations: {e}")
        events = None # This is essentially for continuous data.

    epochs = mne.Epochs(
        raw,
        events,
        event_id=label_map,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        on_missing="warn",
    )
    return epochs


def preprocessing(
    raw: mne.io.Raw,
    channels: list[str] = MOTOR_CHANNELS,
    event_ids: dict[str, int] = EVENT_IDS,
    label_map: dict[str, int] = LABEL_MAP,
    ) -> mne.Epochs:
    """
    Raw -> cleaned + epoched data.
    Reminder that we have not implemented resampling yet, as we will determine the need after checking if it's necessary for performance reasons. Note: resampling will have to be done after ICA.
    No downloading or filesystem assumptions here.
    """

    raw = set_average_reference(raw)
    raw = run_ica_auto(raw) # There is a second None = None argument for n_components. We can manually set n_components if we want as 2nd argument integer.
    raw = select_channels(raw, channels)
    raw = bandpass_mu_beta(raw, l_freq=8.0, h_freq=30.0)

    epochs = make_epochs(raw, event_ids=event_ids, label_map=label_map, tmin=0.0, tmax=4.0)
    return epochs

    # Note: a quick test script is (in terminal): python -c "from pipeline import run_pipeline; result = run_pipeline(1); print(result)"