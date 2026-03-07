import mne
import numpy as np
from mne.preprocessing import ICA

# Keep “domain constants” here (not “loading constants”)
MOTOR_CHANNELS = ["C3", "Cz", "C4", "F3", "F4", "P3", "P4"]

EVENT_IDS = {"T0": 1, "T1": 2, "T2": 3}  # EEGBCI annotation names
LABEL_MAP = {"rest": 1, "left_fist": 2, "right_fist": 3}  # semantic names


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
    """
    Fits ICA on a COPY (after highpass is already applied), then applies to original raw.
    """
    copy = raw.copy()
    picks = mne.pick_types(copy.info, eeg=True, eog=True, exclude="bads")

    ica = ICA(n_components=n_components, max_iter="auto", random_state=random_state)
    ica.fit(copy, picks=picks)

    # NOTE: plotting should be done in UI layer, not here (thread-safe)
    ica.apply(raw)
    return raw


def bandpass_mu_beta(raw: mne.io.Raw, l_freq: float = 8.0, h_freq: float = 30.0) -> mne.io.Raw:
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    return raw


def make_epochs(raw: mne.io.Raw, tmin: float = -1.0, tmax: float = 4.0) -> mne.Epochs:
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
    Single responsibility:
    Raw -> cleaned + epoched data.

    No downloading, no filesystem assumptions.
    """
    raw = set_average_reference(raw)
    raw = select_channels(raw)
    raw = highpass_for_ica(raw, l_freq=1.0)
    raw = run_ica(raw, n_components=len(MOTOR_CHANNELS))
    raw = bandpass_mu_beta(raw, l_freq=8.0, h_freq=30.0)
    epochs = make_epochs(raw, tmin=-1.0, tmax=4.0)
    return epochs