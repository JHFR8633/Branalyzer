# data_ingestion.py
import mne
from mne.datasets import eegbci


def load_eegbci_subject(
    subject: int,
    runs: list[int],
    data_path: str = "./data",
    preload: bool = True,
) -> mne.io.Raw:
    """
    Load and concatenate EEGBCI runs for a subject.

    Responsibilities:
    - Download/load PhysioNet EEGBCI EDF files
    - Read EDF into Raw objects
    - Standardize channel names/types
    - Concatenate runs into a single Raw
    - Set standard montage
    """
    raw_files = eegbci.load_data(subject, runs, path=data_path, update_path=True)

    raws = []
    for f in raw_files:
        r = mne.io.read_raw_edf(f, preload=preload)
        eegbci.standardize(r)
        raws.append(r)

    raw = mne.concatenate_raws(raws)

    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage)

    return raw


def load_user_edf(
    edf_path: str,
    preload: bool = True,
    montage_name: str = "standard_1005",
) -> mne.io.Raw:
    """
    FUTURE: Load user-provided EDF.

    Responsibilities:
    - Read EDF
    - (Optional) standardize/rename channels if needed
    - Set montage if possible
    """
    raw = mne.io.read_raw_edf(edf_path, preload=preload)
    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage)
    return raw