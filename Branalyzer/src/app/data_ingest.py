# data_ingestion.py
import logging
from pathlib import Path
import re
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

def standardize_channel_names(ch_name:str) -> str:
    """
    Attempts to clean channel names for various EEG hardware conventions.
    This includes common prefixes/suffixes and normalized formatting.
    """
    clean_name = re.sub(r'^(EEG\s*)|(-REF|-LE)$', '', ch_name, flags=re.IGNORECASE)
    clean_name = clean_name.strip(' .').upper().replace('Z', 'z')
    return clean_name

def load_user_edf(
        edf_path: str | list[str],
        preload: bool = True,
) -> mne.io.Raw:
    """
    Loads the user-provided EDF file and returns an MNE Raw object.

    Channel names are untouched; the user is responsible for selecting their
    target channels in the Streamlit UI.

    However, this function does handle:
        Ignoring non-EEG channels (e.g., EOG, EMG).
    
    Parameters:
        edf_path: Path to the EDF file.
        preload: Whether to preload the data into memory (default: True).

    Returns:
        An mne.io.raw object containing only EEG channels.
    """
    if isinstance(edf_path, str):
        edf_paths = [edf_path]
    elif isinstance(edf_path, list):
        edf_paths = edf_path
    else:
        raise ValueError("edf_path must be a string or a list of strings.")

    raws = []
    for path in edf_paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"EDF file not found at path: {path}")
        
        if path.suffix.lower() not in (".edf", ".edf+"):
            raise ValueError(f"Invalid file type: {path.suffix}. Expected .edf or .edf+")

        logging.info(f"Loading EDF file from path: {path}")
        raw = mne.io.read_raw_edf(
            path,
            infer_types=True, 
            preload=preload
            )

        # Only keep EEG channels, drop others (e.g., EOG, EMG)
        eeg_picks = mne.pick_types(
            raw.info,
            eeg=True,
            meg=False
        )
        raw_eeg = raw.copy().pick(eeg_picks)

        # Cleaning up channel names by stripping periods and enforcing lowercase where necessary.
        raw_eeg.rename_channels(standardize_channel_names)

        # Attempting to set montage for ICALabel after cleaning. For now, just using standard 1005. Note: Look into other configurations in the future.
        try:
            montage = mne.channels.make_standard_montage("standard_1005")
            # Crucially, match_case=False and match_alias=True will attempt MNE to recognize channels even if the names don't perfectly match the standard montage.
            raw_eeg.set_montage(montage, match_case=False, match_alias=True, on_missing="warn")
        except Exception as e:
            logging.warning(f"Could not set montage for file {path}. ICALabel performance may not work without electrode positions. Error: {e}")

        logging.info(f"Loaded EDF file with {len(raw_eeg.ch_names)} EEG channels and {len(raw_eeg.times)} time points.")
        raws.append(raw_eeg)

    if len(raws) == 1:
        return raws[0]

    combined = mne.concatenate_raws(raws)
    return combined


def extract_annotations(raw: mne.io.Raw) -> dict[str, int]:
    """
    Extract annotations from the mne Raw object and return a dictionary of annotation descriptions and their counts.
    """
    if len(raw.annotations) == 0:
        logging.warning("No annotations found in the Raw object.")
        return {}
    annotations = {}
    descriptions = raw.annotations.description
    for desc in descriptions:
        annotations[desc] = annotations.get(desc, 0) + 1
    return annotations


def build_maps(user_assignment: dict[str, str],) -> tuple[dict[str, int], dict[str, int]]:
    """
    Helper function to convert user event code assignments from Streamlit UI into mappings for first & second classes, and rest (if applicable). NOTE: We are only handling the comparison of TWO classes (and rest, if applicable)

    Parameters:
        user_assignment: A dictionary sourced from the Streamlit UI that maps event descriptions to their found & user-assigned class labels (discovered via extract_annotations).
            Example: {"left_hand": "class_a", "right_hand": "class_b", "rest": "rest"}
    Returns:
        A tuple of two dictionaries (each of these are used somewhere in our pipeline): 
            event_map  — annotation desc -> int code (used as both event_ids AND label_map)
            code_map   — {"Rest": 1, "Class A": 2, "Class B": 3} for extract_X_y
    """
    role_to_code = {"Rest": 1, "Class A": 2, "Class B": 3}
    event_map, code_map = {}, {}

    for annotation_desc, assigned_label in user_assignment.items():
        if assigned_label.lower() == "ignore":
            continue
        code = role_to_code.get(assigned_label)
        if code is None:
            logging.warning(f"Unrecognized assigned label '{assigned_label}' for annotation '{annotation_desc}'. Skipping this annotation.")
            continue
        event_map[annotation_desc] = code
        code_map[f"{assigned_label.lower().replace(' ', '_')}_code"] = code
    
    return event_map, code_map