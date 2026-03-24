from __future__ import annotations

from typing import Optional, Dict, Any, Tuple

import numpy as np
import mne

def _extract_X_y(
    epochs: mne.Epochs,
    *,
    crop_tmin: float,
    crop_tmax: float,
    drop_rest: bool,
    rest_code: int,
    left_code: int,
    right_code: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Convert MNE Epochs -> (X, y) for CSP.
    Returns meta about class counts and preprocessing choices.
    """
    ep = epochs.copy().crop(tmin=crop_tmin, tmax=crop_tmax)

    X = ep.get_data(copy=False)          # (n_epochs, n_ch, n_times)
    y = ep.events[:, -1].copy()          # event codes

    if drop_rest:
        mask = y != rest_code
        X = X[mask]
        y = y[mask]

    # Binary mapping: left vs right -> 0/1
    # If your event codes match EEGBCI style: left=2 right=3 (or your LABEL_MAP)
    uniq = set(np.unique(y).tolist())
    if uniq == {left_code, right_code}:
        y = (y == right_code).astype(int)

    class_counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    meta = {
        "crop_tmin": crop_tmin,
        "crop_tmax": crop_tmax,
        "drop_rest": drop_rest,
        "rest_code": rest_code,
        "left_code": left_code,
        "right_code": right_code,
        "n_samples": int(len(X)),
        "class_counts": class_counts,
        "classes_after_mapping": sorted(list(set(np.unique(y).tolist()))),
    }
    return X, y, meta