from __future__ import annotations

import time
from typing import Optional, Dict, Any, Tuple

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline

import numpy as np
import mne

from schemas import ModelResult

def extract_X_y(
    epochs: mne.Epochs,
    *,
    crop_tmin: float,
    crop_tmax: float,
    drop_rest: bool,
    rest_code: int,
    class_a_code: int,
    class_b_code: int,
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
    if uniq == {class_a_code, class_b_code}:
        y = (y == class_b_code).astype(int)

    class_counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    meta = {
        "crop_tmin": crop_tmin,
        "crop_tmax": crop_tmax,
        "drop_rest": drop_rest,
        "rest_code": rest_code,
        "class_a_code": class_a_code,
        "class_b_code": class_b_code,
        "n_samples": int(len(X)),
        "class_counts": class_counts,
        "classes_after_mapping": sorted(list(set(np.unique(y).tolist()))),
    }
    return X, y, meta


def pipeline_helper(
    clf: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    *,
    name: str,
    meta: Dict[str, Any],

    n_components: int = 4,
    n_splits: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelResult:
    """
    Helper function to run a CSP + classifier pipeline with CV and timing.
    Avoids code repetition across rf.py, lda.py, svm.py, which got messy.
    """

    # Dynamic n_splits, specifically for handling small EEG runs.
    _, y_counts = np.unique(y, return_counts=True)
    min_count = int(np.min(y_counts))
    dyn_n_splits = min(n_splits, min_count)

    if dyn_n_splits < 2:
        return ModelResult(
            name=name,
            accuracy=0.0,
            f1_score=0.0,
            std_dev=0.0,
            inference_time_s=0.0,
            confusion_matrix=None,
            predictions=None,
            meta={**meta, "error": "Need at least 2 classes after filtering/mapping."},
        )
    
    # Stratified CV
    cv = StratifiedKFold(
        n_splits=dyn_n_splits,
        shuffle=True,
        random_state=random_state,
    )

    # Accuracy distribution
    acc_scores = cross_val_score(clf, X, y, cv=cv, n_jobs=None)
    acc_mean = float(np.mean(acc_scores))
    acc_std = float(np.std(acc_scores))

    # Predictions for F1 + confusion matrix
    y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=None)
    f1 = float(f1_score(y, y_pred, average="macro"))
    cm = confusion_matrix(y, y_pred)

     # Fit once on all data for inference timing
    clf.fit(X, y)
    t0 = time.perf_counter()
    _ = clf.predict(X)
    t1 = time.perf_counter()
    infer_time_per_sample = float((t1 - t0) / max(len(X), 1))

    meta.update(
        {
            "n_components": n_components,
            "n_splits": n_splits,
            "test_size": test_size,
            "random_state": random_state,
            "chance_level": float(max(np.mean(y == 0), np.mean(y == 1)))
            if set(np.unique(y).tolist()) <= {0, 1}
            else None,
            "trained_pipeline": clf,
        }
    )

    return ModelResult(
        name=name,
        accuracy=acc_mean,
        f1_score=f1,
        std_dev=acc_std,
        inference_time_s=infer_time_per_sample,
        confusion_matrix=cm,
        predictions=y_pred,
        ground_truth=y,
        meta=meta,
    )