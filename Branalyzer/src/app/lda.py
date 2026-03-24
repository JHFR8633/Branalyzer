from __future__ import annotations

import time

import numpy as np
import mne

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline

from schemas import ModelResult

from csp_extraction import _extract_X_y

def run_csp_lda(
    epochs: mne.Epochs,
    *,
    name: str = "CSP+LDA",
    crop_tmin: float = 1.0,
    crop_tmax: float = 2.0,
    drop_rest: bool = True,
    # These codes should match epoching() event_id
    rest_code: int = 1,
    left_code: int = 2,
    right_code: int = 3,
    n_components: int = 4,
    n_splits: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ModelResult:
    """
    Train + evaluate CSP + LDA on epochs.

    Returns ModelResult with:
    - accuracy: mean CV accuracy
    - std_dev: std of CV accuracy
    - f1_score: macro F1 from CV predictions
    - inference_time_s: average predict() time per sample (fit on all data once)
    - confusion_matrix + predictions: from CV predictions
    - meta: configuration + class counts + trained pipeline (optional)
    """

    X, y, meta = _extract_X_y(
        epochs,
        crop_tmin=crop_tmin,
        crop_tmax=crop_tmax,
        drop_rest=drop_rest,
        rest_code=rest_code,
        left_code=left_code,
        right_code=right_code,
    )

    # Guard rails: CSP needs at least 2 classes
    if len(np.unique(y)) < 2:
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

    # Model pipeline
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    clf = Pipeline([("csp", csp), ("lda", lda)])

    # Stratified CV
    cv = StratifiedKFold(
        n_splits=5,
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

    # Store extras (you can remove trained_pipeline if you don't want heavy objects in meta)
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