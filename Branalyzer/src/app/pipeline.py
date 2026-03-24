from __future__ import annotations

import time

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict, cross_val_score

from schemas import ModelResult, PipelineResult
from preprocessing import load_raw_db, preprocessing
#from csp import run_csp_lda
from lda import run_csp_lda


def extract_simple_features(epochs):
    """Very simple baseline features: mean over time for each channel."""
    data = epochs.get_data()          # (n_epochs, n_channels, n_times)
    X = data.mean(axis=2)             # -> (n_epochs, n_channels)
    y = epochs.events[:, -1]          # labels
    return X, y


def run_pipeline(subject: int = 1) -> PipelineResult:
    start = time.time()

    # 1) Load raw EEGBCI data
    raw = load_raw_db(subject)

    # 2) Preprocess into epochs
    epochs = preprocessing(raw)

    # 3) Simple baseline features for plain LDA
    X, y = extract_simple_features(epochs)

    # 4) Train/evaluate baseline LDA
    clf = LinearDiscriminantAnalysis()

    scores = cross_val_score(clf, X, y, cv=5)
    y_pred = cross_val_predict(clf, X, y, cv=5)

    accuracy = float(scores.mean())
    std_dev = float(scores.std())
    f1 = float(f1_score(y, y_pred, average="macro"))
    cm = confusion_matrix(y, y_pred)

    # Per-sample inference time: fit once on all data, time a single predict call
    clf.fit(X, y)
    t0 = time.perf_counter()
    _ = clf.predict(X)
    t1 = time.perf_counter()
    lda_infer_time = float((t1 - t0) / max(len(X), 1))

    lda_result = ModelResult(
        name="LDA",
        accuracy=accuracy,
        f1_score=f1,
        std_dev=std_dev,
        inference_time_s=lda_infer_time,
        predictions=y_pred,
        ground_truth=y,
        confusion_matrix=cm,
    )

    # 5) Run CSP + LDA model
    csp_result = run_csp_lda(epochs)

    return PipelineResult(
        results=[lda_result, csp_result],
        n_subjects=1,
        n_epochs=len(epochs),
        notes=f"EEGBCI -> preprocessing -> LDA + CSP/LDA (subject {subject})",
    )