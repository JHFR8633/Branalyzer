from __future__ import annotations

import time

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict, cross_val_score

from schemas import ModelResult, PipelineResult
from preprocessing import load_raw_db, preprocessing
from lda import run_csp_lda
from svm import run_csp_svm
from rf import run_csp_rf


# Note: no longer being used, but kept as reference or future comparison.
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

    # 4) Run CSP + LDA, SVM, and RF models
    csp_lda_result = run_csp_lda(epochs)
    csp_svm_result = run_csp_svm(epochs)
    csp_rf_result = run_csp_rf(epochs)

    elapsed = time.time() - start

    return PipelineResult(
        results=[csp_lda_result, csp_svm_result, csp_rf_result],
        n_subjects=1,
        n_epochs=len(epochs),
        notes=f"EEGBCI -> preprocessing -> CSP + LDA/SVM/RF (subject {subject}) | Elapsed: {elapsed:.2f}s",
    )