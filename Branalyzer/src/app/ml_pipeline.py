from __future__ import annotations

import time

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict, cross_val_score

from schemas import ModelResult, PipelineResult
from lda import run_csp_lda
from svm import run_csp_svm
from rf import run_csp_rf

import mne


# Note: no longer being used, but kept as reference or future comparison.
def extract_simple_features(epochs):
    """Very simple baseline features: mean over time for each channel."""
    data = epochs.get_data()          # (n_epochs, n_channels, n_times)
    X = data.mean(axis=2)             # -> (n_epochs, n_channels)
    y = epochs.events[:, -1]          # labels
    return X, y


def run_pipeline(
    epochs: mne.Epochs,
    subject: int = 1,
    run_lda: bool = True,
    run_svm: bool = True,
    run_rf: bool = True,
) -> PipelineResult:
    """
    Run selected models on pre-processed epochs.
    Epochs are now passed in (cached externally) so preprocessing
    doesn't rerun when model code changes.
    """
    start = time.time()

    results = []
    if run_lda:
        results.append(run_csp_lda(epochs))
    if run_svm:
        results.append(run_csp_svm(epochs))
    if run_rf:
        results.append(run_csp_rf(epochs))

    elapsed = time.time() - start

    models_run = ", ".join(r.name for r in results) or "None"
    return PipelineResult(
        results=results,
        n_subjects=1,
        n_epochs=len(epochs),
        notes=f"EEGBCI -> preprocessing -> CSP + [{models_run}] (subject {subject}) | Model time: {elapsed:.2f}s",
    )