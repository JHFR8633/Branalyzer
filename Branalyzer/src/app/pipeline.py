# pipeline.py
from __future__ import annotations

from typing import List

import mne

from schemas import PipelineResult, ModelResult
from data_ingestion import load_eegbci_subject
from preprocessing import preprocessing_from_raw
from models.csp_lda import run_csp_lda


def run_pipeline(subject: int = 1) -> PipelineResult:
    """
    Full EEG pipeline:

    1) Load EEGBCI data
    2) Preprocess into epochs
    3) Run CSP+LDA model
    4) Return structured PipelineResult
    """

    raw: mne.io.Raw = load_eegbci_subject(
        subject=subject,
        runs=[4, 8, 12],
        data_path="./data",
        preload=True,
    )


    epochs: mne.Epochs = preprocessing_from_raw(raw)
    results: List[ModelResult] = []

    csp_result = run_csp_lda(epochs)
    results.append(csp_result)

    return PipelineResult(
        results=results,
        n_subjects=1,
        n_epochs=len(epochs),
        notes="EEGBCI -> preprocessing -> CSP+LDA baseline",
    )