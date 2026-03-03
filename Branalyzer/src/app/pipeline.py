from __future__ import annotations
import time
import numpy as np
from schemas import ModelResult, PipelineResult
import preprocessing as prep
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, cross_val_predict

def extract_simple_features(epochs):
    data = epochs.get_data()              
    X = data.mean(axis=2)                 
    y = epochs.events[:, -1]              
    return X, y

def run_pipeline(subject: int = 1) -> PipelineResult:
    start = time.time()

    # Preprocess + epoch
    epochs = prep.preprocessing(subject)

    X, y = extract_simple_features(epochs)

    # Train LDA with cross-validation
    clf = LinearDiscriminantAnalysis()

    scores = cross_val_score(clf, X, y, cv=5)
    y_pred = cross_val_predict(clf, X, y, cv=5)

    accuracy = float(scores.mean())
    std_dev = float(scores.std())

    lda_result = ModelResult(
        name="LDA",
        accuracy=accuracy,
        f1_score=accuracy,   # simple for now
        std_dev=std_dev,
        inference_time_s=time.time() - start,
        predictions=y_pred,
        confusion_matrix=None,
    )

    return PipelineResult(
        results=[lda_result],
        n_subjects=1,
        n_epochs=len(epochs),
        notes=f"LDA run on subject {subject}",
    )

