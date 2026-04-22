# Modeling Pipeline

This page documents the CSP-based modeling helpers in `Branalyzer/src/app/` that benchmark classic machine-learning classifiers.

Key files:
- `csp_pipeline.py`
- `lda.py`
- `svm.py`
- `rf.py`
- `ml_pipeline.py`
- `schemas.py`

## Core Pipeline Helpers

`extract_X_y(epochs, crop_tmin, crop_tmax, drop_rest, rest_code, left_code, right_code)`
Converts MNE `Epochs` into `(X, y)` arrays and returns metadata about class counts and preprocessing choices.

Shapes:
- `X`: `(n_epochs, n_channels, n_times)`
- `y`: `(n_epochs,)`

`pipeline_helper(clf, X, y, name, meta, n_components=4, n_splits=10, test_size=0.2, random_state=42)`
Runs stratified cross-validation, computes accuracy and F1, builds a confusion matrix, and measures inference time.

Behavior:
- Dynamically reduces `n_splits` based on the minimum class count.
- Returns a `ModelResult` with predictions and ground truth.

## Model Runners

`run_csp_lda(epochs, ...)`
Runs CSP + LDA and returns a `ModelResult`. Uses linear discriminant analysis from scikit-learn.

`run_csp_svm(epochs, ...)`
Runs CSP + linear SVM and returns a `ModelResult`. Uses `SVC(kernel="linear")`.

`run_csp_rf(epochs, ...)`
Runs CSP + Random Forest and returns a `ModelResult`. Uses `RandomForestClassifier` with `n_estimators`.

## Pipeline Wrapper

`run_pipeline(epochs, subject=1, run_lda=True, run_svm=True, run_rf=True)`
Runs the selected model set and returns a `PipelineResult` with notes and summary metadata.

## Shared Schemas

See `schemas.py` for `ModelResult` and `PipelineResult` definitions.
