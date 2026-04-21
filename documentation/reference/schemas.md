# Shared Schemas

This page documents the shared dataclasses in `Branalyzer/src/app/schemas.py`.

## `ModelResult`

Represents the output of a single model run.

Fields:
- `name`: model name (e.g., "LDA", "SVM").
- `accuracy`: mean cross-validation accuracy.
- `f1_score`: macro F1 score.
- `std_dev`: standard deviation of accuracy across folds.
- `inference_time_s`: per-sample inference time estimate.
- `confusion_matrix`: optional confusion matrix array.
- `predictions`: optional per-epoch predictions from cross-validation.
- `ground_truth`: optional per-epoch ground truth labels.
- `meta`: optional dict of run metadata (hyperparameters, class counts, etc).

## `PipelineResult`

Represents a full pipeline run over a set of epochs.

Fields:
- `results`: list of `ModelResult` objects.
- `n_subjects`: optional count of subjects used.
- `n_epochs`: optional number of epochs processed.
- `notes`: optional human-readable summary of the run.
