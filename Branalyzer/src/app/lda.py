from __future__ import annotations

import mne

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from schemas import ModelResult
from csp_pipeline import extract_X_y, pipeline_helper

def run_csp_lda(
    epochs: mne.Epochs,
    *,
    name: str = "LDA",
    crop_tmin: float = 1.0,
    crop_tmax: float = 2.0,
    drop_rest: bool = True,
    # These codes should match epoching() event_id
    rest_code: int = 1,
    class_a_code: int = 2,
    class_b_code: int = 3,
    n_components: int = 4,
    n_splits: int = 10,
    random_state: int = 42,
) -> ModelResult:
    """
    Train + evaluate CSP + LDA on epochs.
    """
    X, y, meta = extract_X_y(
        epochs,
        crop_tmin=crop_tmin,
        crop_tmax=crop_tmax,
        drop_rest=drop_rest,
        rest_code=rest_code,
        class_a_code=class_a_code,
        class_b_code=class_b_code,
    )
    
    # reg="ledoit_wolf" handles instances where there are less than 7 channels remaining after ICLabel exclusion.
    # https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ledoit_wolf.html
    csp = CSP(n_components=n_components, reg="ledoit_wolf", log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    clf = Pipeline([("csp", csp), ("lda", lda)])

    meta["n_components"] = n_components

    return pipeline_helper(
        clf, X, y,
        name = name,
        meta = meta,
        n_splits = n_splits,
        random_state = random_state,
    )