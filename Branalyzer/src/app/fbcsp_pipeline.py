from __future__ import annotations

import time
from typing import Optional, Dict, Any, Tuple

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline

import numpy as np
import mne

from schemas import ModelResult

"""
def filterbankCSP(
          # bandpass filtering
    bands[
        #mu
        (3,8), 
        #theta
        (8,13),
        #beta 
        (13,30),
        #gamma etc 
        (30,60)
        ]  
    # feature selection 

    # perform mne.decoding.csp on each selected feature
    # feed result to rf
)
def run_fb_csp(
    epochs: mne.Epochs,
    *,
    name: str = "Filter Bank CSP and Random Forest",
    crop_tmin: float = 1.0,
    crop_tmax: float = 2.0,
    drop_rest: bool = True,
    # These codes should match epoching() event_id
    rest_code: int = 1,
    left_code: int = 2,
    right_code: int = 3,
    n_components: int = 4,
    n_splits: int = 10,
    n_estimators: int = 100,
    random_state: int = 42,
) -> Model Result """

