from __future__ import annotations
 
import time
 
import numpy as np
import mne
 
from mne.decoding import CSP
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
 
from schemas import ModelResult
from csp_extraction import extract_X_y