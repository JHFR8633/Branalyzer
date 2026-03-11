from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any, Dict, List
import numpy as np

@dataclass
class ModelResult:
    name: str
    accuracy: float
    f1_score: float
    std_dev: float
    inference_time_s: float
    confusion_matrix: Optional[np.ndarray] = None
    predictions: Optional[np.ndarray] = None
    ground_truth: Optional[np.ndarray] = None
    meta: Optional[Dict[str, Any]] = None

@dataclass
class PipelineResult:
    results: List[ModelResult]
    n_subjects: Optional[int] = None
    n_epochs: Optional[int] = None
    notes: Optional[str] = None