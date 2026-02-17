from __future__ import annotations
import time
import numpy as np
from schemas import ModelResult, PipelineResult

def run_pipeline(_: str | None = None) -> PipelineResult:
    # replace with ingestion preprocess features train
    start = time.time()

    results = [
        ModelResult("LDA", 0.71, 0.69, 0.04, inference_time_s=0.01),
        ModelResult("SVM", 0.74, 0.72, 0.03, inference_time_s=0.05),
        ModelResult("Random Forest", 0.70, 0.68, 0.05, inference_time_s=0.02),
    ]

    return PipelineResult(
        results=results,
        n_subjects=None,
        n_epochs=None,
        notes=f"Stub pipeline. Wall time: {time.time() - start:.3f}s",
    )

