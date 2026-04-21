# Pipeline and Workflow

The app guides users through a staged pipeline so each step builds on the last.

## Workflow Stages

1. **Load Data**
   - Choose a PhysioNet EEGBCI subject or upload EDF files.
   - Raw signals become available for inspection immediately.

2. **Preprocess**
   - Apply average reference.
   - Run ICA with ICLabel to remove common artifacts.
   - Select motor-related channels.
   - Bandpass filter in the mu/beta range.
   - Create labeled epochs for classification.

3. **Run Models**
   - Extract CSP features.
   - Evaluate classic classifiers (LDA, SVM, Random Forest).
   - Compute metrics and build confusion matrices.

4. **Inspect Results**
   - Compare model metrics and inference times.
   - Review per-epoch predictions against ground truth.
   - Visualize raw vs. filtered vs. ICA-cleaned waveforms.

## Model Approach

Branalyzer uses a classic CSP + classifier pipeline for benchmarking:

- **Feature Extraction**: Common Spatial Patterns (CSP) on motor imagery epochs.
- **Classifiers**: LDA, linear SVM, and Random Forest.
- **Evaluation**: Stratified cross-validation with accuracy, macro F1, and confusion matrices.
