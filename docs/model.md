# Model Documentation

## Problem Formulation
Multi-label issue detection (6 issues) + anomaly + quality score (0-100) → 3-class label.

## Feature Design
20 features covering sharpness (Laplacian, Sobel, Canny, FFT), exposure (mean/median/p5/p95, dark/bright ratios), contrast (std, p95-p5), noise (Immerkaer MAD), color (HSV saturation), texture (entropy), compression (blockiness).

## Data
Synthetic: 60 procedural cleans → 420 samples (per_clean=6, 35% multi-label). Split 294/63/63 by source id (leakage-safe). Manifest `data/manifest.json` with seed 42.

## Model
- Scaler: StandardScaler
- Classifier: MultiOutputClassifier(RandomForest 180 trees, max_depth 14, balanced_subsample)
- Anomaly: IsolationForest (120 trees, contamination 0.12) on clean training features

Hyperparameters chosen for CPU efficiency (<1s training) and generalization; no grid search to keep deterministic.

## Training
`python backend/ml/train.py` — extract features, fit scaler, train clf+iso, save bundle with importances.

## Evaluation (test 63, leakage-safe)
- Macro F1 0.795, Micro 0.803 vs baseline 0.460
- Per-class F1: blur 0.867, under 0.889, over 0.667, noise 0.818, severe 0.833, defect 0.696
- Quality accuracy 0.556 (confusion matrix in data/evaluation)
- Failure: ACCEPTABLE recall 0 due to label overlap; dark/bright natural scenes confused.

## Limitations
Synthetic defects ≠ real industrial; anomaly reports *potential* defect; score not MOS.

## Version
1.0.0, artifacts `backend/artifacts/model.joblib`
