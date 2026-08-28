# Evaluation Summary

Test samples: 126

## Per-issue F1
- blur: P=0.682 R=0.750 F1=0.714
- underexposure: P=0.917 R=0.815 F1=0.863
- overexposure: P=0.643 R=0.643 F1=0.643
- noise: P=0.792 R=0.905 F1=0.844
- severe_degradation: P=0.542 R=0.684 F1=0.605
- potential_defect: P=0.700 R=0.280 F1=0.400

Macro F1: 0.678
Micro F1: 0.691
Baseline Macro F1: 0.510

Quality Accuracy: 0.603
Quality Macro F1: 0.443

## Notes
- Leakage-safe: split by source clean id before synthesis.
- Anomaly detector trained only on clean training samples.
