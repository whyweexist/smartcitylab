"""
Training pipeline: extract features -> scaler -> MultiOutput RF + IsolationForest -> save
"""
import json, joblib, argparse
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.multioutput import MultiOutputClassifier

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.features import extract_features, FEATURE_NAMES, features_to_vector
from app.preprocessing import safe_decode

ISSUE_TYPES = ["blur","underexposure","overexposure","noise","severe_degradation","potential_defect"]

def load_image_rgb(path: Path):
    # Use safe decode style but via cv2 for speed
    data = path.read_bytes()
    from app.preprocessing import safe_decode as sd
    rgb, gray, _ = sd(data)
    return rgb, gray

def extract_for_manifest(manifest_path, image_dir):
    with open(manifest_path) as f:
        manifest = json.load(f)
    X_list, y_dict, labels = [], {k:[] for k in ISSUE_TYPES}, []
    meta = []
    for entry in manifest:
        ipath = Path(image_dir) / entry["image_id"]
        if not ipath.exists():
            continue
        data = ipath.read_bytes()
        from app.preprocessing import safe_decode
        try:
            rgb, gray, _ = safe_decode(data)
        except:
            continue
        feats = extract_features(rgb, gray)
        vec = features_to_vector(feats)
        X_list.append(vec)
        issues = set(entry["issues"])
        for k in ISSUE_TYPES:
            y_dict[k].append(1 if k in issues else 0)
        labels.append(entry["quality_label"])
        meta.append(entry)
    X = np.stack(X_list)
    y = np.stack([y_dict[k] for k in ISSUE_TYPES], axis=1)  # n x 6
    return X, y, labels, meta, manifest

def train(manifest="data/manifest.json", image_dir="data/generated", out="backend/artifacts/model.joblib", seed=42):
    manifest = Path(manifest)
    image_dir = Path(image_dir)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    X, y, labels, meta, full_manifest = extract_for_manifest(manifest, image_dir)
    # Split based on manifest split field
    with open(manifest) as f:
        man = json.load(f)
    # Align with X order: we iterated in manifest order filtered by existence, need mapping
    # Rebuild split arrays aligning
    splits = []
    for entry in man:
        ipath = image_dir / entry["image_id"]
        if not ipath.exists():
            continue
        # check if that entry was included (safe_decode success)
        splits.append(entry["split"])
    splits = np.array(splits)
    # Ensure X matches splits length
    assert len(X)==len(splits), f"mismatch {len(X)} vs {len(splits)}"

    train_idx = splits=="train"
    val_idx = splits=="val"
    test_idx = splits=="test"

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"Train {X_train.shape[0]} Val {X_val.shape[0]} Test {X_test.shape[0]}  Features {X.shape[1]}")

    scaler = StandardScaler()
    Xs_train = scaler.fit_transform(X_train)
    Xs_val = scaler.transform(X_val)
    Xs_test = scaler.transform(X_test) if X_test.size else Xs_val

    # Classifier
    base = RandomForestClassifier(n_estimators=180, max_depth=14, min_samples_leaf=4, n_jobs=-1, random_state=seed, class_weight="balanced_subsample")
    clf = MultiOutputClassifier(base, n_jobs=-1)
    clf.fit(Xs_train, y_train)

    # Anomaly: isolation forest on clean only (y all zero)
    clean_mask = (y_train.sum(axis=1)==0)
    X_clean = Xs_train[clean_mask] if clean_mask.sum()>10 else Xs_train
    iso = IsolationForest(n_estimators=120, contamination=0.12, random_state=seed)
    iso.fit(X_clean)

    # Feature importances per issue
    importances = {}
    for idx, name in enumerate(ISSUE_TYPES):
        est = clf.estimators_[idx]
        importances[name] = est.feature_importances_.tolist()

    # quick val metrics
    from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
    y_pred_val = clf.predict(Xs_val)
    print("Per-class F1 val:")
    for i, name in enumerate(ISSUE_TYPES):
        f1 = f1_score(y_val[:,i], y_pred_val[:,i], zero_division=0)
        print(f"  {name}: {f1:.3f}")
    print("Macro F1 val:", f1_score(y_val, y_pred_val, average="macro", zero_division=0))
    print("Micro F1 val:", f1_score(y_val, y_pred_val, average="micro", zero_division=0))

    bundle = {
        "scaler": scaler,
        "clf": clf,
        "iso": iso,
        "feature_names": FEATURE_NAMES,
        "model_version": "1.0.0",
        "importances": importances,
        "seed": seed,
        "train_size": int(X_train.shape[0]),
        "val_size": int(X_val.shape[0]),
        "test_size": int(X_test.shape[0]),
    }
    joblib.dump(bundle, out)
    print(f"Saved to {out}")

    # Save metadata
    with open(out.with_suffix(".meta.json"), "w") as f:
        json.dump({"feature_names": FEATURE_NAMES, "issue_types": ISSUE_TYPES, "importances": importances, "seed": seed}, f, indent=2)
    return bundle

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.json")
    ap.add_argument("--image_dir", default="data/generated")
    ap.add_argument("--out", default="backend/artifacts/model.joblib")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train(args.manifest, args.image_dir, args.out, args.seed)
