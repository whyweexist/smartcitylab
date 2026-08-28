import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from .config import get_settings
from .features import FEATURE_NAMES, features_to_vector

settings = get_settings()

class ModelBundle:
    def __init__(self, scaler, clf, iso, feature_names, model_version, importances):
        self.scaler = scaler
        self.clf = clf  # MultiOutputClassifier wrapping RF
        self.iso = iso  # IsolationForest
        self.feature_names = feature_names
        self.model_version = model_version
        self.importances = importances

_bundle = None

def load_bundle(path: str = None) -> ModelBundle:
    global _bundle
    if _bundle is not None:
        return _bundle
    p = Path(path or settings.model_path)
    if not p.exists():
        # fallback to search
        alt = Path(__file__).parent.parent / "artifacts" / "model.joblib"
        if alt.exists():
            p = alt
        else:
            raise FileNotFoundError(f"Model artifact not found at {p} . Run training pipeline.")
    data = joblib.load(p)
    _bundle = ModelBundle(
        scaler=data["scaler"],
        clf=data["clf"],
        iso=data["iso"],
        feature_names=data["feature_names"],
        model_version=data.get("model_version", settings.model_version),
        importances=data.get("importances", {}),
    )
    return _bundle

def is_loaded() -> bool:
    try:
        load_bundle()
        return True
    except:
        return False

def predict_issues(feats: dict) -> Tuple[Dict[str, float], np.ndarray]:
    bundle = load_bundle()
    vec = features_to_vector(feats).reshape(1, -1)
    Xs = bundle.scaler.transform(vec)
    # predict_proba for each output
    probs = {}
    # clf is MultiOutputClassifier -> estimators_
    # order matches ISSUE_TYPES in training
    from .config import ISSUE_TYPES
    for idx, name in enumerate(ISSUE_TYPES):
        est = bundle.clf.estimators_[idx]
        # some estimators may not have predict_proba if single class; handle
        if hasattr(est, "predict_proba"):
            proba = est.predict_proba(Xs)[0]
            # classes_ may be [0,1] or single
            if len(est.classes_) == 1:
                p = 1.0 if est.classes_[0] == 1 else 0.0
            elif len(proba) == 2:
                # find index of class 1
                if 1 in est.classes_:
                    i1 = list(est.classes_).index(1)
                    p = float(proba[i1])
                else:
                    p = 0.0
            else:
                p = float(proba[-1])
        else:
            p = float(est.predict(Xs)[0])
        probs[name] = float(np.clip(p, 0, 1))
    return probs, Xs

def anomaly_score(Xs: np.ndarray) -> float:
    bundle = load_bundle()
    # isolation forest: decision_function higher = normal, lower = anomaly. Convert to 0-1 anomaly prob
    try:
        dec = bundle.iso.decision_function(Xs)[0]
        score = bundle.iso.score_samples(Xs)[0]
        # Normalize: anomaly prob = sigmoid(-dec)
        # approximate
        prob = 1 / (1 + np.exp(dec * 3))
        return float(np.clip(prob, 0, 1)), float(dec), float(score)
    except Exception:
        return 0.0, 0.0, 0.0

def anomaly_score_wrapped(feats):
    bundle = load_bundle()
    vec = features_to_vector(feats).reshape(1, -1)
    Xs = bundle.scaler.transform(vec)
    prob, dec, sc = anomaly_score(Xs)
    return prob, dec, sc
