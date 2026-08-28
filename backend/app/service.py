import time
import json
import uuid
from pathlib import Path
from .preprocessing import safe_decode
from .features import extract_features
from .inference import predict_issues, anomaly_score_wrapped, load_bundle
from .scoring import compute_quality_score, classify_label, severity_from_conf
from .explainability import issue_explanation, build_explanations
from .storage import save_bytes
from .config import get_settings, ISSUE_TYPES

settings = get_settings()

def analyze_image(data: bytes, original_filename: str, mime_type: str):
    t0 = time.time()
    rgb, gray, (w, h) = safe_decode(data)
    feats = extract_features(rgb, gray)

    # ML inference
    probs, Xs = predict_issues(feats)
    anomaly_prob, dec, sc = anomaly_score_wrapped(feats)

    # Merge anomaly into potential_defect probability (ensemble): max of ml and anomaly
    # Keep both but adjust probs["potential_defect"] = weighted
    ml_defect = probs.get("potential_defect", 0)
    combined_defect = float(max(ml_defect, anomaly_prob * 0.9))
    # soft blend
    probs["potential_defect"] = float(0.6 * ml_defect + 0.4 * anomaly_prob if ml_defect>0 else anomaly_prob*0.7)

    score = compute_quality_score(probs, anomaly_prob)
    label = classify_label(score, probs, anomaly_prob)

    # overall confidence: inverse of uncertainty; use max prob or distance from 0.5
    max_prob = max(probs.values()) if probs else 0
    # confidence heuristic: high if score extreme or max_prob high
    overall_conf = float(max(0.55, max(max_prob, abs(score-50)/50 * 0.6 + 0.4 )))
    overall_conf = min(1.0, overall_conf)

    # Build issues list with severity/explanation where conf >= threshold
    issues = []
    bundle = None
    try:
        bundle = load_bundle()
        importances = bundle.importances
    except:
        importances = {}

    thresholds = {
        "blur": 0.40,
        "underexposure": 0.40,
        "overexposure": 0.40,
        "noise": 0.40,
        "severe_degradation": 0.40,
        "potential_defect": 0.45,
    }
    # anomaly also triggers potential_defect even if ml low
    if anomaly_prob > 0.55 and probs["potential_defect"] < 0.45:
        probs["potential_defect"] = anomaly_prob

    for itype in ISSUE_TYPES:
        conf = probs.get(itype, 0)
        thresh = thresholds.get(itype, 0.4)
        # anomaly extra condition
        if itype == "potential_defect":
            if conf < thresh and anomaly_prob < 0.5:
                continue
            conf = max(conf, anomaly_prob*0.85)
        else:
            if conf < thresh:
                continue
        sev = severity_from_conf(conf)
        exp = issue_explanation(itype, feats, conf, anomaly_prob)
        evidence = {}
        # attach relevant evidence stats
        if itype == "blur":
            evidence = {"laplacian_var": round(feats["laplacian_var"],2), "sobel_mean": round(feats["sobel_mean"],2), "edge_density": round(feats["edge_density"],4)}
        elif itype in ("underexposure","overexposure"):
            evidence = {"brightness_mean": round(feats["brightness_mean"],1), "dark_ratio": round(feats["dark_ratio"],3), "bright_ratio": round(feats["bright_ratio"],3)}
        elif itype == "noise":
            evidence = {"noise_est": round(feats["noise_est"],3), "hf_ratio": round(feats["hf_ratio"],3)}
        elif itype == "severe_degradation":
            evidence = {"blockiness": round(feats["blockiness"],3), "contrast_p95_p5": round(feats["contrast_p95_p5"],1), "laplacian_var": round(feats["laplacian_var"],1)}
        elif itype == "potential_defect":
            evidence = {"anomaly_prob": round(anomaly_prob,3), "entropy": round(feats["entropy"],2), "saturation_std": round(feats["saturation_std"],1)}
        issues.append({
            "type": itype,
            "severity": sev,
            "confidence": round(float(conf),3),
            "explanation": exp,
            "evidence": evidence
        })

    # If degraded but no issues flagged, force at least one based on top prob
    if not issues and label != "ACCEPTABLE":
        top = max(probs, key=lambda k: probs[k])
        conf = probs[top]
        issues.append({
            "type": top,
            "severity": severity_from_conf(conf),
            "confidence": round(float(conf),3),
            "explanation": issue_explanation(top, feats, conf, anomaly_prob),
            "evidence": {}
        })

    inference_ms = (time.time() - t0) * 1000

    # persistence: save file
    stored = save_bytes(data, original_filename)

    image_stats = {k: round(float(v), 4) if isinstance(v, float) else float(v) for k, v in feats.items()}
    explanations = build_explanations(probs, feats, anomaly_prob, importances)

    # model version
    try:
        mv = load_bundle().model_version
    except:
        mv = settings.model_version

    result = {
        "stored_filename": stored,
        "width": w,
        "height": h,
        "quality_score": score,
        "quality_label": label,
        "confidence": round(float(overall_conf),3),
        "issues": issues,
        "image_stats": image_stats,
        "explanations": explanations,
        "model_version": mv,
        "inference_ms": round(float(inference_ms),1),
        "probs": probs,
        "anomaly_prob": anomaly_prob,
        "feats": feats,
    }
    return result
