from typing import Dict, List

EVIDENCE_TEMPLATES = {
    "blur": "Low sharpness (Laplacian var {laplacian_var:.1f}, Sobel mean {sobel_mean:.1f}, edge density {edge_density:.3f}) suggests blur / insufficient focus.",
    "underexposure": "Low luminance (mean {brightness_mean:.1f}, p5 {brightness_p5:.1f}, dark ratio {dark_ratio:.2f}) indicates underexposure.",
    "overexposure": "High luminance (mean {brightness_mean:.1f}, p95 {brightness_p95:.1f}, bright ratio {bright_ratio:.2f}) indicates overexposure / highlight clipping.",
    "noise": "Elevated noise estimate {noise_est:.2f} with HF ratio {hf_ratio:.3f} suggests visible noise/grain.",
    "severe_degradation": "Multiple degradation signals (blockiness {blockiness:.2f}, contrast range {contrast_p95_p5:.1f}, entropy {entropy:.2f}) indicate severe degradation.",
    "potential_defect": "Anomaly score {anomaly_prob:.2f} with texture irregularity (entropy {entropy:.2f}, saturation std {saturation_std:.1f}) suggests potential visual defect/anomaly.",
}

def build_explanations(probs: Dict[str, float], feats: Dict, anomaly_prob: float, importances: Dict = None) -> List[Dict]:
    explanations = []
    # global feature importance context
    top_features = []
    if importances:
        # importances is dict issue->list
        pass

    for issue, conf in probs.items():
        if conf < 0.35 and issue != "potential_defect":
            continue
        if issue == "potential_defect" and conf < 0.3 and anomaly_prob < 0.35:
            continue
        tmpl = EVIDENCE_TEMPLATES.get(issue, "")
        try:
            text = tmpl.format(anomaly_prob=anomaly_prob, **feats)
        except:
            text = tmpl
        # add confidence context
        text += f" (model confidence {conf:.2f})"
        explanations.append({"issue": issue, "text": text})
    if not explanations:
        explanations.append({"issue": "none", "text": f"No strong defect signal. Quality appears acceptable. Top sharpness {feats.get('laplacian_var',0):.1f}, brightness mean {feats.get('brightness_mean',0):.1f}."})
    return explanations

def issue_explanation(issue_type: str, feats: dict, conf: float, anomaly_prob: float = 0) -> str:
    tmpl = EVIDENCE_TEMPLATES.get(issue_type, "Evidence based on image statistics.")
    try:
        return tmpl.format(anomaly_prob=anomaly_prob, **feats) + f" (confidence {conf:.2f})"
    except:
        return tmpl
