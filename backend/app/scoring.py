from .config import get_settings, ACCEPTABLE, DEGRADED, POTENTIALLY_DEFECTIVE

settings = get_settings()

def compute_quality_score(probs: dict, anomaly_prob: float) -> float:
    # Weighted penalty
    penalty = (
        probs.get("blur", 0) * settings.weight_blur +
        probs.get("underexposure", 0) * settings.weight_underexposed +
        probs.get("overexposure", 0) * settings.weight_overexposed +
        probs.get("noise", 0) * settings.weight_noise +
        probs.get("severe_degradation", 0) * settings.weight_severe +
        probs.get("potential_defect", 0) * settings.weight_defect * 0.6 +  # ML defect prob
        anomaly_prob * settings.weight_defect * 0.4
    )
    # max penalty ~ sum weights = 104 -> normalize to 0-100
    max_penalty = settings.weight_blur + settings.weight_underexposed + settings.weight_overexposed + settings.weight_noise + settings.weight_severe + settings.weight_defect
    # penalty up to max_penalty if all probs 1
    score = 100 - (penalty / max_penalty * 100)
    score = max(0, min(100, score))
    return round(float(score), 1)

def classify_label(score: float, probs: dict, anomaly_prob: float) -> str:
    # POTENTIALLY_DEFECTIVE if anomaly high or severe high or defect high
    if anomaly_prob > 0.6 or probs.get("potential_defect", 0) > 0.55 or probs.get("severe_degradation", 0) > 0.65:
        return POTENTIALLY_DEFECTIVE
    if score >= 70 and max(probs.values()) < 0.5 and anomaly_prob < 0.4:
        return ACCEPTABLE
    # intermediate
    if score < 45:
        # if severe signals still defective, else degraded
        if anomaly_prob > 0.45:
            return POTENTIALLY_DEFECTIVE
        return DEGRADED
    return DEGRADED

def severity_from_conf(conf: float) -> str:
    if conf >= 0.75:
        return "high"
    if conf >= 0.45:
        return "medium"
    return "low"
