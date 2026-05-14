# ==============================================================================
# SSL-BIFL: Metrics and Verdict Logic
# ==============================================================================

import numpy as np


def compute_metrics(pred: np.ndarray,
                    clean_mask: np.ndarray,
                    best_thresh: float) -> dict:
    """
    Compute all forensic metrics from prediction map and binary mask.

    Returns:
        forged_percentage — % of pixels flagged as forged
        confidence        — model confidence score 0-99
        pred_max          — raw sigmoid max (diagnostic)
        pred_mean         — raw sigmoid mean (diagnostic)
        best_threshold    — threshold selected by scanner
        activation_ratio  — fraction of image flagged
    """
    forged_pct     = float(round(clean_mask.mean() * 100, 2))
    pred_max       = float(pred.max())
    pred_mean      = float(pred.mean())
    activation     = float(clean_mask.mean())

    # Confidence: combination of pred_max and spatial coherence
    # High pred_max alone is not enough — we also reward coherent regions
    spatial_coherence = float(
        np.std(pred[clean_mask > 0]) if clean_mask.sum() > 0 else 0
    )
    raw_confidence = pred_max * 0.70 + (1.0 - spatial_coherence) * 0.30
    confidence     = int(min(99, round(raw_confidence * 99)))

    return {
        "forged_percentage": forged_pct,
        "confidence":        confidence,
        "pred_max":          round(pred_max,   4),
        "pred_mean":         round(pred_mean,  4),
        "best_threshold":    round(best_thresh, 4),
        "activation_ratio":  round(activation,  4),
    }


def build_verdict(metrics: dict) -> dict:
    """
    Forensic verdict logic based on multiple evidence factors.

    Verdict levels:
        AUTHENTIC     — no meaningful activation
        INCONCLUSIVE  — weak signal, below reliable threshold
        SUSPICIOUS    — moderate activation, possible forgery
        FORGED        — strong activation, confident detection

    Uses both forged_percentage and pred_max as evidence.
    Conservative approach — only flags FORGED when multiple
    evidence factors agree.
    """
    forged_pct = metrics["forged_percentage"]
    pred_max   = metrics["pred_max"]
    confidence = metrics["confidence"]

    if forged_pct < 0.10 and pred_max < 0.30:
        level     = "AUTHENTIC"
        is_forged = False
        message   = "No significant forgery indicators detected"

    elif forged_pct < 0.30 or (forged_pct < 1.0 and pred_max < 0.50):
        level     = "INCONCLUSIVE"
        is_forged = False
        message   = f"Weak signal detected ({forged_pct:.2f}% activation) — insufficient evidence"

    elif forged_pct < 2.0 and pred_max < 0.70:
        level     = "SUSPICIOUS"
        is_forged = True
        message   = f"Suspicious region detected ({forged_pct:.2f}% activation)"

    else:
        level     = "FORGED"
        is_forged = True
        message   = f"Forgery localized — {forged_pct:.2f}% of image flagged"

    return {
        "is_forged":      is_forged,
        "verdict_level":  level,
        "verdict_message": message,
    }