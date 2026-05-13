# ==============================================================================
# SSL-BIFL: Post-Processing Pipeline
# Thresholding, morphological cleaning, edge refinement
# ==============================================================================

import cv2
import numpy as np


def apply_morphological_cleaning(mask_bin: np.ndarray,
                                  kernel_size: int = 3) -> np.ndarray:
    """
    Remove noise and fill holes in binary mask.
    OPEN first (remove isolated noise pixels),
    CLOSE second (fill holes inside detected region).
    Matches Colab apply_morphological_cleaning() with smaller kernel
    to preserve thin splice boundary detections.
    """
    kernel     = np.ones((kernel_size, kernel_size), np.uint8)
    mask_clean = cv2.morphologyEx(mask_bin,   cv2.MORPH_OPEN,  kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)
    return mask_clean


def adaptive_threshold_scan(pred: np.ndarray) -> tuple:
    """
    Scan 10%-90% of actual pred_max range (not fixed 0.2-0.7).
    The model outputs weak signals (pred_max ~0.4-0.9) so fixed
    thresholds miss the activation entirely.

    Selects threshold maximising:
        proxy = mean_confidence_in_region * 0.7 + activation_ratio * 0.3
    for activations in 3%-50% range (realistic forgery region sizes).
    """
    pred_max = float(pred.max())

    if pred_max < 0.01:
        return 0.01, np.zeros_like(pred, dtype=np.uint8)

    scan_min  = pred_max * 0.10
    scan_max  = pred_max * 0.90
    scan_step = (scan_max - scan_min) / 20

    best_proxy  = -1.0
    best_thresh = scan_min
    best_bin    = None

    t = scan_min
    while t <= scan_max + 1e-6:
        t         = round(t, 5)
        candidate = apply_morphological_cleaning(
            (pred > t).astype(np.uint8), kernel_size=3
        )
        activated = float(candidate.mean())

        if 0.03 <= activated <= 0.50:
            region_preds = pred[candidate > 0]
            if len(region_preds) > 0:
                proxy = float(region_preds.mean()) * 0.7 + activated * 0.3
                if proxy > best_proxy:
                    best_proxy  = proxy
                    best_thresh = t
                    best_bin    = candidate

        t += scan_step

    if best_bin is None:
        best_thresh = pred_max * 0.30
        best_bin    = apply_morphological_cleaning(
            (pred > best_thresh).astype(np.uint8), kernel_size=3
        )

    return best_thresh, best_bin


def ensemble_threshold(pred: np.ndarray) -> tuple:
    """
    Majority-vote ensemble across 5 thresholds.
    Each threshold covers a different fraction of pred_max.
    A pixel is marked forged only if flagged by >= 2 of 5 thresholds.
    More robust than any single threshold alone.
    """
    pred_max = float(pred.max())

    if pred_max < 0.01:
        return 0.01, np.zeros_like(pred, dtype=np.uint8)

    fractions  = [0.20, 0.35, 0.50, 0.65, 0.80]
    thresholds = [pred_max * f for f in fractions]
    votes      = np.zeros_like(pred, dtype=np.int32)

    for t in thresholds:
        candidate = apply_morphological_cleaning(
            (pred > t).astype(np.uint8), kernel_size=3
        )
        votes += candidate

    majority = (votes >= 2).astype(np.uint8)
    majority = apply_morphological_cleaning(majority, kernel_size=3)
    return thresholds[2], majority


def edge_aware_refinement(mask_bin: np.ndarray,
                           img_rgb: np.ndarray) -> np.ndarray:
    """
    Snap mask boundaries to image content edges using Canny detector.
    Splice forgeries always have boundaries aligned with image edges.
    Combines:
        - Original mask pixels
        - Solid interior regions (larger kernel morphology)
        - Image edge zone intersected with mask

    This makes localization sharper and more forensically precise.
    """
    gray         = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges        = cv2.Canny(gray, threshold1=30, threshold2=100)
    edge_kernel  = np.ones((3, 3), np.uint8)
    edge_dilated = cv2.dilate(edges, edge_kernel, iterations=1)
    edge_zone    = (edge_dilated > 0).astype(np.uint8)

    solid_region = apply_morphological_cleaning(mask_bin, kernel_size=5)
    refined      = np.clip(
        mask_bin.astype(np.int32)
        + solid_region.astype(np.int32)
        + edge_zone.astype(np.int32) * mask_bin.astype(np.int32),
        0, 1
    ).astype(np.uint8)

    return refined


def frequency_domain_boost(pred: np.ndarray,
                            img_rgb: np.ndarray) -> np.ndarray:
    """
    High-frequency anomaly detection to boost weak predictions.
    Computes the DFT of the grayscale image and extracts high-frequency
    components. Areas with unusual high-frequency content (splice boundaries)
    are used to boost the prediction map.

    This is complementary to the spatial domain U-Net output.
    """
    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dft     = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)

    # High-pass filter — remove low-frequency centre
    h, w    = gray.shape
    cy, cx  = h // 2, w // 2
    radius  = min(h, w) // 8
    mask_lp = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask_lp, (cx, cy), radius, 1.0, -1)
    mask_hp = 1.0 - mask_lp

    # High-frequency magnitude
    magnitude  = np.abs(dft_shift * mask_hp)
    mag_norm   = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    mag_resize = cv2.resize(mag_norm, (256, 256))

    # Blend: 80% spatial prediction + 20% frequency anomaly
    boosted = np.clip(pred * 0.80 + mag_resize * 0.20, 0, 1)
    return boosted.astype(np.float32)