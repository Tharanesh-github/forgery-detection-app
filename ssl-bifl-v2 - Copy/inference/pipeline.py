# ==============================================================================
# SSL-BIFL: Forensic Inference Pipeline
# Orchestrates multi-scale inference, ensemble thresholding,
# frequency domain analysis, and edge-aware refinement
# ==============================================================================

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

from models.loader import DEVICE
from .preprocessing import resize_and_normalize
from .postprocessing import (
    ensemble_threshold,
    adaptive_threshold_scan,
    edge_aware_refinement,
    frequency_domain_boost,
    apply_morphological_cleaning,
)
from .metrics import compute_metrics, build_verdict

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])


class ForensicPipeline:
    """
    End-to-end forensic image analysis pipeline.

    Stages:
        1. Multi-scale spatial inference  (3 scales averaged)
        2. Frequency domain boosting      (DFT high-pass anomaly)
        3. Ensemble thresholding          (majority vote across 5 thresholds)
        4. Edge-aware mask refinement     (Canny boundary snapping)
        5. Forensic metrics + verdict     (4-level verdict system)
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()

    # ── Private helpers ────────────────────────────────────────────────────

    def _infer(self, img_rgb: np.ndarray) -> np.ndarray:
        """Run single forward pass. Returns sigmoid prediction map."""
        _, tensor = resize_and_normalize(img_rgb)
        with torch.no_grad():
            pred = torch.sigmoid(
                self.model(tensor)
            ).squeeze().cpu().numpy()
        return pred

    def _multi_scale_infer(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Run inference at 3 spatial scales and combine.

        Scale 1 (weight 0.60): full image → 256x256
        Scale 2 (weight 0.25): 80% centre crop → 256x256
        Scale 3 (weight 0.15): 60% centre crop → 256x256

        Averaging reduces single-scale artefacts and improves
        detection of forgeries at different spatial extents.
        """
        h, w = img_rgb.shape[:2]

        # Scale 1 — full image
        pred_full = self._infer(img_rgb)

        # Scale 2 — 80% centre crop
        m2     = 0.80
        cy, cx = h // 2, w // 2
        h2, w2 = int(h * m2), int(w * m2)
        y1, x1 = max(0, cy - h2 // 2), max(0, cx - w2 // 2)
        crop2  = img_rgb[y1:y1 + h2, x1:x1 + w2]
        pred_mid = self._infer(crop2)

        # Scale 3 — 60% centre crop
        m3     = 0.60
        h3, w3 = int(h * m3), int(w * m3)
        y2, x2 = max(0, cy - h3 // 2), max(0, cx - w3 // 2)
        crop3  = img_rgb[y2:y2 + h3, x2:x2 + w3]
        pred_small = self._infer(crop3)

        combined = pred_full * 0.60 + pred_mid * 0.25 + pred_small * 0.15
        return combined.astype(np.float32)

    # ── Public interface ───────────────────────────────────────────────────

    def run(self,
            img_rgb:   np.ndarray,
            threshold: float = 0.5,
            condition: str   = "clean") -> dict:
        """
        Full pipeline execution.

        Args:
            img_rgb:   uint8 RGB image (any size, will be resized internally)
            threshold: 0.2-0.7 slider value from UI (0.5 = auto mode)
            condition: clean | noise | jpeg

        Returns complete result dict ready for JSON response.
        """
        img_256 = cv2.resize(img_rgb, (256, 256))

        # Stage 1: Multi-scale spatial inference
        pred_spatial = self._multi_scale_infer(img_256)

        # Stage 2: Frequency domain boost
        pred_boosted = frequency_domain_boost(pred_spatial, img_256)

        print(f"[Pipeline] spatial max={pred_spatial.max():.4f} "
              f"boosted max={pred_boosted.max():.4f} cond={condition}")

        # Stage 3: Threshold — manual or auto ensemble
        if abs(threshold - 0.5) > 0.01:
            pred_max      = float(pred_boosted.max())
            scaled_thresh = pred_max * (threshold / 0.7)
            binary        = (pred_boosted > scaled_thresh).astype(np.uint8)
            clean_mask    = apply_morphological_cleaning(binary, kernel_size=3)
            best_thresh   = scaled_thresh
        else:
            best_thresh, clean_mask = ensemble_threshold(pred_boosted)

        # Stage 4: Edge-aware refinement
        clean_mask = edge_aware_refinement(clean_mask, img_256)

        # Stage 5: Metrics and verdict
        metrics = compute_metrics(pred_boosted, clean_mask, best_thresh)
        verdict = build_verdict(metrics)

        print(f"[Pipeline] forged={metrics['forged_percentage']:.2f}% "
              f"verdict={verdict['verdict_level']}")

        # Stage 6: Build visual outputs
        return {
            **metrics,
            **verdict,
            "condition_applied": condition,
            "original_b64":      self._rgb_to_b64(img_256),
            "heatmap_b64":       self._pred_to_heatmap_b64(pred_boosted),
            "mask_b64":          self._mask_to_b64(clean_mask),
            "overlay_b64":       self._build_overlay_b64(img_256, pred_boosted),
        }

    # ── Visualisation helpers ──────────────────────────────────────────────

    def _rgb_to_b64(self, img_rgb: np.ndarray) -> str:
        pil_img = Image.fromarray(img_rgb.astype(np.uint8))
        buf     = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def _pred_to_heatmap_b64(self, pred: np.ndarray) -> str:
        """Normalise to actual range so heatmap is visible even for weak signals."""
        pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        fig, ax   = plt.subplots(figsize=(4, 4), dpi=100)
        ax.imshow(pred_norm, cmap="hot", vmin=0, vmax=1)
        ax.axis("off")
        plt.tight_layout(pad=0)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def _mask_to_b64(self, mask_bin: np.ndarray) -> str:
        colored     = cv2.applyColorMap(
            (mask_bin * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        return self._rgb_to_b64(colored_rgb)

    def _build_overlay_b64(self,
                            img_rgb: np.ndarray,
                            pred:    np.ndarray) -> str:
        pred_norm   = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        heatmap_rgb = (plt.cm.jet(pred_norm)[:, :, :3] * 255).astype(np.uint8)
        overlay     = np.clip(
            img_rgb.astype(np.float32) * 0.60
            + heatmap_rgb.astype(np.float32) * 0.40,
            0, 255,
        ).astype(np.uint8)
        return self._rgb_to_b64(overlay)