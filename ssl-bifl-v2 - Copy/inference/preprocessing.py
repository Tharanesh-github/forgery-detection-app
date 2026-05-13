# ==============================================================================
# SSL-BIFL: Image Preprocessing
# All preprocessing matches Colab evaluate_robustness() exactly
# ==============================================================================

import cv2
import numpy as np
import torch

from models.loader import DEVICE

# ImageNet normalization — matches training code exactly
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])
INPUT_SIZE = (256, 256)


def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """
    Decode uploaded image bytes to RGB numpy array.
    Uses cv2.imdecode (identical to cv2.imread from disk) to match Colab.
    Returns uint8 RGB image at original resolution.
    """
    nparr   = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_bgr is None:
        raise ValueError("Could not decode image — unsupported format or corrupted file")

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def resize_and_normalize(img_rgb: np.ndarray) -> tuple:
    """
    Resize to 256x256 and normalize with ImageNet stats.
    Matches Colab evaluate_robustness() preprocessing exactly:
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (256, 256))
        tensor = torch.from_numpy(
            (img.astype(np.float32) / 255.0 - mean) / std
        ).permute(2,0,1).float().unsqueeze(0).to(DEVICE)
    Returns (resized_img_uint8, normalized_tensor).
    """
    img_resized = cv2.resize(img_rgb, INPUT_SIZE)
    tensor = torch.from_numpy(
        (img_resized.astype(np.float32) / 255.0 - MEAN) / STD
    ).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
    return img_resized, tensor


def apply_stress(img_rgb: np.ndarray, condition: str) -> np.ndarray:
    """
    Apply degradation stress test. Matches Colab evaluate_robustness().

    Conditions:
        clean — no modification
        noise — Gaussian noise σ=20 (matches Colab Noisy condition)
        jpeg  — JPEG Q50 recompression (matches Colab Social Media condition)
    """
    condition = condition.lower().strip()

    if condition == "noise":
        noise = np.clip(
            np.random.normal(0, 20, img_rgb.shape), -255, 255
        ).astype(np.int16)
        return np.clip(
            img_rgb.astype(np.int16) + noise, 0, 255
        ).astype(np.uint8)

    elif condition == "jpeg":
        _, enc = cv2.imencode(
            ".jpg",
            cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 50],
        )
        return cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB)

    return img_rgb