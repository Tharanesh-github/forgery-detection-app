# ==============================================================================
# SSL-BIFL: Blind Image Forgery Localization — FastAPI Entry Point
# Student: V. Tharanesh (Tharanesh Vigneswaran)
# Hosted on: Hugging Face Spaces (Docker SDK) | Port: 7860
#
# Architecture:
#   models/loader.py          — model loading and state dict remapping
#   inference/preprocessing.py — image loading, normalization, stress tests
#   inference/postprocessing.py — thresholding, morphology, edge refinement
#   inference/metrics.py       — confidence scoring and 4-level verdict
#   inference/pipeline.py      — full orchestration (multi-scale + freq domain)
# ==============================================================================

import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

from models          import load_model, DEVICE
from inference       import ForensicPipeline
from inference.preprocessing import load_image_from_bytes, apply_stress

# ==============================================================================
# 1. STARTUP — load model and build pipeline
# ==============================================================================
MODEL_PATH = "./Last_Final_Deployment_ResNet18_March_Eighteenth.pth"

print(f"[SSL-BIFL] Device: {DEVICE}")
model    = load_model(MODEL_PATH)
pipeline = ForensicPipeline(model)
print("[SSL-BIFL] Forensic pipeline ready")

# ==============================================================================
# 2. FASTAPI APP
# ==============================================================================
app = FastAPI(
    title       = "SSL-BIFL Forgery Localization API",
    description = (
        "Self-Supervised Blind Image Forgery Localization — FYP V. Tharanesh\n\n"
        "Pipeline: Multi-scale inference → Frequency domain boost → "
        "Ensemble thresholding → Edge-aware refinement → 4-level verdict"
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "https://*.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 3. ENDPOINTS
# ==============================================================================

@app.get("/")
def root():
    return {
        "project":     "SSL-BIFL Forgery Localization",
        "student":     "V. Tharanesh (Tharanesh Vigneswaran)",
        "version":     "2.0.0",
        "status":      "online",
        "device":      str(DEVICE),
        "pipeline":    [
            "1. Multi-scale spatial inference (3 scales)",
            "2. Frequency domain DFT boost",
            "3. Ensemble thresholding (5-threshold majority vote)",
            "4. Edge-aware Canny boundary refinement",
            "5. 4-level forensic verdict (Authentic / Inconclusive / Suspicious / Forged)",
        ],
        "endpoints":   ["/health", "/debug", "/analyze"],
    }


@app.get("/health")
def health():
    return {
        "status":  "online",
        "device":  str(DEVICE),
        "model":   "ResNet-18 + U-Net",
        "version": "2.0.0",
    }


@app.get("/debug")
def debug():
    """
    Diagnostic endpoint.
    Runs model on random noise to verify output range.
    Useful for checking model loaded correctly after deployment.
    """
    import torch, numpy as np
    from inference.preprocessing import MEAN, STD
    noise_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    tensor    = torch.from_numpy(
        (noise_img.astype(np.float32) / 255.0 - MEAN) / STD
    ).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
    import torch
    with torch.no_grad():
        out = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()
    return {
        "pred_min":  float(round(out.min(),  4)),
        "pred_max":  float(round(out.max(),  4)),
        "pred_mean": float(round(out.mean(), 4)),
        "pipeline_stages": 5,
        "status": "operational",
    }


@app.post("/analyze")
async def analyze(
    file:      UploadFile = File(...),
    threshold: float      = Form(0.5),
    condition: str        = Form("clean"),
):
    """
    Main forensic analysis endpoint.

    Accepts:
        file      — image upload (JPG / PNG / TIF)
        threshold — 0.2-0.7 operating point (0.5 = auto ensemble mode)
        condition — clean | noise | jpeg (stress test mode)

    Returns:
        forged_percentage  — % pixels flagged
        best_threshold     — selected threshold value
        is_forged          — boolean verdict
        verdict_level      — AUTHENTIC / INCONCLUSIVE / SUSPICIOUS / FORGED
        verdict_message    — human readable explanation
        confidence         — 0-99 confidence score
        pred_max           — raw model output max (diagnostic)
        pred_mean          — raw model output mean (diagnostic)
        activation_ratio   — fraction of image flagged
        condition_applied  — stress test condition used
        original_b64       — base64 PNG of resized original
        heatmap_b64        — base64 PNG of prediction heatmap
        mask_b64           — base64 PNG of binary forgery mask
        overlay_b64        — base64 PNG of heatmap overlay on image
    """
    contents  = await file.read()
    img_rgb   = load_image_from_bytes(contents)
    condition = condition.lower().strip()

    if condition != "clean":
        img_rgb = apply_stress(img_rgb, condition)

    result = pipeline.run(
        img_rgb   = img_rgb,
        threshold = threshold,
        condition = condition,
    )

    return result