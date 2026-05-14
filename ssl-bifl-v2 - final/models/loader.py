# ==============================================================================
# SSL-BIFL: Model Loader
# Handles state dict key remapping and model initialization
# ==============================================================================

import torch
import segmentation_models_pytorch as smp

DEVICE = torch.device("cpu")


def _remap_state_dict(raw: dict) -> dict:
    """
    Handle models saved from custom wrapper classes.
    Strips 'unet.' prefix and skips 'srm.weight' if present.
    Returns clean state dict compatible with plain smp.Unet.
    """
    has_unet_prefix = any(k.startswith("unet.") for k in raw.keys())
    has_srm         = "srm.weight" in raw.keys()

    print(f"[ModelLoader] Keys sample  : {list(raw.keys())[:3]}")
    print(f"[ModelLoader] unet. prefix : {has_unet_prefix}")
    print(f"[ModelLoader] srm.weight   : {has_srm}")

    if not has_unet_prefix:
        return raw

    remapped = {}
    for key, value in raw.items():
        if key.startswith("unet."):
            remapped[key[len("unet."):]] = value
        elif key == "srm.weight":
            pass   # SRM filter not part of smp.Unet
        else:
            remapped[key] = value

    print(f"[ModelLoader] Remapped {len(raw)} → {len(remapped)} keys")
    return remapped


def load_model(model_path: str) -> torch.nn.Module:
    """
    Build smp.Unet (ResNet-18 encoder) and load weights from .pth file.
    Automatically handles key remapping for legacy wrapper-saved models.
    Returns model in eval mode on CPU.
    """
    print(f"[ModelLoader] Loading: {model_path}")

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    ).to(DEVICE)

    raw_state_dict    = torch.load(model_path, map_location=DEVICE)
    clean_state_dict  = _remap_state_dict(raw_state_dict)

    missing, unexpected = model.load_state_dict(
        clean_state_dict, strict=False
    ).missing_keys, model.load_state_dict(
        clean_state_dict, strict=False
    ).unexpected_keys

    if missing:
        print(f"[ModelLoader] WARNING missing keys  : {missing[:3]}")
    if unexpected:
        print(f"[ModelLoader] WARNING unexpected keys: {unexpected[:3]}")

    model.eval()
    print(f"[ModelLoader] Model ready — {sum(p.numel() for p in model.parameters()):,} parameters")
    return model