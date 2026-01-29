import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# ==========================================
# 1. MODEL ARCHITECTURE (Must match training!)
# ==========================================
class ResNetUNet(nn.Module):
    def __init__(self, n_class=1):
        super().__init__()
        self.base_model = models.resnet18(weights=None)
        self.base_layers = list(self.base_model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer2 = self.base_layers[5]
        self.layer3 = self.base_layers[6]
        self.layer4 = self.base_layers[7]
        self.up4 = self.upsample_block(512, 256)
        self.up3 = self.upsample_block(512, 128)
        self.up2 = self.upsample_block(256, 64)
        self.up1 = self.upsample_block(128, 64)
        self.up0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_class, kernel_size=1)
        )
    def upsample_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        up4 = self.up4(layer4)
        up3 = self.up3(torch.cat([up4, layer3], 1))
        up2 = self.up2(torch.cat([up3, layer2], 1))
        up1 = self.up1(torch.cat([up2, layer1], 1))
        out = self.up0(torch.cat([up1, layer0], 1))
        return out

# ==========================================
# 2. APP CONFIGURATION
# ==========================================
st.set_page_config(page_title="Forgery Detector", layout="wide")
st.title("🕵️‍♂️ Image Forgery Detection System")
st.markdown("### Deep Learning based Copy-Move Detection")

# Sidebar
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Sensitivity Threshold", 0.0, 1.0, 0.5, 0.05)
st.sidebar.info("Lower sensitivity (0.1-0.3) detects more but adds noise. Higher (0.7+) is stricter.")

# ==========================================
# 3. LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    device = torch.device('cpu') # Safer for inference
    model = ResNetUNet()
    
    # PATH TO YOUR SAVED MODEL
    # We check local first, then drive
    model_path = 'best_model_5k_v2.pth' 
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"❌ Could not find model file: {model_path}. Please make sure it is uploaded.")
        return None

model = load_model()

# ==========================================
# 4. INFERENCE PIPELINE
# ==========================================
uploaded_file = st.file_uploader("Upload an Image to Analyze", type=["jpg", "png", "jpeg", "tif"])

if uploaded_file is not None and model is not None:
    # A. Display Original
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)

    # B. Preprocess
    img_resized = image.resize((256, 256))
    img_np = np.array(img_resized)
    
    # Normalize
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0) # Batch dimension

    # C. Prediction
    with st.spinner("Analyzing pixels..."):
        with torch.no_grad():
            output = model(img_tensor)
            pred_prob = torch.sigmoid(output).squeeze().numpy()
    
    # D. Post-Processing (Threshold & Clean)
    pred_mask = (pred_prob > confidence_threshold).astype(np.uint8) * 255
    
    # Cleaning (Morphological Operations) to remove dots
    kernel = np.ones((5,5), np.uint8)
    pred_clean = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)
    pred_clean = cv2.morphologyEx(pred_clean, cv2.MORPH_OPEN, kernel)

    # E. Create Heatmap Overlay
    heatmap = cv2.applyColorMap(pred_clean, cv2.COLORMAP_JET)
    img_cv = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_cv, 0.7, heatmap, 0.3, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # F. Display Results
    with col2:
        st.image(pred_clean, caption="Detection Mask", use_column_width=True)
    
    with col3:
        st.image(overlay, caption="Forgery Heatmap", use_column_width=True)

    # G. Verdict
    if np.max(pred_clean) > 0:
        st.error(f"⚠️ FORGERY DETECTED! (Confidence > {confidence_threshold})")
    else:
        st.success("✅ Image appears Authentic.")
