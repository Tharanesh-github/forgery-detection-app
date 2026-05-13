# SSL-BIFL: Self-Supervised Learning for Blind Image Forgery Localization

![React](https://img.shields.io/badge/Frontend-React%20%7C%20Vite-61DAFB?logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/Al%20Engine-PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Deployment](https://img.shields.io/badge/Cloud-Hugging%20Face%20%7C%20Vercel-FFD21E?logo=huggingface&logoColor=black)

## 📌 Overview
**SSL-BIFL** is a full-stack, cloud-native Artificial Intelligence framework designed to detect and localize digital image tampering (such as copy-move and splicing forgeries). 

Addressing the "supervised bottleneck" and the "compression paradox" in digital forensics, this framework does not rely on human-labeled datasets. Instead, it utilizes a novel **Self-Supervised Learning (SSL)** paradigm trained purely on authentic images. It is explicitly engineered to survive real-world social media degradation (like WhatsApp JPEG compression) that typically destroys fragile forensic evidence.

## 🚀 Live Demo
**Frontend (Vercel):** [https://ssl-bifl-ui.vercel.app/](https://ssl-bifl-ui.vercel.app/)  
**Backend API (Hugging Face Spaces):** [https://huggingface.co/spaces/Tharanesh-Vigneswaran/ssl-bifl-v2](https://huggingface.co/spaces/Tharanesh-Vigneswaran/ssl-bifl-v2)  

## ✨ Key Technical Features
* **Zero-Shot Generalization:** Capable of detecting unseen, human-crafted forgeries without prior exposure to labeled forgery datasets.
* **Compression-Aware Robustness:** Integrates stochastic JPEG compression (Q50) simulation during training, allowing the AI to detect tampering even after severe social media compression.
* **Hybrid AI Architecture:** Utilizes a **ResNet-18 Encoder** to extract microscopic, high-frequency noise floors, paired with a **U-Net Spatial Decoder** for precise, pixel-level localization.
* **Dynamic Threshold Scanner:** Automatically adjusts detection sensitivity and utilizes OpenCV morphological cleaning to eliminate stray pixel noise (false positives).
* **Zero-Persistence Data Tier:** Prioritizes user privacy by holding images strictly in a volatile RAM buffer (`io.BytesIO`) during inference; data is never written to disk.

## 🛠️ Technology Stack
* **Frontend Tier:** React, Vite (Single Page Application)
* **Application Tier:** FastAPI, Python 3
* **Inference Tier:** PyTorch, OpenCV, NumPy, Scikit-learn
* **Infrastructure:** Vercel (UI Hosting), Hugging Face Spaces (GPU Inference API)

## 📊 Evaluation & Performance
The model was trained on the lossless **DIV2K** dataset via dynamic pseudo-forgery generation and evaluated on unseen real-world datasets:

| Dataset Environment | F1-Score | Recall | Specificity (Accuracy)|
| :--- | :--- | :--- | :--- |
| **Synthetic COCO (Control)** | **0.944** | 0.971 | 0.995 |
| **CASIA v2.0 (Zero-Shot)** | **0.104** | 0.056 | 0.989 |

*Note: While a domain generalization gap exists on real-world data, the model successfully maintains a **98.9% specificity** on CASIA v2.0, achieving a near-zero false-positive rate on authentic images.*

## 💻 System Architecture & Local Setup

This project uses a decoupled frontend-backend architecture. You will need to run two separate terminal instances.

### 1. Application & Inference Tier (`SSL-BIFL-V2`)
This repository contains the PyTorch AI engine and the FastAPI server.

**Repository Structure:**
```text
SSL-BIFL-V2/
├── .git/
├── inference/
│   ├── __init__.py
│   ├── metrics.py
│   ├── pipeline.py
│   ├── postprocessing.py
│   └── preprocessing.py
├── models/
│   ├── __init__.py
│   └── loader.py
├── .gitattributes
├── app.py
├── Dockerfile
├── Last_Final_Deployment_ResNet18_March_Eighteenth.pth
├── README.md
└── requirements.txt
```

**Run Instructions:**
```bash
# Navigate to the backend directory
cd SSL-BIFL-V2

# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI server using Uvicorn
uvicorn app:app --reload --port 8000
```

### 2. Presentation Tier (`SSL-BIFL-UI`)
This repository contains the React/Vite Single Page Application.

**Repository Structure:**
```text
SSL-BIFL-UI/
├── .git/
├── node_modules/
├── public/
├── src/
│   ├── assets/
│   ├── App.css
│   ├── App.jsx
│   └── main.jsx
├── .env
├── .gitignore
├── eslint.config.js
├── index.html
├── package-lock.json
├── package.json
├── README.md
└── vite.config.js
```

**Run Instructions:**
```bash
# Navigate to the frontend directory
cd SSL-BIFL-UI

# Install Node dependencies
npm install

# Start the Vite development server
npm run dev
```

## 🤝 Acknowledgments
* **DIV2K Dataset** (Agustsson & Timofte, 2017) for providing high-resolution authentic images for self-supervised training.
* **CASIA 2.0 Dataset** (Dong et al., 2013) for the real-world zero-shot evaluation benchmark.
