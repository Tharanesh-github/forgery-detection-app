# forgery-detection-app
A Self-Supervised Computer Vision system that detects "Copy-Move" forgeries in images. Built with PyTorch (ResNet+UNet) and deployed via Streamlit. Capable of localizing manipulated regions without seeing the original image.
# 🕵️‍♂️ Blind Image Forgery Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Overview
This project is a **Self-Supervised Deep Learning System** designed to detect and localize **Copy-Move Forgeries** in digital images. Unlike traditional methods that require the original image for comparison, this model works in a **"Blind"** setting—it detects anomalies based solely on the manipulated image itself.

The system was trained on **synthetic polygon-based forgeries** generated from the COCO dataset and tested on the real-world **CASIA 2.0** forgery dataset.

## 🚀 Demo
**Try the Live App here:** [LINK_TO_YOUR_STREAMLIT_APP](https://share.streamlit.io/)
*(Replace this with your actual Streamlit Cloud URL)*

## ✨ Key Features
* **Self-Supervised Learning:** Trained purely on automatically generated synthetic data (no human labeling required).
* **Advanced Data Augmentation:** Uses random **Polygon Generation** & **Edge Blending** to simulate realistic Photoshop manipulations during training.
* **Architecture:** Custom **ResNet-18 Encoder** + **U-Net Decoder** for precise pixel-level localization.
* **Visual Interpretation:** Outputs a binary mask and a **Red Heatmap** overlay to clearly show forged regions.
* **Robustness:** Capable of detecting forgeries with irregular shapes, not just rectangular crops.

## 🛠️ Tech Stack
* **Core Framework:** PyTorch
* **Computer Vision:** OpenCV, PIL, Albumentations
* **Frontend/UI:** Streamlit
* **Data Processing:** NumPy, Pandas

## 🧠 Model Architecture
The model treats forgery detection as a **Binary Segmentation Task**:
1.  **Input:** A suspected RGB Image.
2.  **Encoder (ResNet-18):** Extracts high-level features (texture inconsistencies, noise artifacts).
3.  **Decoder (Upsampling):** Reconstructs a segmentation mask identifying the forged pixels.
4.  **Output:** A probability map (0 = Authentic, 1 = Forged).



## 📊 Performance
* **Training Data:** COCO 2017 Validation Set (5k images) with dynamic synthesis.
* **Testing Data:** CASIA 2.0 (Tp - Tampered dataset).
* **Metric:** F1-Score (Pixel-level).
* **Result:** The model demonstrates the capability to localize forgeries in cross-domain scenarios without ever seeing real CASIA samples during training.

## 💻 How to Run Locally

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/forgery-detection-app.git](https://github.com/yourusername/forgery-detection-app.git)
    cd forgery-detection-app
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure

├── app.py # Main Streamlit Application ├── best_model_5k_v2.pth # Trained Model Weights ├── requirements.txt # Python Dependencies ├── README.md # Project Documentation └── .gitignore # Files to ignore (e.g., pycache)

## 🤝 Acknowledgments
* **COCO Dataset** for providing the source images for training.
* **CASIA Dataset** for the evaluation benchmark.
