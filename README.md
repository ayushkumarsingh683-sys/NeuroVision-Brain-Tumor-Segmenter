
# 🧠 Brain MRI Tumor Segmentation & Analysis Dashboard

An end-to-end deep learning web application built with **PyTorch** and **Streamlit** for automated brain tumor segmentation from MRI scans. This repository contains the complete pipeline for data preprocessing, custom U-Net/ViT model training, and an interactive web-based dashboard for real-time inference.

---

## 👨‍💻 Author
* **Ayush Kumar Singh**  
* **Roll Number:** 2402901520088  
* **Institution:** ABES Institute of Technology, Ghaziabad  

---

## 🚀 Key Features
* **Interactive Streamlit Dashboard:** Upload patient MRI scans and get real-time visual tumor segmentation masks.
* **Deep Learning Architectures:** Modular implementation using custom convolutional networks and transformer-based segmenters (`unet_baseline.py`, `vit_segmenter.py`).
* **Robust Data Pipeline:** Optimized dataset handling using PIL and PyTorch `Dataset` & `DataLoader` classes to seamlessly process medical image formats.
* **Lightweight & Fast Training:** Designed to run efficiently even on small data subsets for rapid prototyping and local execution.

---

## 🏛️ System Architecture & Workflow

```text
[ Raw MRI Scans (.tif/.png) ] 
       │
       ▼
[ dataset.py (PIL & PyTorch Loader) ] 
       │
       ▼
[ Model Training (train.py) ] ──> [ models/unet_baseline.py / vit_segmenter.py ]
       │                                     │
       ▼                                     ▼
[ Saved Weights (.pth) ] ──────────────> [ Streamlit Web App (app.py) ]
                                             │
                                             ▼
                               [ Real-time Tumor Segmentation ]
Component Breakdown:Data Pipeline (dataset.py): Handles image and binary mask loading, normalization, channel rearrangement (HWC to CHW), and tensor conversion using PyTorch.Model Core (models/):unet_baseline.py: Encoder-decoder architecture with skip connections for precise localization.   vit_segmenter.py: Vision Transformer integration for advanced spatial feature extraction[cite: 2].Training Engine (train.py): Trains the model using GPU/CPU backends and saves trained checkpoints.Web Interface (app.py): Provides a clean UI for patient metadata entry and visual comparison between original scans and predicted segmentation masks[cite: 2].📂 Project Directory StructurePlaintextMedical_Segmentation/
│
├── dataset/                # Local image & mask samples
│   ├── images/
│   └── masks/
│
├── models/                 # Model architectures & weights
│   ├── saved_weights/      # Trained model checkpoints (.pth)[cite: 2]
│   ├── unet_baseline.py    # Baseline U-Net architecture[cite: 2]
│   └── vit_segmenter.py    # ViT-based segmenter architecture[cite: 2]
│
├── app.py                  # Streamlit web dashboard
├── dataset.py              # PyTorch Dataset loader class
├── train.py                # Model training script
└── requirements.txt        # Project dependencies
🛠️ Installation & Setup1. Clone the RepositoryBashgit clone [https://github.com/ayushkumarsingh683-sys/Brain-MRI-Segmentation-App.git](https://github.com/ayushkumarsingh683-sys/Brain-MRI-Segmentation-App.git)
cd Brain-MRI-Segmentation-App
2. Install DependenciesEnsure you have Python installed, then install the required packages:Bashpip install torch torchvision streamlit pillow numpy opencv-python
3. Train the Model (Optional / Quick Start)To train the model on local samples (or a subset of data):Bashpython train.py
4. Run the Streamlit DashboardLaunch the web interface locally:Bashstreamlit run app.py
💡 Tech StackDeep Learning: PyTorch, TorchvisionFrontend & UI: StreamlitImage Processing: PIL (Pillow), NumPy, OpenCV
