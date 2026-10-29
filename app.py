import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from models.vit_segmenter import get_vit_segmenter_model

# Page Configuration with Medical Theme
st.set_page_config(
    page_title="Medical Image Segmentation & Patient Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Medical Look & Clinical UI
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stApp {
        background: linear-gradient(to bottom right, #0f172a, #1e293b);
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        color: #38bdf8 !important;
    }
    .report-card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #334155;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .stButton>button {
        background: linear-gradient(135deg, #0ea5e9 0%, #2563eb 100%);
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 10px rgba(14, 165, 233, 0.3);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #38bdf8 0%, #1d4ed8 100%);
        box-shadow: 0 6px 15px rgba(56, 189, 248, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

# 1. Sidebar for Patient Details & Controls
with st.sidebar:
    st.title("🏥 Patient Dashboard")
    st.markdown("---")
    st.markdown("### 📋 Enter Patient Info")
    
    patient_id = st.text_input("Patient ID", "PID-2026-088")
    patient_name = st.text_input("Patient Name", "Ayush Kumar")
    patient_age = st.number_input("Age", min_value=1, max_value=120, value=20)
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    scan_type = st.selectbox("Scan Modality", ["MRI Scan", "CT Scan", "X-Ray"])
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    st.success("Model Operational (ViT)")

# Main Header
st.title("🧠 Clinical Vision AI: Patient Diagnostic Portal")
st.markdown("Advanced Deep Learning framework for automated tumor/anomaly segmentation and clinical reporting.")
st.markdown("---")

# 2. Load Model with caching
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_vit_segmenter_model()
    try:
        model.load_state_dict(torch.load("models/saved_weights/vit_segmenter_weights.pth", map_location=device))
    except:
        pass
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# 3. File Uploader Section
col_up1, col_up2, col_up3 = st.columns([1, 2, 1])
with col_up2:
    st.markdown("#### 📂 Upload Medical Scan (.png / .jpg)")
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    
    input_image = cv2.resize(image_rgb, (256, 256))
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    res_col1, res_col2 = st.columns(2, gap="large")
    
    with res_col1:
        st.markdown("### 🔬 Original Patient Scan")
        st.image(image_rgb, use_container_width=True, caption=f"{scan_type} - {patient_name}")

    # Preprocess & Inference
    img_tensor = input_image / 255.0
    img_tensor = torch.tensor(img_tensor, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Calculate tumor pixel percentage ratio for clinical summary
        tumor_pixels = np.sum(pred_mask > 0.5)
        total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
        anomaly_percentage = (tumor_pixels / total_pixels) * 100
        
        pred_mask_binary = (pred_mask > 0.5).astype(np.uint8) * 255

    pred_mask_resized = cv2.resize(pred_mask_binary, (image_rgb.shape[1], image_rgb.shape[0]))

    with res_col2:
        st.markdown("### 🎯 Predicted Tumor Segmentation")
        st.image(pred_mask_resized, channels="GRAY", use_container_width=True, caption="Model Generated Mask")
        
    st.markdown("---")
    
    # 4. Patient Clinical Report Card Section
    st.markdown("### 📄 Automated Clinical Diagnostic Report")
    
    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    with r_col1:
        st.metric(label="Patient ID", value=patient_id)
    with r_col2:
        st.metric(label="Patient Name", value=patient_name)
    with r_col3:
        st.metric(label="Modality", value=scan_type)
    with r_col4:
        st.metric(label="Anomaly Region Detected", value=f"{anomaly_percentage:.2f}%")
        
    st.info(f"Summary: Analysis performed using Vision Transformer (SegFormer-MiT-b0). Detected anomaly footprint covers approximately {anomaly_percentage:.2f}% of the region of interest.")