import streamlit as st
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from model import UNet
from utils import preprocess_image, postprocess_mask
import os
from google.cloud import storage  # if using GCP
import hmac

# Page config
st.set_page_config(
    page_title="Medical Image Tumor Segmentation",
    page_icon="🏥",
    layout="wide"
)

def check_password():
    def password_entered():
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    return st.session_state["password_correct"]

def main():
    if not check_password():
        st.stop()
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page", 
        ["Home", "Segmentation", "About"]
    )

    if page == "Home":
        show_home()
    elif page == "Segmentation":
        show_segmentation()
    else:
        show_about()

def show_home():
    st.title("Medical Image Tumor Segmentation")
    st.write("""
    Welcome to the Medical Image Tumor Segmentation tool. This application helps medical
    professionals identify and segment tumors in medical images using deep learning.
    """)
    
    st.subheader("Key Features")
    st.write("""
    - Upload medical images (MRI, CT scans)
    - Automatic tumor segmentation
    - Visualization of results
    - Download segmentation masks
    """)

def show_segmentation():
    st.title("Tumor Segmentation")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a medical image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        # Process image
        if st.button("Perform Segmentation"):
            # Load model
            model = load_model()
            
            # Preprocess
            processed_image = preprocess_image(image)
            
            # Inference
            with torch.no_grad():
                model.eval()
                prediction = model(processed_image)
                mask = postprocess_mask(prediction)
            
            # Display result
            with col2:
                st.subheader("Segmentation Result")
                plt.figure(figsize=(10, 10))
                plt.imshow(image, cmap='gray')
                plt.imshow(mask, alpha=0.3, cmap='red')
                plt.axis('off')
                st.pyplot(plt)
                
                # Download button
                st.download_button(
                    label="Download Segmentation Mask",
                    data=cv2.imencode('.png', mask)[1].tobytes(),
                    file_name="segmentation_mask.png",
                    mime="image/png"
                )

def show_about():
    st.title("About")
    st.write("""
    This application uses deep learning to perform medical image segmentation for tumor detection.
    It implements a U-Net architecture trained on medical imaging datasets.
    
    ### How it works:
    1. Upload your medical image
    2. The image is preprocessed and normalized
    3. A deep learning model segments the tumor regions
    4. Results are displayed and can be downloaded
    
    ### Technologies Used:
    - Streamlit
    - PyTorch
    - OpenCV
    - Python
    """)

def load_model():
    model = UNet(in_channels=1, out_channels=1)
    
    # Option 1: Load from environment variable path
    weights_path = os.getenv('MODEL_WEIGHTS_PATH', 'model_weights.pth')
    
    # Option 2: Load from cloud storage
    if os.getenv('USE_CLOUD_STORAGE'):
        storage_client = storage.Client()
        bucket = storage_client.bucket(os.getenv('BUCKET_NAME'))
        blob = bucket.blob('model_weights.pth')
        blob.download_to_filename('/tmp/model_weights.pth')
        weights_path = '/tmp/model_weights.pth'
    
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    return model

if __name__ == "__main__":
    main() 
