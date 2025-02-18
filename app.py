import streamlit as st
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from src.model import UNet
try:
    from src.utils import preprocess_image, postprocess_mask, load_model
    from src.config import APP_NAME, APP_ICON, DEFAULT_PASSWORD, SAMPLE_IMAGE_PATH
except ImportError as e:
    st.error(f"Error importing required modules: {str(e)}")
    st.info("Please make sure all required files are in the correct location.")
    raise
import os
import hmac

# Page config
st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout="wide"
)

def check_password():
    def password_entered():
        # Try to get password from secrets, fallback to default
        correct_password = st.secrets.get("password", DEFAULT_PASSWORD)
        if hmac.compare_digest(st.session_state["password"], correct_password):
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.write("Please enter the password to access the application.")
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
    st.title(APP_NAME)
    
    # Add a sample image section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Sample Images")
    if st.sidebar.button("Load Sample Image"):
        try:
            image = Image.open(SAMPLE_IMAGE_PATH)
            st.image(image, caption="Sample Brain MRI", use_column_width=True)
        except Exception as e:
            st.error(f"Could not load sample image: {str(e)}")
            st.info("Please ensure you have sample images in the sample_data directory")
    
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
    
    # Add sample image option
    use_sample = st.checkbox("Use sample image")
    
    # File uploader
    if use_sample:
        try:
            image = Image.open(SAMPLE_IMAGE_PATH)
        except Exception as e:
            st.error("Could not load sample image")
            return
    else:
        uploaded_file = st.file_uploader("Choose a medical image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
        else:
            return
    
    # Display original image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    # Process image
    if st.button("Perform Segmentation"):
        with st.spinner("Loading model and performing segmentation..."):
            # Load model
            model = load_model()
            if model is None:
                return
            
            try:
                # Preprocess
                processed_image = preprocess_image(image)
                
                # Inference
                with torch.no_grad():
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
            except Exception as e:
                st.error(f"Error during segmentation: {str(e)}")

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

if __name__ == "__main__":
    main() 
