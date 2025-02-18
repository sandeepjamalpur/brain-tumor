import streamlit as st
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from model import UNet
from utils import preprocess_image, postprocess_mask
import os
- from google.cloud import storage  # if using GCP
import hmac

# Page config
st.set_page_config(
    page_title="Medical Image Tumor Segmentation",
    page_icon="🏥",
    layout="wide"
)

def load_model():
    model = UNet(in_channels=1, out_channels=1)
    
    # Option 1: Load from environment variable path
    weights_path = os.getenv('MODEL_WEIGHTS_PATH', 'model_weights.pth')
    
-   # Option 2: Load from cloud storage
-   if os.getenv('USE_CLOUD_STORAGE'):
-       storage_client = storage.Client()
-       bucket = storage_client.bucket(os.getenv('BUCKET_NAME'))
-       blob = bucket.blob('model_weights.pth')
-       blob.download_to_filename('/tmp/model_weights.pth')
-       weights_path = '/tmp/model_weights.pth'
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    except Exception as e:
        st.error(f"Failed to load model weights from {weights_path}: {str(e)}")
        st.info("Please ensure you have model weights file available or configure the correct path.")
        return None
        
    return model
