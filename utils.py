import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import streamlit as st
from model import UNet
import os
from config import MODEL_PATH, IMAGE_SIZE

def preprocess_image(image):
    """Preprocess the input image for the model."""
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to specified size
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Add batch dimension
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def postprocess_mask(prediction):
    """Convert model output to displayable mask."""
    # Convert to numpy array
    mask = prediction.squeeze().cpu().numpy()
    
    # Threshold the predictions
    mask = (mask > 0.5).astype(np.uint8) * 255
    
    return mask

@st.cache_resource
def load_model():
    """Load the model with caching."""
    try:
        model = UNet(in_channels=1, out_channels=1)
        device = torch.device('cpu')
        model = model.to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load model weights from {MODEL_PATH}: {str(e)}")
        st.info("Please ensure you have model weights file available or configure the correct path.")
        return None

@st.cache_data
def cache_preprocess_image(image):
    return preprocess_image(image) 
