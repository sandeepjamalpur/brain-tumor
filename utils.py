import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import streamlit as st

def preprocess_image(image):
    """Preprocess the input image for the model."""
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 256x256
    image = image.resize((256, 256))
    
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

@st.cache_resource  # Cache the model to avoid reloading
def get_model():
    return load_model()

@st.cache_data  # Cache preprocessed images
def cache_preprocess_image(image):
    return preprocess_image(image) 