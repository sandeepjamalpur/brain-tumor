import urllib.request
import os

# Create sample_data directory if it doesn't exist
os.makedirs('sample_data', exist_ok=True)

# Download a sample brain MRI image
url = "https://raw.githubusercontent.com/mateuszbuda/brain-segmentation-pytorch/master/assets/TCGA_CS_4944.png"
urllib.request.urlretrieve(url, 'sample_data/sample_brain_mri.jpg') 