import torch
from model import UNet

# Create model
model = UNet(in_channels=1, out_channels=1)

# Save dummy weights
torch.save(model.state_dict(), 'model_weights.pth') 