"""
Run this script directly to debug DDColor output values.
Usage: python debug_colorizer.py
"""
import sys
sys.path.insert(0, r'D:\Irozuke\backend')

import torch
import numpy as np
import cv2
from pathlib import Path
from basicsr.archs.ddcolor_arch import DDColor

MODELS_DIR = Path(r'D:\Irozuke\backend\models')
model_path = MODELS_DIR / 'ddcolor_artistic.pth'

# Load weights
print("Loading weights...")
state   = torch.load(str(model_path), map_location='cpu', weights_only=False)
weights = state.get('params', state)

# Build model
model = DDColor(
    encoder_name        = 'convnext-l',
    decoder_name        = 'MultiScaleColorDecoder',
    input_size          = (512, 512),
    num_output_channels = 2,
    num_queries         = 100,
    do_normalize        = False,
)
missing, unexpected = model.load_state_dict(weights, strict=False)
print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
model.eval()

# Create a simple test image (gray gradient)
test = np.ones((512, 512, 3), dtype=np.float32) * 0.5
tensor = torch.from_numpy(test.transpose(2,0,1)).unsqueeze(0)

print("Running inference...")
with torch.no_grad():
    out = model(tensor)

print(f"Output shape: {out.shape}")
print(f"Output min: {out.min().item():.4f}")
print(f"Output max: {out.max().item():.4f}")
print(f"Output mean: {out.mean().item():.4f}")
print(f"Output std: {out.std().item():.4f}")

# Try different scalings
ab = out.squeeze(0).permute(1,2,0).numpy()
print(f"\nab[:,:,0] range: {ab[:,:,0].min():.3f} to {ab[:,:,0].max():.3f}")
print(f"ab[:,:,1] range: {ab[:,:,1].min():.3f} to {ab[:,:,1].max():.3f}")
print("\nIf values are near 0: model weights are mismatched (garbage output)")
print("If values are in [-1,1]: scale by 128 for Lab")
print("If values are in [-128,127]: use directly")
