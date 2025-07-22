import cv2
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from models.colorization_net import ColorizationNet

# Load image and preprocess
img_path = "image_colorization/sample_gray.jpg"
image = Image.open(img_path).convert("L")
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
l_tensor = transform(image).unsqueeze(0)  # Shape: 1x1x64x64

# Load model
model = ColorizationNet()
model.load_state_dict(torch.load("models/colorization_model.pth", map_location=torch.device("cpu")))
model.eval()

# Predict ab channels
with torch.no_grad():
    ab_output = model(l_tensor).squeeze(0).cpu().numpy()  # Shape: (2, 64, 64)

# Prepare LAB image for OpenCV
l = l_tensor.squeeze(0).squeeze(0).cpu().numpy() * 255  # (64, 64)
ab = ab_output * 128  # Scale back from [-1, 1] to [-128, 128]

lab = np.zeros((64, 64, 3), dtype=np.float32)
lab[:, :, 0] = l
lab[:, :, 1:] = ab.transpose(1, 2, 0)

# Convert LAB to RGB
rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)

# Show Original (Grayscale) and Colorized
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].imshow(l, cmap='gray')
axs[0].set_title("Input (Grayscale)")
axs[0].axis("off")

axs[1].imshow(rgb)
axs[1].set_title("Output (Colorized)")
axs[1].axis("off")

plt.tight_layout()
plt.show()
