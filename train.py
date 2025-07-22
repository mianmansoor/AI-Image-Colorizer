import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils.preprocessing import load_images_from_tiny_imagenet, convert_to_lab
from tqdm import tqdm
from models.colorization_net import ColorizationNet  # <-- Correct architecture import

# -------------------------------------
# 1. Custom Dataset Class
# -------------------------------------
class ColorizationDataset(Dataset):
    def __init__(self, lab_images):
        self.l_channel = lab_images[:, :, :, 0]
        self.ab_channels = lab_images[:, :, :, 1:]
        self.l_channel = self.l_channel / 255.0
        self.ab_channels = self.ab_channels / 255.0

    def __len__(self):
        return len(self.l_channel)

    def __getitem__(self, idx):
        l = self.l_channel[idx]
        ab = self.ab_channels[idx]
        l = torch.tensor(l).unsqueeze(0).float()     # Shape: 1x256x256
        ab = torch.tensor(ab).permute(2, 0, 1).float()  # Shape: 2x256x256
        return l, ab

# -------------------------------------
# 2. Load and preprocess images
# -------------------------------------
print("Loading and preprocessing images...")
images = load_images_from_tiny_imagenet("image_colorization/dataset/tiny_image_200", max_images=1000)
lab_images = convert_to_lab(images)

dataset = ColorizationDataset(lab_images)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# -------------------------------------
# 3. Model, Loss, Optimizer
# -------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ColorizationNet().to(device)  # Uses encoder/decoder version
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -------------------------------------
# 4. Train the model
# -------------------------------------
print("Training started...")
epochs = 20
for epoch in range(epochs):
    total_loss = 0
    print(f"\nEpoch {epoch+1}/{epochs}")
    for l, ab in tqdm(dataloader, desc="Batch", leave=False):
        l, ab = l.to(device), ab.to(device)
        optimizer.zero_grad()
        output = model(l)
        loss = criterion(output, ab)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss: {total_loss/len(dataloader):.4f}")

# -------------------------------------
# 5. Save the trained model
# -------------------------------------
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/colorization_model.pth")
print("✅ Model saved to models/colorization_model.pth")
