import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import io

from utils.preprocessing import to_grayscale
from models.colorization_net import ColorizationNet

st.set_page_config(page_title="Image Colorizer", layout="wide")
st.title("🎨 Image Colorization App")
st.markdown("Upload a grayscale image to see the colorized output using your trained model.")

uploaded_file = st.file_uploader("Upload a grayscale image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and preprocess
    original_image = Image.open(uploaded_file).convert("RGB")
    grayscale_image = to_grayscale(original_image).convert("L")  # Grayscale

    # Resize for model
    grayscale_image_resized = grayscale_image.resize((64, 64))
    transform = transforms.ToTensor()
    l_tensor = transform(grayscale_image_resized).unsqueeze(0)  # Shape: [1, 1, 64, 64]

    # Load model
    model = ColorizationNet()
    model.load_state_dict(torch.load("models/colorization_model.pth", map_location=torch.device('cpu')))
    model.eval()

    # Run model to predict ab channels
    with torch.no_grad():
        ab_output = model(l_tensor)  # [1, 2, 64, 64]

    # Convert L + ab → Lab image
    l_np = l_tensor[0][0].cpu().numpy() * 100  # Rescale L to 0-100
    ab_np = ab_output[0].cpu().numpy() * 128   # Rescale ab to -128 to 127
    lab_image = np.zeros((64, 64, 3), dtype=np.float32)
    lab_image[:, :, 0] = l_np
    lab_image[:, :, 1:] = ab_np.transpose(1, 2, 0)

    # Convert Lab to RGB
    rgb_image = cv2.cvtColor(lab_image, cv2.COLOR_Lab2RGB)
    rgb_image = np.clip(rgb_image, 0, 1)
    rgb_pil = Image.fromarray((rgb_image * 255).astype(np.uint8))

    # Display side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Grayscale Input")
        st.image(grayscale_image, use_container_width=True)

    with col2:
        st.subheader("Colorized Output")
        st.image(rgb_pil, use_container_width=True)

        # Download button
        buffer = io.BytesIO()
        rgb_pil.save(buffer, format="PNG")
        st.download_button(
            label="📥 Download Colorized Image",
            data=buffer.getvalue(),
            file_name="colorized_output.png",
            mime="image/png"
        )
