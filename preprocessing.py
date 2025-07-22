import os
import cv2
import numpy as np
from PIL import Image

def load_images_from_tiny_imagenet(folder, size=(256, 256), max_images=1000):
    image_list = []
    train_path = os.path.join(folder, "train")

    class_folders = [os.path.join(train_path, d) for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]

    for class_folder in class_folders:
        images_path = os.path.join(class_folder, "images")
        for img_file in os.listdir(images_path):
            if img_file.endswith(".JPEG"):
                img_path = os.path.join(images_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, size)
                    image_list.append(img)
            if len(image_list) >= max_images:
                break
        if len(image_list) >= max_images:
            break

    return np.array(image_list)

def convert_to_lab(images):
    lab_images = [cv2.cvtColor(img, cv2.COLOR_BGR2LAB) for img in images]
    return np.array(lab_images)

# ✅ ADD THESE TWO BELOW

def load_image(image_path, size=(64, 64)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size)
    img = np.array(img) / 255.0
    return img

def to_grayscale(image_rgb):
    """
    Converts a color image (PIL.Image) to grayscale using luminance formula.

    Args:
        image_rgb: PIL.Image in RGB format

    Returns:
        PIL.Image in grayscale
    """
    img_np = np.array(image_rgb)  # Convert to NumPy array
    gray_np = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])  # Grayscale formula
    gray_img = Image.fromarray(gray_np.astype(np.uint8))  # Convert back to PIL image
    return gray_img

