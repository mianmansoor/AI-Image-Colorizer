# 🎨 Image Colorization App

A simple deep learning-powered web app built with **Streamlit** and **PyTorch** to colorize grayscale images using a custom-trained convolutional neural network.


## 📁 Project Structure

image\_colorization/
├── app.py                        # Main Streamlit app
├── models/
│   ├── colorization\_net.py      # Model architecture
│   └── colorization\_model.pth   # Trained model weights
├── utils/
│   └── preprocessing.py         # Image preprocessing logic
├── requirements.txt
└── README.md


## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/image_colorization.git
cd image_colorization
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> ✅ Python 3.8+ recommended

---

## 🚀 Run the App

You can launch the Streamlit app by running:

```bash
streamlit run app.py
```

Or if your file is in a custom path:

```bash
streamlit run "d:/Saved Files/VS Code/image_colorization/app.py"
```

## ⚙️ Features

* Upload grayscale images (JPG, JPEG, PNG)
* Display side-by-side comparison of grayscale and colorized images
* Download button for the output image
* LAB color space-based processing for better results

---

## 📌 Notes

* The model input is expected in grayscale (1 channel), size 64×64.
* This project is optimized for learning, not production accuracy.
* Make sure the input image is clean and has distinguishable edges.

---

## 🤝 Acknowledgements

Created by **Mian Mansoor**
Using: Streamlit • PyTorch • Pillow • NumPy

---

## 📄 License

This project is for educational purposes. No license file has been attached yet.

```

---

Let me know if you want:
- A live preview badge (if you plan to deploy on Streamlit Cloud)
- Sample input/output images
- A short training guide for the model
```
