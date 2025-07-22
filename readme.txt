=======================================
🎨 IMAGE COLORIZATION APP (Streamlit + PyTorch)
=======================================

A simple deep learning-powered web app built using Streamlit and PyTorch to colorize grayscale images using a custom-trained CNN model.

--------------------------------------------------
📁 PROJECT STRUCTURE
--------------------------------------------------

image_colorization/
├── app.py                        # Main Streamlit app
├── models/
│   ├── colorization_net.py      # Model architecture
│   └── colorization_model.pth   # Trained model weights
├── utils/
│   └── preprocessing.py         # Image preprocessing helper
├── requirements.txt             # Required libraries
└── readme.txt                   # Project guide

--------------------------------------------------
💻 SYSTEM REQUIREMENTS
--------------------------------------------------

- Python 3.8 or above
- pip (Python package manager)
- Virtual environment recommended

--------------------------------------------------
📦 INSTALLATION INSTRUCTIONS
--------------------------------------------------

1. Clone the repository or download the code folder.

   git clone https://github.com/your-username/image_colorization.git
   cd image_colorization

2. Install dependencies using pip:

   pip install -r requirements.txt

3. Make sure the trained model file exists:

   Place your trained model named "colorization_model.pth" inside the "models" directory.

--------------------------------------------------
🚀 RUNNING THE APP
--------------------------------------------------

To launch the Streamlit web app, run:

   streamlit run app.py

Or if you're using a custom path:

   streamlit run "d:/Saved Files/VS Code/image_colorization/app.py"

--------------------------------------------------
🖼️ APP FUNCTIONALITY
--------------------------------------------------

- Upload a grayscale image (JPG, JPEG, or PNG).
- The app will:
   → Convert it to grayscale
   → Preprocess and resize (64x64)
   → Run it through your trained model
   → Display both grayscale and colorized versions side-by-side
   → Let you download the colorized output

--------------------------------------------------
📚 TRAINING GUIDE (MODEL CREATION)
--------------------------------------------------

1. Prepare your dataset:
   - Use color images (e.g., CIFAR-10, ImageNet subset).
   - Convert to LAB color space using OpenCV or PIL.
   - Use the L channel as input and ab channels as target.

2. Create your model (see: colorization_net.py):
   - Input: (batch, 1, 64, 64)
   - Output: (batch, 2, 64, 64)

3. Training loop (example logic):

   for images in dataloader:
       grayscale_input = images[:, :1, :, :]   # L channel
       ab_target = images[:, 1:, :, :]         # ab channels

       optimizer.zero_grad()
       ab_output = model(grayscale_input)
       loss = criterion(ab_output, ab_target)
       loss.backward()
       optimizer.step()

4. Save the trained model:

   torch.save(model.state_dict(), 'models/colorization_model.pth')

--------------------------------------------------
🖼️ SAMPLE INPUT/OUTPUT IMAGES
--------------------------------------------------

- Place test grayscale images in any format (jpg, png).
- Upload through the web app interface.
- The app will generate and display the colorized version.

💡 Tip: Try using historical B&W photos or sketches to test colorization quality.

--------------------------------------------------
📤 EXPORTING YOUR MODEL
--------------------------------------------------

After training your model:

1. Save the trained weights:

   torch.save(model.state_dict(), "models/colorization_model.pth")

2. For Streamlit app, just ensure:
   - The `colorization_model.pth` file is placed under `/models`.
   - Model input shape matches (1, 64, 64).

3. If deploying, consider converting to TorchScript for optimization.

--------------------------------------------------
📌 IMPORTANT NOTES
--------------------------------------------------

- Model input is (1-channel) grayscale, resized to 64x64.
- Output is ab channels, combined with L for full LAB to RGB conversion.
- You can improve output with a larger dataset or GAN-based model.

--------------------------------------------------
🧠 TECH STACK
--------------------------------------------------

- PyTorch (model training & inference)
- Streamlit (web interface)
- PIL (image processing)
- NumPy (numerical operations)

--------------------------------------------------
👨‍💻 CREDITS
--------------------------------------------------

Project By: Mian Mansoor
Built with: PyTorch, Streamlit,NumPy, OpenCV, PIL

--------------------------------------------------
📄 LICENSE
--------------------------------------------------

For educational and academic use only.  
No commercial distribution allowed without written permission.

Link:https://drive.google.com/file/d/1TsDLFDHlQo8oL5lt9o4Nq5Vbvuk7ajPD/view?usp=sharing

Happy Colorizing! 🎨
