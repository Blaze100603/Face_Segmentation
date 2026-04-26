#  Face Segmentation using Deep Learning

##  Project Overview

This project implements a **Face Segmentation System** using a **U-Net Convolutional Neural Network** to detect and isolate facial regions from images.
The model is trained on annotated face bounding-box data converted into binary segmentation masks and deployed through an interactive **Streamlit web application**.

Users can upload images, run real-time face segmentation, visualize mask overlays, and download prediction outputs.

---

##  Features

* Face region segmentation using Deep Learning
* U-Net architecture for pixel-wise prediction
* Real-time inference using TensorFlow
* Interactive Streamlit web app
* Adjustable threshold and overlay transparency
* Download segmented mask, overlay image, and report
* Lightweight deployment-ready model

---

##  Model Architecture

* **Architecture:** U-Net
* **Input Size:** 256 × 256 × 3
* **Output Size:** 256 × 256 × 1
* **Framework:** TensorFlow / Keras

---

##  Project Structure

```text
final_project/
│── streamlit_app.py
│── train.npy
│── requirements.txt
│── README.md
│
└── models/
    ├── unet_mobilenetv2_model.keras
    ├── unet_mobilenetv2_model.h5
    ├── deployment_config.json
```

---

##  Dataset Information

* File: `train.npy`
* Total Samples: **409**
* Image Size: **256 × 256**
* Masks generated from face annotations

---

##  Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

##  Run Streamlit App

```bash
streamlit run streamlit_app.py
```

---

##  Usage

1. Launch the Streamlit app
2. Upload an image (`jpg`, `jpeg`, `png`)
3. View:

   * Original image
   * Segmentation mask
   * Overlay result
4. Download outputs

---

##  Output Metrics

The application displays:

* Face Coverage %
* Face Pixels Count
* Inference Time (ms)

---

##  Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* OpenCV
* Streamlit
* Matplotlib
* Scikit-learn

---

##  Future Improvements

* Real face contour segmentation masks
* Multi-face instance segmentation
* Live webcam support
* Faster inference using TensorFlow Lite / ONNX
* Mobile deployment

---

##  Author

Developed as a Deep Learning Computer Vision Project.

---
