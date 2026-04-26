

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import io
import time
import json
import os


st.set_page_config(
    page_title="Face Segmentation App",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color:#0e1117;
    color:white;
}
h1,h2,h3 {
    color:white;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_segmentation_model():

    possible_paths = [
        "models/unet_mobilenetv2_model.keras",
        "models/unet_mobilenetv2_model.h5",
        "models/face_segmentation_model.keras",
        "face_segmentation_model.keras"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return tf.keras.models.load_model(path, compile=False)

    return None


model = load_segmentation_model()
IMG_SIZE = 256

def preprocess_image(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def postprocess_mask(pred_mask, threshold=0.5):
    mask = pred_mask[0, :, :, 0]
    mask = (mask > threshold).astype(np.uint8) * 255

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def create_overlay(image, mask, alpha=0.4):
    image = cv2.resize(image, (256,256))

    green = np.zeros_like(image)
    green[:,:,1] = 255

    mask_3 = cv2.merge([mask,mask,mask]) / 255.0
    overlay = image.copy()

    overlay = image * (1-mask_3) + ((1-alpha)*image + alpha*green) * mask_3
    overlay = overlay.astype(np.uint8)

    return overlay

def calculate_metrics(mask):
    binary = (mask > 0).astype(np.uint8)

    total = binary.size
    face_pixels = binary.sum()
    coverage = (face_pixels / total) * 100

    return coverage, face_pixels

st.title(" Real-Time Face Segmentation")
st.markdown("Upload image → Predict face regions → Download result")

st.sidebar.header("⚙ Settings")

threshold = st.sidebar.slider(
    "Mask Threshold",
    0.05, 0.95, 0.35, 0.05
)

alpha = st.sidebar.slider(
    "Overlay Transparency",
    0.1, 1.0, 0.45, 0.05
)

st.sidebar.markdown("---")

if model is not None:
    st.sidebar.success("✅ Model Loaded")
else:
    st.sidebar.error("❌ Model Not Found")

uploaded_file = st.file_uploader(
    "Upload an Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    col1, col2, col3 = st.columns(3)

    with st.spinner("Running inference..."):

        start = time.time()

        inp = preprocess_image(image_np)

        pred = model.predict(inp, verbose=0)

        infer_time = (time.time() - start) * 1000

        mask = postprocess_mask(pred, threshold)

        overlay = create_overlay(image_np, mask, alpha)

        coverage, face_pixels = calculate_metrics(mask)

    col1.image(image_np, caption="Original", use_container_width=True)
    col2.image(mask, caption="Mask", use_container_width=True)
    col3.image(overlay, caption="Overlay", use_container_width=True)

    st.markdown("---")

    m1, m2, m3 = st.columns(3)

    m1.metric("Coverage", f"{coverage:.2f}%")
    m2.metric("Face Pixels", f"{face_pixels}")
    m3.metric("Inference", f"{infer_time:.2f} ms")

    st.markdown("---")
    st.subheader("⬇ Download Results")

    c1, c2, c3 = st.columns(3)

    buf1 = io.BytesIO()
    Image.fromarray(mask).save(buf1, format="PNG")

    c1.download_button(
        "Download Mask",
        data=buf1.getvalue(),
        file_name="mask.png",
        mime="image/png"
    )

    buf2 = io.BytesIO()
    Image.fromarray(overlay).save(buf2, format="PNG")

    c2.download_button(
        "Download Overlay",
        data=buf2.getvalue(),
        file_name="overlay.png",
        mime="image/png"
    )

    report = {
        "coverage_percent": float(coverage),
        "face_pixels": int(face_pixels),
        "threshold": float(threshold),
        "inference_ms": float(infer_time)
    }

    c3.download_button(
        "Download Report",
        data=json.dumps(report, indent=4),
        file_name="report.json",
        mime="application/json"
    )

st.markdown("---")