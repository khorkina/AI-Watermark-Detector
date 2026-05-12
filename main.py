import streamlit as st
from PIL import Image
import numpy as np
import cv2
from scipy.fftpack import dct

st.set_page_config(
    page_title="AI Watermark Detector",
    layout="centered"
)

st.title("AI Watermark Detector")

st.write(
    "Upload an image. The app analyzes high-frequency image patterns "
    "and returns a rough AI-photo probability score."
)

uploaded_file = st.file_uploader(
    "Upload image",
    type=["png", "jpg", "jpeg", "webp"]
)


def calculate_high_frequency_score(image: Image.Image) -> float:
    img = image.convert("RGB")
    arr = np.array(img)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (512, 512))

    gray_float = np.float32(gray) / 255.0

    laplacian = cv2.Laplacian(gray_float, cv2.CV_32F)
    high_freq_energy = np.mean(np.abs(laplacian))

    return float(high_freq_energy)


def calculate_dct_score(image: Image.Image) -> float:
    img = image.convert("L")
    img = img.resize((512, 512))

    arr = np.float32(np.array(img)) / 255.0

    dct_result = dct(dct(arr.T, norm="ortho").T, norm="ortho")

    high_freq = dct_result[256:, 256:]
    score = np.mean(np.abs(high_freq))

    return float(score)


def classify_image(high_freq_score: float, dct_score: float):
    combined = (high_freq_score * 8.0) + (dct_score * 40.0)
    probability = min(max(combined * 100.0, 0), 100)

    if probability >= 60:
        label = "AI PHOTO"
    else:
        label = "NOT AI PHOTO"

    return label, probability


if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded image", use_container_width=True)

    high_freq_score = calculate_high_frequency_score(image)
    dct_score = calculate_dct_score(image)

    label, probability = classify_image(high_freq_score, dct_score)

    st.subheader("Result")
    st.metric("Classification", label)
    st.metric("AI probability score", f"{probability:.2f}%")

    st.subheader("Technical scores")
    st.write(f"High-frequency score: `{high_freq_score:.6f}`")
    st.write(f"DCT score: `{dct_score:.6f}`")

    st.warning(
        "This is a demo statistical detector. It is not an official OpenAI, "
        "Google SynthID, or C2PA verifier."
    )
else:
    st.info("Upload an image to start.")
