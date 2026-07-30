# app.py — Deploy the Face Mask Detection model
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="centered"
)
st.markdown("""
<style>

.main {
    background-color: #f8f9fa;
}

h1 {
    color: #0E76A8;
    text-align: center;
}

div[data-testid="stFileUploader"] {
    border: 2px dashed #4CAF50;
    border-radius: 10px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load Model
# -----------------------------
import keras

st.write("TensorFlow Version:", tf.__version__)
st.write("Keras Version:", keras.__version__)

model = tf.keras.models.load_model("face_mask_model.h5")
class_labels = ["WithMask", "WithoutMask"]

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📌 About")
st.sidebar.write("""
This application detects whether a person is wearing a face mask using a
deep learning model based on **EfficientNetB0**.

**Model:** EfficientNetB0  
**Framework:** TensorFlow + Streamlit
""")

st.sidebar.info("Upload a clear face image in JPG, JPEG or PNG format.")

# -----------------------------
# Main UI
# -----------------------------
st.title("😷 Face Mask Detection")
st.markdown(
    "Upload an image to check whether the person is **wearing a face mask**."
)

st.divider()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(
        img,
        caption="Uploaded Image",
        use_container_width=True
    )

    img_resized = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    with st.spinner("Analyzing image..."):
        predictions = model.predict(img_array, verbose=0)

    score = predictions[0]
    predicted_class = class_labels[np.argmax(score)]
    confidence = float(np.max(score))

    st.divider()

    if predicted_class == "WithMask":
        st.success("✅ Person is Wearing a Face Mask")
    else:
        st.error("❌ Person is NOT Wearing a Face Mask")

    st.write(f"### Confidence: {confidence*100:.2f}%")

    st.progress(confidence)

    st.subheader("Prediction Probabilities")

    st.write(f"😷 With Mask: **{score[0]*100:.2f}%**")
    st.write(f"🙂 Without Mask: **{score[1]*100:.2f}%**")