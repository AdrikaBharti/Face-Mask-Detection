import os
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st
from tensorflow.keras.applications.efficientnet import preprocess_input

# Get the directory where app.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "face_mask.h5")

# Load the model using the dynamic path
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Face Mask Detection",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enlarged typography & modern styling
st.markdown("""
<style>
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Extra Large Titles */
    .hero-title {
        font-size: 4rem !important;
        font-weight: 900 !important;
        color: #0E76A8;
        text-align: center;
        margin-bottom: 0px;
    }
    .hero-subtitle {
        font-size: 1.6rem !important;
        color: #475569;
        text-align: center;
        margin-bottom: 2.5rem;
    }
    
    /* Larger Section Headers */
    .section-header {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #1E293B;
        margin-bottom: 1rem;
    }

    /* Larger Metrics & Alerts */
    [data-testid="stMetricValue"] {
        font-size: 2.8rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
    }
    
    /* Prominent Upload Area */
    div[data-testid="stFileUploader"] {
        border: 2px dashed #0EA5E9;
        border-radius: 12px;
        background-color: #F0F9FF;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Cached Model Loader
# -----------------------------
@st.cache_resource
def load_face_mask_model():
    model_path = "face_mask_model.h5"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found in root directory!")
        return None
    return tf.keras.models.load_model(model_path, compile=False)

model = load_face_mask_model()
class_labels = ["WithMask", "WithoutMask"]

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
   
    st.title("About App")
    st.markdown("""
    This application utilizes an **EfficientNetB0** deep learning model trained to detect whether individuals are wearing face masks.
    
    **Technical Specs:**
    * **Architecture:** EfficientNetB0
    * **Input Resolution:** 224x224
    * **Frameworks:** TensorFlow & Streamlit
    """)
    st.info("💡 **Tip:** Upload clear, well-lit portraits for highest accuracy.")

# -----------------------------
# Header Section (Enlarged)
# -----------------------------
st.markdown('<p class="hero-title">😷 Face Mask Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Real-time mask detection powered by Deep Learning</p>', unsafe_allow_html=True)

# -----------------------------
# Main Content Grid
# -----------------------------
# Column 1 = Upload & Compact Image, Column 2 = Expanded Results
col1, col2 = st.columns([1, 1.3], gap="large")

with col1:
    st.markdown('<p class="section-header">1. Upload Image</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        
        # Constrain image width strictly to 280px to keep it compact
        _, img_center, _ = st.columns([1, 2, 1])
        with img_center:
            st.image(img, caption="Uploaded Image", width=280)

with col2:
    st.markdown('<p class="section-header">2. Detection Results</p>', unsafe_allow_html=True)
    
    if uploaded_file is None:
        st.info("👈 Upload an image on the left to see prediction analytics.")
    elif model is None:
        st.error("Model couldn't be loaded. Please check your workspace files.")
    else:
        # Preprocessing
        img_resized = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_array)

        with st.spinner("Analyzing image features..."):
            predictions = model.predict(img_preprocessed, verbose=0)

        score = predictions[0]
        predicted_index = np.argmax(score)
        predicted_class = class_labels[predicted_index]
        confidence = float(np.max(score)) * 100

        # Enlarge Result Banners
        if predicted_class == "WithMask":
            st.success("# ✅ Mask Detected")
        else:
            st.error("# ❌ No Mask Detected")

        # Larger Metric Cards
        m1, m2 = st.columns(2)
        m1.metric("Status", predicted_class)
        m2.metric("Confidence", f"{confidence:.2f}%")

        st.progress(confidence / 100.0)

        # Class Probabilities Breakdown
        st.markdown("### Class Probabilities")
        st.write(f"😷 **With Mask:** {score[0]*100:.2f}%")
        st.progress(float(score[0]))
        
        st.write(f"🙂 **Without Mask:** {score[1]*100:.2f}%")
        st.progress(float(score[1]))
