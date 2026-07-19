import streamlit as st
import numpy as np
import os
from PIL import Image
import json
from keras.models import load_model

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Plant Disease Prediction",
    page_icon="🌿",
    layout="centered"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
body {
    background-color: #f5fff7;
}
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #2e7d32;
}
.sub-title {
    text-align: center;
    color: #4caf50;
    font-size: 18px;
    margin-bottom: 30px;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-top: 20px;
}
.predict-btn {
    width: 100%;
}
.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD DATA ------------------
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "trained model", "Plant_model.keras")

@st.cache_resource
def load_my_model():
    return load_model(model_path)

model = load_my_model()

# ------------------ IMAGE PROCESSING ------------------
def load_image(image_file, target_size=(224, 224)):
    img = Image.open(image_file).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(model, image_file, class_dict):
    pre_img = load_image(image_file)
    prediction = model.predict(pre_img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return class_dict[str(predicted_class)]


with st.sidebar:
    st.header("🌱 About")
    st.write(
        "This system uses a deep learning model to detect "
        "plant diseases from leaf images."
    )
    st.markdown("**Supported formats:** JPG, JPEG, PNG")
    st.markdown("**Model:** CNN (Keras)")
    st.markdown("---")
    st.markdown("👨‍💻 *Built with Streamlit*")

# ------------------ MAIN UI ------------------
st.markdown('<div class="main-title">🌿 Plant Disease Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload a leaf image to detect plant disease</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "📤 Upload Plant Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.image(uploaded_file, caption="Uploaded Leaf Image", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

    if st.button("🔍 Predict Disease", use_container_width=True):
        with st.spinner("Analyzing image... 🌱"):
            prediction = predict_image(model, uploaded_file, class_indices)

        st.success(f"✅ **Predicted Disease:** {prediction}")

    st.markdown("</div>", unsafe_allow_html=True)


st.markdown(
    '<div class="footer">🌾 AI for Smart Agriculture</div>',
    unsafe_allow_html=True
)
