import streamlit as st
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import base64
from io import BytesIO

# Streamlit page setup
st.set_page_config(page_title="Dog Breed Identifier", layout="centered")

# Custom CSS Styling
st.markdown("""
    <style>
        .title {
            font-size: 36px;
            font-weight: 700;
            text-align: center;
            color: #1e272e;
            margin-bottom: 5px;
        }
        .desc {
            text-align: center;
            color: #8395a7;
            font-size: 16px;
            margin-bottom: 30px;
        }
        .label-container {
            background-color: #ffecec;
            border-left: 6px solid #e74c3c;
            padding: 20px;
            margin: 20px auto 20px auto;
            width: 50%;
            text-align: center;
            border-radius: 12px;
            box-shadow: 0 0 6px rgba(0,0,0,0.05);
        }
        .breed-name {
            color: #e74c3c;
            font-size: 24px;
            font-weight: bold;
            margin: 0;
        }
        .center-img {
            display: block;
            margin-left: auto;
            margin-right: auto;
            border-radius: 12px;
            width: 35%;
            margin-top: 25px;
            box-shadow: 0 0 8px rgba(0, 0, 0, 0);
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>🐶 Dog Breed Identifier🐶 </div>", unsafe_allow_html=True)
st.markdown("<div class='desc'>Upload a dog pic</div>", unsafe_allow_html=True)

# Load model + labels
@st.cache_resource
def load_model(model_path):
    model = keras.models.load_model(model_path,
                                    custom_objects={"KerasLayer": hub.KerasLayer})
    return model

@st.cache_data
def load_labels():
    import pandas as pd
    labels_csv = pd.read_csv("I:/Dog-Breed-Identification-Using-DeepLearning/data/labels (1).csv")
    unique_breeds = np.unique(labels_csv['breed'].to_numpy())
    return unique_breeds

model = load_model("I:/Dog-Breed-Identification-Using-DeepLearning/data/20250803-060103-full-images-dataset-10000-images-mobilenet.h5")
unique_breeds = load_labels()

# Upload section
uploaded_file = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png"])

# Prediction section
if uploaded_file:
    with st.spinner("Finding the fluff identity..."):
        image = Image.open(uploaded_file).convert("RGB")
        image_resized = image.resize((224, 224))
        image_array = np.array(image_resized) / 255.0
        image_tensor = tf.expand_dims(image_array, axis=0)

        predictions = model.predict(image_tensor)
        predicted_index = np.argmax(predictions[0])
        predicted_breed = unique_breeds[predicted_index]

        # Convert image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Render prediction label container
        st.markdown(f"""
            <div class='label-container'>
                <p class='breed-name'>🐶{predicted_breed}🐶 </p>
            </div>
        """, unsafe_allow_html=True)

        # Render image separately below
        st.markdown(f"""
            <img src="data:image/jpeg;base64,{img_str}" class="center-img" />
        """, unsafe_allow_html=True)
