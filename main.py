import streamlit as st
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# App setup
st.set_page_config(page_title="Dog Breed Identifier", layout="centered")
st.markdown("<h1 style='text-align: center;'>🐶 Dog Breed Identifier</h1>", unsafe_allow_html=True)
st.write("Upload an image of a dog and I'll tell you what breed it is!")

# Load model and labels
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

# UI container
with st.container():
    uploaded_file = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Show spinner while predicting
        with st.spinner("Predicting breed..."):

            # Load and preprocess image
            image = Image.open(uploaded_file).convert("RGB")
            image_resized = image.resize((224, 224))
            image_array = np.array(image_resized) / 255.0
            image_tensor = tf.expand_dims(image_array, axis=0)

            # Prediction
            predictions = model.predict(image_tensor)
            predicted_index = np.argmax(predictions[0])
            predicted_breed = unique_breeds[predicted_index]

        # Create centered layout using columns
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                f"<h3 style='color:green; text-align: left;'>🐾 Predicted Breed: <b>{predicted_breed}</b></h3>",
                unsafe_allow_html=True
            )
            st.image(image, caption="Uploaded Dog Image", width=200)

