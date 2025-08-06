import streamlit as st
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
import numpy as np
from PIL import Image

# Title and UI Setup
st.set_page_config(page_title="Dog Breed Identifier", layout="centered")
st.title("🐶 Dog Breed Identifier")
st.write("Upload an image of a dog and I'll tell you what breed it is!")

# Load model and labels (you MUST have these already loaded or load them here)
@st.cache_resource
def load_model(model_path):
    print(f"loading model from: {model_path}")
    model = keras.models.load_model(model_path,
                                    custom_objects={"KerasLayer": hub.KerasLayer})
    return model


@st.cache_data
def load_labels():
    import pandas as pd
    labels_csv = pd.read_csv("I:/Dog-Breed-Identification-Using-DeepLearning/data/labels (1).csv")
    unique_breeds = np.unique(labels_csv['breed'].to_numpy())
    return unique_breeds

model = load_model()
unique_breeds = load_labels()

# File uploader
uploaded_file = st.file_uploader("Upload a dog image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Dog Image", use_column_width=True)

    # Preprocess image
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0  # Normalize
    image_tensor = tf.expand_dims(image_array, axis=0)  # Add batch dimension

    # Prediction
    predictions = model.predict(image_tensor)
    predicted_index = np.argmax(predictions[0])
    predicted_breed = unique_breeds[predicted_index]

    # Show result
    st.markdown(f"### 🐾 Predicted Breed: **{predicted_breed}**")
