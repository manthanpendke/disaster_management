import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model  # Corrected import
from PIL import Image

# Load the pre-trained model
model = load_model('model.h5')  # Replace 'model.h5' with the path to your model file

# Define category mappings
disaster_names = {0: "cyclone", 1: "earthquake", 2: "flood", 3: "wildfire"}
casualty_names = {0: "low", 1: "medium", 2: "high", 3: "critical"}
danger_names = {0: "low", 1: "medium", 2: "high", 3: "extreme"}
authority_names = {0: "local", 1: "state", 2: "central", 3: "international"}

# Streamlit App
st.title("Disaster Response Prediction App")
st.write("Upload an image to predict disaster type, casualty level, danger level, and the required authority response.")

# File upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png"])

if uploaded_file is not None:
    try:
        # Preprocess the image using Pillow
        image = Image.open(uploaded_file).resize((224, 224))
        image_array = np.asarray(image) / 255.0  # Normalize pixel values to [0, 1]
        input_data = np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict using the model
        predictions = model.predict(input_data)

        # Display predictions
        st.subheader("Predictions:")
        st.write(f"**Disaster Type:** {disaster_names[np.argmax(predictions[0])]}")
        st.write(f"**Casualty Level:** {casualty_names[np.argmax(predictions[1])]}")
        st.write(f"**Danger Level:** {danger_names[np.argmax(predictions[2])]}")
        st.write(f"**Authority Level:** {authority_names[np.argmax(predictions[3])]}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
