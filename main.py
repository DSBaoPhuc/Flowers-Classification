import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import cv2

st.set_page_config(
    page_title="Flowers Classification",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main {
        background: linear-gradient(to right, #ADD8E6, #66CCFF); /* Gradient from Light Blue to Blue */
    }
    .header-title {
        color: #000000;
        font-size: 2em;
        text-align: center;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        padding: 10px;
        font-size: 12px;
        color: #000000;
    }
    .stButton>button {
        background-color: #008CBA; /* Blue */
        color: white;
        border-radius: 5px;
        font-size: 16px;
        padding: 10px 20px;
    }
    .top-left {
        position: absolute;
        top: 10px;
        left: 10px;
        font-family: 'Arial', sans-serif;
        color: #333333;
    }
    .course {
        font-size: 1.8em; /* Increase font size for course */
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Name and ID
course = "Machine Learning Platform"

# Header
st.markdown(
    '<h1 class="header-title">Flower Classification</h1>', unsafe_allow_html=True
)

flower_name = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# Load the trained model
model = load_model("Flower_Recog_Model.keras")


# Function to classify images
def classify_images(image):
    input_image = tf.keras.utils.load_img(image, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    outcome = (
        "The Image belongs to "
        + flower_name[np.argmax(result)]
        + " with a score of "
        + str((np.max(result) * 100))
    )
    return outcome


# Sidebar
st.sidebar.markdown(
    f"""
    <div style="font-family: 'Arial', sans-serif; color: #333333; margin-bottom: 20px; ">
        <div class="course">{course}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.title("Upload Options")
upload_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
url_input = st.sidebar.text_input("Or enter an Image URL")

# Handle uploaded file
if upload_file is not None:
    st.image(upload_file, width=300)
    outcome = classify_images(upload_file)
    st.success(outcome)

# Handle URL
if url_input:
    try:
        response = requests.get(url_input)
        if response.status_code == 200:
            image_data = BytesIO(response.content)
            try:
                image = Image.open(image_data)
                image = image.convert("RGB")
                st.image(image, width=300)

                # Save image to temporary buffer
                img_buffer = BytesIO()
                image.save(img_buffer, format="PNG")
                img_buffer.seek(0)

                # Classify and predict image
                outcome = classify_images(img_buffer)
                st.success(outcome)
            except Exception as e:
                st.error(f"Error processing image: {e}")
        else:
            st.error("Failed to retrieve image. Please check the URL.")
    except Exception as e:
        st.error(f"Error loading image from URL: {e}")

# Footer
st.markdown(
    """
    <div class="footer">
    Flower Classification App.
    </div>
""",
    unsafe_allow_html=True,
)
