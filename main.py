import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
import streamlit as st
import requests
from io import BytesIO
from PIL import Image
import yaml

# Load configuration from YAML file
def load_config():
    with open("config.yml", "r") as file:
        return yaml.safe_load(file)

# Classify image based on trained model
def classify_image(image, model, config):
    img_size = config['loader']['img_size']
    flower_names = config['classification']['flower_names']

    input_image = tf.keras.utils.load_img(image, target_size=(img_size, img_size))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    return f"The image belongs to {flower_names[np.argmax(result)]} with a score of {np.max(result) * 100:.2f}%"

# Load configuration
config = load_config()

# Load trained model
model_path = config['model']['saved_model_path']
model = load_model(model_path)

# Streamlit app setup
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

# Header
st.markdown('<h1 class="header-title">Flower Classification</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Upload Options")
upload_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
url_input = st.sidebar.text_input("Or enter an Image URL")

# Handle uploaded file
if upload_file is not None:
    st.image(upload_file, caption="Uploaded Image", use_column_width=True)
    result = classify_image(upload_file, model, config)
    st.success(result)

# Handle URL
if url_input:
    try:
        response = requests.get(url_input)
        if response.status_code == 200:
            image_data = BytesIO(response.content)
            try:
                image = Image.open(image_data)
                image = image.convert("RGB")
                st.image(image, caption="Image from URL", use_column_width=True)

                # Save image to temporary buffer
                img_buffer = BytesIO()
                image.save(img_buffer, format="PNG")
                img_buffer.seek(0)

                # Classify and predict image
                result = classify_image(img_buffer, model, config)
                st.success(result)
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
