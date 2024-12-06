import yaml
import tensorflow as tf
from keras.models import load_model
import streamlit as st
from PIL import Image
import numpy as np

def load_config():
    with open("config.yml", "r") as file:
        return yaml.safe_load(file)

def classify_image(image, model, config):
    """
    Classify an input image and return the prediction result.
    """
    img_size = config['loader']['img_size']
    flower_names = config['classification']['flower_names']

    input_image = tf.keras.utils.load_img(image, target_size=(img_size, img_size))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array, 0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    return f"The Image belongs to {flower_names[np.argmax(result)]} with a score of {np.max(result) * 100:.2f}%"

# Load configuration
config = load_config()

# Load trained model
model_path = config['model']['saved_model_path']
model = load_model(model_path)

# Streamlit app
st.title("🌸 Flower Classification App")
upload_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if upload_file:
    st.image(upload_file, caption="Uploaded Image", use_column_width=True)
    result = classify_image(upload_file, model, config)
    st.success(result)
