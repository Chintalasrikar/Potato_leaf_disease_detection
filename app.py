import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os

file_id = "1SgL1OsfeCaujkvm9QCYmZBf67C8ObUXL"
url = 'https://drive.google.com/file/d/1SgL1OsfeCaujkvm9QCYmZBf67C8ObUXL/view?usp=drive_link'
model_path = "potato.keras"

if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

model_path = "trained_plant_disease_model.keras"
def model_prediction(test_image):
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element
# Sidebar
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])
#app_mode = st.sidebar.selectbox("Select Page", ["Home", " ", "Disease Recognition"])
