# Run this in Colab for a web app!
!pip install streamlit pyngrok

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('fake_detector.h5')
st.title("🕵️ Fake News Image Detector")
st.write("Upload a news photo – I'll tell if it's fake!")

file = st.file_uploader("Choose image", type=['jpg', 'png', 'jpeg'])
if file:
    img = Image.open(file).resize((224, 224))
    st.image(img, "Your Image")
    img_array = np.array(img) / 255.0
    img_array = img_array[np.newaxis, ...]
    pred = model.predict(img_array)[0][0]
    if pred > 0.5:
        st.error(f"**FAKE NEWS!** 😱 Confidence: {pred:.2f}")
    else:
        st.success(f"**REAL!** 📰 Confidence: {1-pred:.2f}")
