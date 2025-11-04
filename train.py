# Fake News Image Detector - Beginner Version
# Run in Google Colab: https://colab.research.google.com

!pip install tensorflow matplotlib seaborn scikit-learn numpy

import urllib.request
import zipfile
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Step 1: Download Dataset
print("Downloading pictures... 🖼️")
urllib.request.urlretrieve("https://github.com/subeeshvasu/2017-ICASSP-Awesome-Tampering-Dataset/releases/download/v1.0/CASIA2.zip", "data.zip")
with zipfile.ZipFile("data.zip", 'r') as zip_ref:
    zip_ref.extractall("data")
print("Done! Real in data/Au, Fake in data/Tp")

# Step 2: Load Data
img_size = (224, 224)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_gen = datagen.flow_from_directory('data', target_size=img_size, batch_size=32, class_mode='binary', subset='training')
val_gen = datagen.flow_from_directory('data', target_size=img_size, batch_size=32, class_mode='binary', subset='validation')

# Step 3: Build Model
base_model = tf.keras.applications.MobileNetV2(input_shape=img_size + (3,), include_top=False, weights='imagenet')
base_model.trainable = False
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Model ready! 🧠")

# Step 4: Train
print("Training... (5-10 mins) ☕")
history = model.fit(train_gen, validation_data=val_gen, epochs=5)

# Step 5: Save Model
model.save('fake_detector.h5')
print("Model saved! Download it!")

# Step 6: Test Function
def check_image(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=img_size)
    img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]
    label = "FAKE" if pred > 0.5 else "REAL"
    print(f"{label}! Confidence: {pred:.2f}")
    return label, pred

# Upload a test image in Colab and run: check_image('your_photo.jpg')
