import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pickle

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set title
st.title("Image Classification with MobileNetV2 by Kaung Khant Kyaw_6531501213")

# Upload file
upload_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if upload_file is not None:
    # Display the uploaded image
    img = Image.open(upload_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))  # MobileNetV2 expects 224x224
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make prediction
    prediction = model.predict(x)
    top_preds = decode_predictions(prediction, top=3)[0]

    # Display predictions
    st.subheader("Predictions:")
    for i, pred in enumerate(top_preds):
        st.write(f"{i+1}.**{pred[1]}** - {round(pred[2]*100,2)}")
