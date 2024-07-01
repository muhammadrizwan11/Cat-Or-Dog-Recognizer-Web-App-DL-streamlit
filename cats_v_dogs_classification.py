import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Function to load and preprocess the uploaded image
def load_and_process_image(image_file):
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (256, 256))  # Resize to match model's expected sizing
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize pixel values
    return img

# Load your trained model
model_path = 'cats-v-dogs-classification_model.h5'  # Update with your model path
model = tf.keras.models.load_model(model_path)

# Streamlit UI
st.title('Cat or Dog Classifier')
st.write('Upload an image and we will predict whether it\'s a cat or a dog!')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = load_and_process_image(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make predictions
    prediction = model.predict(image)
    class_names = ['Cat', 'Dog']
    predicted_class = class_names[int(prediction[0, 0])]
    st.write(f'Prediction: {predicted_class}')

