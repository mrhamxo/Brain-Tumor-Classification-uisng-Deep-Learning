import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19

# Load the VGG19 model and define the custom architecture
base_model = VGG19(include_top=False, input_shape=(240, 240, 3))
x = base_model.output
flat = Flatten()(x)
class_1 = Dense(4608, activation='relu')(flat)
drop_out = Dropout(0.2)(class_1)
class_2 = Dense(1152, activation='relu')(drop_out)
output = Dense(2, activation='softmax')(class_2)
model_03 = Model(base_model.inputs, output)

# Load the weights
model_03.load_weights('vgg19_model_03.weights.h5')
st.write('Model loaded successfully.')

# Function to get class name based on prediction
def get_class_name(class_no):
    if class_no == 0:
        return "No Brain Tumor"
    elif class_no == 1:
        return "Yes Brain Tumor"

# Function to get the result from the model
def get_result(image):
    # Convert the uploaded file to an image
    image = Image.open(image).convert('RGB')
    image = image.resize((240, 240))  # Resize the image to the input shape
    input_img = np.expand_dims(np.array(image), axis=0)  # Expand dimensions to match the input shape
    result = model_03.predict(input_img)  # Predict using the model
    result01 = np.argmax(result, axis=1)[0]  # Get the index of the class with the highest probability
    return result01

# Streamlit UI
st.title("Brain Tumor Classification")
st.write("Upload an MRI image to classify if it has a brain tumor.")

# File uploader for the MRI image
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

# Display the uploaded image in a smaller size and a Predict button if an image is uploaded
if uploaded_file is not None:
    # Display the uploaded image in a smaller size
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=False, width=300)  # Adjust width to fit nicely
    
    # Add a "Predict" button
    if st.button("Predict"):
        st.write("Classifying...")

        try:
            # Get the prediction result
            value = get_result(uploaded_file)
            result = get_class_name(value)

            # Display the result
            st.success(f"Classification Result: {result}")

        except Exception as e:
            # Handle any errors during prediction
            st.error(f"An error occurred while processing the file: {str(e)}")
