import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model("trained_plant_disease_model.keras")

def model_prediction(image_data):
    model = load_model()
    image = Image.open(image_data).convert("RGB")
    image = image.resize((128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar
st.sidebar.title("AgriCult")
st.sidebar.markdown("""
**Navigation**
""")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Main Banner Image
try:
    img = Image.open("PLANT-DISEASE-IDENTIFICATION\Diseases.png")
    st.image(img, use_container_width=True)
except Exception:
    st.warning("Banner image not found.")

if app_mode == "HOME":
    st.markdown("""
        <h1 style='text-align: center;'>SMART DISEASE DETECTION</h1>
        <p style='text-align: center;'>Welcome to AgriCult! Effortlessly identify plant diseases using deep learning. Select 'Disease Recognition' to get started.</p>
    """, unsafe_allow_html=True)

elif app_mode == "DISEASE RECOGNITION":
    st.header("Disease Recognition")
    st.write("Upload a leaf image to identify the disease.")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_container_width=True)
        st.info("Please upload an image to proceed.")

        if st.button("Predict"):
            try:
                with st.spinner("Predicting..."):
                    result_index = model_prediction(test_image)
                    class_name = [
                        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                        'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                        'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                        'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                        'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                        'Tomato___healthy'
                    ]
                    st.success(f"Model predicts: {class_name[result_index]}")
                    st.balloons()
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    else:
        st.info("Please upload an image to proceed.")

# Footer (always visible at the bottom)
st.markdown("""
<hr style='margin-top: 2em; margin-bottom: 0.5em;'/>
<div style='text-align: center; color: gray; font-size: 0.9em;'>
AgriCult &copy; 2025 | Powered by Streamlit & TensorFlow
</div>
""", unsafe_allow_html=True)