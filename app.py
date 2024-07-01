import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
# from streamlit_extras import open_browser


def load_and_prep_image(image, img_shape=224):
    img = Image.open(image)
    img = img.resize((img_shape, img_shape))
    img = np.array(img) / 255.0
    return img
def pksn(img_path):
    class_names = ['healthy', 'parkinson']
    loaded_model = tf.keras.models.load_model("parkinson.h5")

    img = load_and_prep_image(img_path)
    img = np.expand_dims(img, axis=0)
    pred = loaded_model.predict(img)
    pred_class = class_names[int(tf.round(pred))]
    
    return 0 if pred_class == "healthy" else 1


st.title("Medverse AI")

# if st.sidebar.button('Return to Home Page'):
#     open_browser("https://pmp438.pythonanywhere.com/")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=False, width=150)
    
    # Save uploaded image to a temporary path
    img_path = "temp_image.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.write("Classifying...")
    prediction = pksn(img_path)
    dic = {0: "Healthy", 1: "Patient"}
    
    st.write(f"Prediction: {dic[prediction]}")

    st.markdown(
    """
    <div style="text-align: center; padding-top: 20px;">
        <a href="https://pmp438.pythonanywhere.com/" target="_blank">
            <button style="padding: 10px 20px; font-size: 16px; cursor: pointer;">Return to Home Page</button>
        </a>
    </div>
    """,
    unsafe_allow_html=True
    )
