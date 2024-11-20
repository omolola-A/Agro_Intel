import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import json

# Load the model
model = load_model(r"C:\Users\Dell\Desktop\3MTT\Hackaton 2.0\Model_training\cnn_model.keras")

# Load class labels
with open(r"C:\Users\Dell\Desktop\3MTT\Hackaton 2.0\Model_training\class_indices.json", "r") as f:
    class_indices = json.load(f)
class_labels = list(class_indices.keys())

# Predefined recommendations for each category
recommendations = {
    "Cashew anthracnose": "Apply copper-based fungicides and prune affected branches. Ensure good airflow by maintaining proper spacing between trees.",
    "Cashew gumosis": "Remove infected plant material and improve soil drainage. Apply fungicides such as phosphonate as needed.",
    "Cashew healthy": "Your cashew crop is healthy! Continue regular monitoring and maintain good farming practices.",
    "Cashew leaf miner": "Use insecticidal sprays or biological control agents like parasitic wasps. Remove heavily infested leaves.",
    "Cashew red rust": "Spray sulfur-based fungicides and remove affected leaves. Regularly monitor for early signs of infection.",
    "Cassava bacterial blight": "Plant disease-resistant cassava varieties and avoid using infected planting materials. Apply copper-based bactericides if necessary.",
    "Cassava brown spot": "Use resistant cassava varieties and ensure proper crop spacing to improve airflow. Remove and destroy infected leaves.",
    "Cassava green mite": "Introduce natural predators like *Typhlodromalus manihoti* or use neem-based miticides.",
    "Cassava healthy": "Your cassava crop is healthy! Continue regular monitoring and maintain good farming practices.",
    "Cassava mosaic": "Plant virus-resistant cassava varieties and control whiteflies using insecticides or biological agents.",
    "Maize fall armyworm": "Apply recommended insecticides or use natural predators like parasitoids. Early planting can help avoid infestation.",
    "Maize grasshoper": "Spray insecticides or apply biological agents such as fungal pathogens. Destroy egg pods in the soil during offseason.",
    "Maize healthy": "Your maize crop is healthy! Keep monitoring and follow good agricultural practices.",
    "Maize leaf beetle": "Use recommended insecticides or biological control like nematodes. Remove adult beetles manually if infestation is low.",
    "Maize leaf blight": "Use fungicides such as mancozeb or chlorothalonil. Rotate crops to prevent reinfection.",
    "Maize leaf spot": "Apply fungicides and remove affected leaves. Ensure proper crop spacing for good airflow.",
    "Maize streak virus": "Control vector insects like leafhoppers using insecticides. Plant resistant maize varieties.",
    "Tomato healthy": "Your tomato crop is healthy! Keep monitoring and maintain proper farming practices.",
    "Tomato leaf blight": "Use fungicides like mancozeb or chlorothalonil. Remove and destroy infected leaves.",
    "Tomato leaf curl": "Control whiteflies using insecticides or natural predators. Remove and destroy infected plants.",
    "Tomato septoria leaf spot": "Apply fungicides and avoid overhead irrigation. Remove and destroy infected leaves.",
    "Tomato verticulium wilt": "Plant disease-resistant tomato varieties and rotate crops to prevent soil contamination."
}

# Function to get recommendations offline
def get_recommendations_offline(predicted_class):
    return recommendations.get(predicted_class, "No recommendation available for this category.")

# App Title
st.title("Crop Pest and Disease Prediction App with Recommendations")

# File uploader
uploaded_file = st.file_uploader("Upload a Crop Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = load_img(uploaded_file, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]

    # Show result
    st.write(f"Prediction: **{predicted_class}**")

    # Get offline recommendations
    advice = get_recommendations_offline(predicted_class)
    st.write(f"Recommendations: {advice}")
