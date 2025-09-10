import streamlit as st
import numpy as np
import pickle
import base64
import gdown
import os

model_path = "invest_wise_classifier.pkl"
file_id = "1IVrXG6o9N14j1KAN5_QzQxNmPCYqgE1Q"
gdrive_url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(model_path):
    gdown.download(gdrive_url, model_path, quiet=False)

with open(model_path, "rb") as f:
    model = pickle.load(f)

# Function to set the background image (use your JPEG or PNG here)
def set_background_jpeg(jpeg_file):
    with open(jpeg_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
        }}

        /* Bold and bright labels for input fields */
        label[data-baseweb="label"] {{
            font-weight: bold !important;
            color: #FFD700 !important;  /* Bright gold */
            text-shadow: 0 0 5px rgba(0,0,0,0.7);
        }}

        /* Style the buttons */
        div.stButton > button {{
            font-weight: bold !important;
            font-size: 18px !important;
            background-color: #FFD700 !important;
            color: black !important;
            border-radius: 8px !important;
            padding: 8px 15px !important;
        }}

        /* Style success and error message fonts */
        .stAlert > div > div > div > p {{
            font-weight: bold !important;
            font-size: 20px !important;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )

# Set background image file name here
set_background_jpeg("background.jpeg")

# Sidebar About Us section with emojis
st.sidebar.title("About Us 🏡📊")
st.sidebar.markdown("""
Welcome to *Invest Wise* — your trusted partner in making smarter real estate investment decisions. 🏠✨  
Our mission is to simplify the complex world of property investments by providing clear, data-backed recommendations tailored to your needs. Using advanced machine learning models, we analyze key factors like property type, location, price, and market trends to guide you toward the best investment opportunities. 📈💡  
At Invest Wise, we believe that informed decisions lead to better financial futures. Whether you’re a first-time buyer or a seasoned investor, our user-friendly chatbot makes understanding real estate investments easy and accessible. 🤖👍  
Thank you for choosing Invest Wise — investing made wise, simple, and confident. 💰✅
""")

# Load saved classifier model
with open("invest_wise_classifier.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Invest Wise: Smart Property Classifier for Mumbai City 💬")

# User input fields matching your schema
bhk = st.number_input("Number of bedrooms, halls, kitchens (bhk) 🛏🛋🍽", min_value=1, step=1, format="%d")

type_dict = {
    "Apartment": 0,
    "Villa": 1,
    "Studio apartment": 2,
    "Penthouse": 3,
    "Independent house": 4
}
type_name = st.selectbox("Property type 🏘", list(type_dict.keys()))
type_encoded = type_dict[type_name]

area = st.number_input("Area in square feet 📐", min_value=1, step=1, format="%d")

price = st.number_input("Property price (in lakhs) 💵", min_value=0.0, step=0.01, format="%.2f")

region_dict = {
    "Navi Mumbai & Thane": 0,
    "Western Suburbs": 1,
    "Extended Western Suburbs": 2,
    "Central Suburbs": 3,
    "Extended Eastern Suburbs": 4,
    "Harbour Suburbs": 5,
    "South Mumbai": 6,
    "Other areas": 7
}
region_name = st.selectbox("Region/zone of the city 🗺", list(region_dict.keys()))
region_encoded = region_dict[region_name]

status_dict = {
    "Ready to move": 0,
    "Under Construction": 1
}
status_name = st.selectbox("Property status 🏗", list(status_dict.keys()))
status_encoded = status_dict[status_name]

age_dict = {
    "New": 0,
    "Resale": 1
}
age_name = st.selectbox("Property age classification 🕰", list(age_dict.keys()))
age_encoded = age_dict[age_name]

if st.button("Predict Investment 🔍"):
    input_features = np.array([[bhk, type_encoded, area, price, region_encoded, status_encoded, age_encoded]])
    prediction = model.predict(input_features)
    if prediction[0] == 'Good Investment':
        st.success("🎉 This is a Good Investment! 🚀")
    else:
        st.error("⚠ This might not be the best Investment.")
