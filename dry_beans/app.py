import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="Dry Bean Classification",
    layout="wide"
)

@st.cache_resource
def load_files():

    with open("knn_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    with open("label_encoder.pkl", "rb") as file:
        encoder = pickle.load(file)

    return model, scaler, encoder


model, scaler, encoder = load_files()

st.markdown("""
<style>

.stApp{
background-color:#0E1117;
}

.title{
font-size:42px;
font-weight:bold;
color:#7CFC00;
text-align:center;
}

.subtitle{
font-size:18px;
color:white;
text-align:center;
}

.block{
background:#1B263B;
padding:18px;
border-radius:15px;
margin-bottom:12px;
}

.footer{
text-align:center;
color:gray;
padding-top:20px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Dry Bean Classification using KNN</div>", unsafe_allow_html=True)

# st.markdown("<div class='subtitle'>Machine Learning Project | K-Nearest Neighbors Classifier</div>", unsafe_allow_html=True)

# st.divider()

# st.sidebar.header("Model Information")

# st.sidebar.write("Algorithm : KNN Classifier")

# st.sidebar.write("Dataset : Dry Bean")

# st.sidebar.write("Classes : 7")

# st.sidebar.write("Scaling : StandardScaler")

# st.sidebar.success("Model Loaded Successfully")

st.subheader("Enter Bean Features")

col1, col2 = st.columns(2)
with col1:

    area = st.number_input("Area", value=50000.0)

    perimeter = st.number_input("Perimeter", value=800.0)

    major = st.number_input("Major Axis Length", value=300.0)

    minor = st.number_input("Minor Axis Length", value=200.0)

    aspect = st.number_input("Aspect Ratio", value=1.5)

    eccentricity = st.number_input("Eccentricity", value=0.70)

    convex = st.number_input("Convex Area", value=51000.0)

    equiv = st.number_input("Equivalent Diameter", value=250.0)

with col2:

    extent = st.number_input("Extent", value=0.75)

    solidity = st.number_input("Solidity", value=0.98)

    roundness = st.number_input("Roundness", value=0.80)

    compactness = st.number_input("Compactness", value=0.85)

    shape1 = st.number_input("Shape Factor 1", value=0.005)

    shape2 = st.number_input("Shape Factor 2", value=0.0015)

    shape3 = st.number_input("Shape Factor 3", value=0.99)

    shape4 = st.number_input("Shape Factor 4", value=0.998)

if st.button("🌱 Predict Bean Type", use_container_width=True):

    data = np.array([[
        area,
        perimeter,
        major,
        minor,
        aspect,
        eccentricity,
        convex,
        equiv,
        extent,
        solidity,
        roundness,
        compactness,
        shape1,
        shape2,
        shape3,
        shape4
    ]])

    data = scaler.transform(data)

    prediction = model.predict(data)

    bean = encoder.inverse_transform(prediction)

    probability = np.max(model.predict_proba(data)) * 100

    # st.success("Prediction Completed Successfully!")

    st.markdown("## 🌾 Predicted Bean Type")

    st.info(bean[0])

    # st.metric("Confidence", f"{probability:.2f}%")

st.divider()

