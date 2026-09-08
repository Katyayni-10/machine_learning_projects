import streamlit as st
import numpy as np
import pickle

st.set_page_config(
    page_title="Wine Quality Classification",
    layout="wide"
)


@st.cache_resource
def load_model():

    with open("svc_model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    return model, scaler


model, scaler = load_model()

st.markdown("""
<style>

.stApp{
background-color:#0E1117;
}

.title{
font-size:42px;
font-weight:bold;
color:#C2185B;
text-align:center;
}

.subtitle{
font-size:18px;
color:white;
text-align:center;
margin-bottom:20px;
}

.result{
background:#1B263B;
padding:20px;
border-radius:15px;
text-align:center;
font-size:30px;
font-weight:bold;
color:#00E676;
}

.footer{
text-align:center;
color:gray;
padding-top:25px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Wine Quality Classification</div>", unsafe_allow_html=True)

st.subheader("Enter Wine Properties")

col1, col2 = st.columns(2)

with col1:

    fixed_acidity = st.number_input("Fixed Acidity", value=7.4)

    volatile_acidity = st.number_input("Volatile Acidity", value=0.70)

    citric_acid = st.number_input("Citric Acid", value=0.00)

    residual_sugar = st.number_input("Residual Sugar", value=1.90)

    chlorides = st.number_input("Chlorides", value=0.076)

    free_so2 = st.number_input("Free Sulfur Dioxide", value=11.0)

with col2:

    total_so2 = st.number_input("Total Sulfur Dioxide", value=34.0)

    density = st.number_input("Density", value=0.9978, format="%.4f")

    ph = st.number_input("pH", value=3.51)

    sulphates = st.number_input("Sulphates", value=0.56)

    alcohol = st.number_input("Alcohol", value=9.40)


if st.button("🍷 Predict Wine Quality", use_container_width=True):

    sample = np.array([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_so2,
        total_so2,
        density,
        ph,
        sulphates,
        alcohol
    ]])

    sample = scaler.transform(sample)

    prediction = model.predict(sample)

    st.markdown(
        f"""
        <div class='result'>
        Predicted Wine Quality <br><br>
        Quality Score : {prediction[0]}
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()
