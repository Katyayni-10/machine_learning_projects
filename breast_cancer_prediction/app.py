import streamlit as st
import numpy as np
import pickle

st.set_page_config(
    page_title="Breast Cancer Prediction",
    layout="centered"
)

with open("breast_cancer_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

st.title("Breast Cancer Prediction")
st.write("Enter the patient's medical measurements below.")

col1, col2 = st.columns(2)
with col1:
    radius_mean = st.number_input(
        "Radius Mean",
        min_value=0.0,
        step=0.01
    )
with col2:
    perimeter_mean = st.number_input(
        "Perimeter Mean",
        min_value=0.0,
        step=0.01
    )

col3, col4 = st.columns(2)
with col3:
    area_mean = st.number_input(
        "Area Mean",
        min_value=0.0,
        step=0.01
    )
with col4:
    concavity_mean = st.number_input(
        "Concavity Mean",
        min_value=0.0,
        step=0.001,
        format="%.3f"
    )

col5, col6 = st.columns(2)
with col5:
    concave_points_mean = st.number_input(
        "Concave Points Mean",
        min_value=0.0,
        step=0.001,
        format="%.3f"
    )
with col6:
    radius_worst = st.number_input(
        "Radius Worst",
        min_value=0.0,
        step=0.01
    )

col7, col8 = st.columns(2)
with col7:
    perimeter_worst = st.number_input(
        "Perimeter Worst",
        min_value=0.0,
        step=0.01
    )
with col8:
    area_worst = st.number_input(
        "Area Worst",
        min_value=0.0,
        step=0.01
    )

st.write("")

if st.button("Predict", use_container_width=True):
    data = np.array([[
        radius_mean,
        perimeter_mean,
        area_mean,
        concavity_mean,
        concave_points_mean,
        radius_worst,
        perimeter_worst,
        area_worst
    ]])
    data = scaler.transform(data)
    prediction = model.predict(data)
    if prediction[0] == 1:
        st.error("Malignant (Cancer Detected)")
    else:
        st.success("Benign (No Cancer Detected)")

st.markdown("""
<style>
.stApp{
    background: rgb(37, 32, 15);
}
.main-title{
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:white;
}
.sub{
    text-align:center;
    font-size:18px;
    color:white;
    margin-bottom:30px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Button */
.stButton > button {
    background-color: rgb(114, 89, 59);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: rgb(102, 89, 59);
    color: white;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color:rgb(19, 13, 69);
    color: white;
    text-align: center;
    padding: 10px;
    font-size: 14px;
    border-top: 1px solid #ddd;
}
</style>

<div class="footer">
    Developed by <b>Katyayni</b>
</div>
""", unsafe_allow_html=True)