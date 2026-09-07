import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Diabetes Prediction",
    layout="centered"
)

with open("diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)

st.markdown("""
<style>
.stApp{
    background: rgb(41, 10, 14);
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

st.title("Diabetes Prediction")
st.write("Enter the patient's details below: ")
# st.divider()

col1, col2 = st.columns(2)
with col1:
    preg = st.number_input(
        "Pregnancies",
        min_value=0,
        step=1
    )
with col2:
    glucose = st.number_input(
        "Glucose",
        min_value=0,
        step=1
    )

col3, col4 = st.columns(2)
with col3:
    bp = st.number_input(
        "Blood Pressure",
        min_value=0,
        step=1
    )
with col4:
    skin = st.number_input(
        "Skin Thickness",
        min_value=0,
        step=1
    )

col5, col6 = st.columns(2)
with col5:
    insulin = st.number_input(
        "Insulin",
        min_value=0,
        step=1
    )
with col6:
    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        step=0.1,
        format="%.1f"
    )

col7, col8 = st.columns(2)
with col7:
    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        step=0.01,
        format="%.2f"
    )
with col8:
    age = st.number_input(
        "Age",
        min_value=1,
        step=1
    )

st.write("")
if st.button("Predict Diabetes", use_container_width=True):
    data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(data)
    if prediction[0] == 1:
        st.error("The person is likely to have Diabetes.")
    else:
        st.success("The person is unlikely to have Diabetes.")


st.markdown("""
<style>

/* Button */
.stButton > button {
    background-color: rgb(122, 10, 14);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: rgb(95, 10, 14);
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