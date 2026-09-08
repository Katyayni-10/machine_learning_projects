import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Salary Predictor",
    layout="centered"
)

with open("salary_model.pkl", "rb") as file:
    model = pickle.load(file)

st.markdown("""
<style>

.stApp{
    background: rgb(19, 13, 69);
}

.main-title{
    text-align:center;
    font-size:42px;
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

st.markdown(
"<h1 class='main-title'>Salary Prediction</h1>",
unsafe_allow_html=True)

st.markdown(
"<p class='sub'>Predict Employee Salary based on Years of Experience</p>",
unsafe_allow_html=True)

with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    years = st.number_input(
        "Years of Experience",
        min_value=0.0,
        max_value=50.0,
        step=0.1
    )
    if st.button("Predict Salary", use_container_width=True):
        prediction = model.predict(np.array([[years]]))
        st.success(f"Predicted Salary: ₹ {prediction[0]:,.2f}")
st.markdown("</div>", unsafe_allow_html=True)

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