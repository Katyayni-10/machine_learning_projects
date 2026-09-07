import streamlit as st
import numpy as np
import pickle

st.set_page_config(
    page_title="Fish Weight Prediction",
    layout="wide"
)

@st.cache_resource
def load_model():

    with open("knn_regressor.pkl", "rb") as file:
        model = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    with open("label_encoder.pkl", "rb") as file:
        encoder = pickle.load(file)

    return model, scaler, encoder

model, scaler, encoder = load_model()

st.markdown("""
<style>

.stApp{
background-color:#0E1117;
}

.title{
font-size:42px;
font-weight:bold;
text-align:center;
color:#00BFFF;
}

.subtitle{
font-size:18px;
text-align:center;
color:white;
margin-bottom:20px;
}

.result{
padding:18px;
background:#1B263B;
border-radius:12px;
text-align:center;
font-size:28px;
color:#7CFC00;
font-weight:bold;
}

.footer{
text-align:center;
color:gray;
padding-top:30px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>Fish Weight Prediction</div>", unsafe_allow_html=True)

st.subheader("Enter Fish Details")

col1, col2 = st.columns(2)

with col1:

    species = st.selectbox(
        "Species",
        encoder.classes_
    )

    length1 = st.number_input(
        "Length1",
        min_value=0.0,
        value=23.2
    )

    length2 = st.number_input(
        "Length2",
        min_value=0.0,
        value=25.4
    )

with col2:

    length3 = st.number_input(
        "Length3",
        min_value=0.0,
        value=30.0
    )

    height = st.number_input(
        "Height",
        min_value=0.0,
        value=11.52
    )

    width = st.number_input(
        "Width",
        min_value=0.0,
        value=4.02
    )


if st.button("Predict Fish Weight", use_container_width=True):

    species = encoder.transform([species])[0]

    sample = np.array([[
        species,
        length1,
        length2,
        length3,
        height,
        width
    ]])

    sample = scaler.transform(sample)

    prediction = model.predict(sample)

    # st.success("Prediction Completed Successfully!")

    st.markdown(
        f"""
        <div class='result'>
        Estimated Fish Weight<br><br>
        {prediction[0]:.2f} grams
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()