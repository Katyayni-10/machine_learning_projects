import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Medical Insurance Cost Predictor",
    layout="wide"
)

with open("decision_tree_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Medical Insurance Cost Prediction")
st.markdown("Predict the estimated medical insurance charges using a trained Decision Tree Regressor.")

st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25
    )

    sex = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1
    )
with col2:
    children = st.number_input(
        "Number of Children",
        min_value=0,
        max_value=10,
        value=0
    )

    smoker = st.selectbox(
        "Smoker",
        ["No", "Yes"]
    )

    region = st.selectbox(
        "Region",
        ["Northeast", "Northwest", "Southeast", "Southwest"]
    )

st.markdown("---")

sex_encoded = 0 if sex == "Female" else 1
smoker_encoded = 0 if smoker == "No" else 1

region_mapping = {
    "Northeast": 0,
    "Northwest": 1,
    "Southeast": 2,
    "Southwest": 3
}

region_encoded = region_mapping[region]

if st.button("Predict Insurance Charges", use_container_width=True):

    user_input = np.array([[
        age,
        sex_encoded,
        bmi,
        children,
        smoker_encoded,
        region_encoded
    ]])

    prediction = model.predict(user_input)

    st.success("Prediction Completed Successfully!")

    st.metric(
        label="Estimated Medical Insurance Charges",
        value=f"${prediction[0]:,.2f}"
    )
