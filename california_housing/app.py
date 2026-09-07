import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="California House Price Predictor",
    layout="wide"
)

with open("random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("California House Price Prediction")
st.markdown("Predict the **Median House Value** using a trained Random Forest Regressor.")

MedInc = st.number_input(
    "Median Income",
    min_value=0.0,
    value=3.5,
    step=0.1
)

HouseAge = st.number_input(
    "House Age",
    min_value=1.0,
    value=25.0,
    step=1.0
)

AveRooms = st.number_input(
    "Average Rooms",
    min_value=1.0,
    value=5.0,
    step=0.1
)

AveBedrms = st.number_input(
    "Average Bedrooms",
    min_value=0.5,
    value=1.0,
    step=0.1
)

Population = st.number_input(
    "Population",
    min_value=1,
    value=1000,
    step=10
)

AveOccup = st.number_input(
    "Average Occupancy",
    min_value=1.0,
    value=3.0,
    step=0.1
)

Latitude = st.number_input(
    "Latitude",
    value=34.0,
    step=0.1
)

Longitude = st.number_input(
    "Longitude",
    value=-118.0,
    step=0.1
)

if st.button("Predict House Value"):

    user_data = np.array([[
        MedInc,
        HouseAge,
        AveRooms,
        AveBedrms,
        Population,
        AveOccup,
        Latitude,
        Longitude
    ]])

    prediction = model.predict(user_data)

    st.success(f"🏠 Predicted Median House Value: **${prediction[0]:.3f}**")
