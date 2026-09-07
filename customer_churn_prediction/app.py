import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

with open("churn_model.pkl","rb") as file:
    model = pickle.load(file)

st.title("Customer Churn Prediction")
st.write("Enter Customer Details")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
with col2:
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])

col3, col4 = st.columns(2)
with col3:
    partner = st.selectbox("Partner", ["No", "Yes"])
with col4:
    tenure = st.number_input("Tenure (Months)", min_value=0)

col5, col6 = st.columns(2)
with col5:
    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )
with col6:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

col7, col8 = st.columns(2)
with col7:
    monthly = st.number_input(
        "Monthly Charges",
        min_value=0.0
    )
with col8:
    total = st.number_input(
        "Total Charges",
        min_value=0.0
    )

gender = {"Female": 0, "Male": 1}[gender]
senior = {"No": 0, "Yes": 1}[senior]
partner = {"No": 0, "Yes": 1}[partner]

internet = {
    "DSL": 0,
    "Fiber optic": 1,
    "No": 2
}[internet]

contract = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}[contract]

if st.button("Predict Churn", use_container_width=True):

    data = np.array([[
        gender,
        senior,
        partner,
        tenure,
        internet,
        contract,
        monthly,
        total
    ]])
    prediction = model.predict(data)
    if prediction[0] == 1:
        st.error("Customer is likely to Churn")
    else:
        st.success("Customer is likely to Stay")


st.markdown("""
<style>
.stApp{
    background: rgb(9, 10, 31);
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
    background-color: rgb(9, 10, 131);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: rgb(9, 10, 75);
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