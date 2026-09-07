import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Loan Approval Predictor",
    layout="centered"
)

with open("loan_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Loan Approval Prediction")
st.write("Enter the applicant details below.")

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
with col2:
    married = st.selectbox("Married", ["No", "Yes"])

col3, col4 = st.columns(2)
with col3:
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
with col4:
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])

col5, col6 = st.columns(2)
with col5:
    self_emp = st.selectbox("Self Employed", ["No", "Yes"])
with col6:
    property_area = st.selectbox(
        "Property Area",
        ["Rural", "Semiurban", "Urban"]
    )

col7, col8 = st.columns(2)
with col7:
    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0
    )
with col8:
    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0
    )

col9, col10 = st.columns(2)
with col9:
    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0
    )
with col10:
    loan_term = st.number_input(
        "Loan Amount Term",
        min_value=0
    )

credit_history = st.selectbox(
    "Credit History",
    ["Good", "Bad"]
)
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 0 if education == "Graduate" else 1
self_emp = 1 if self_emp == "Yes" else 0
credit_history = 1 if credit_history == "Good" else 0
dependents = {
    "0":0,
    "1":1,
    "2":2,
    "3+":3
}[dependents]
property_area = {
    "Rural":0,
    "Semiurban":1,
    "Urban":2
}[property_area]
if st.button("Predict Loan Status", use_container_width=True):
    data = np.array([[
        gender,
        married,
        dependents,
        education,
        self_emp,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_area
    ]])
    prediction = model.predict(data)
    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")
st.markdown("""
<style>
.stApp{
    background: rgb(9, 47, 14);
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
    background-color: rgb(9, 109, 14);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: rgb(9, 91, 14);
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