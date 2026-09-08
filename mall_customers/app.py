import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title='Mall Customer Segmentation',
    layout='centered'
)

kmeans = pickle.load(open('kmeans_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

df = pd.read_csv('Mall_Customers.csv')

st.markdown('''
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #60a5fa;
}
.sub {
    text-align: center;
    color: #cbd5e1;
    margin-bottom: 25px;
}
.result {
    background: rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.15);
}
</style>
''', unsafe_allow_html=True)

st.markdown('<div class="main-title">Mall Customer Segmentation</div>', unsafe_allow_html=True)

income = st.number_input(
    "Annual Income (k$)",
    min_value=0,
    max_value=200,
    value=50,
    step=1
)

score = st.number_input(
    "Spending Score (1-100)",
    min_value=1,
    max_value=100,
    value=50,
    step=1
)

if st.button('Predict Customer Cluster'):
    user_data = [[income, score]]
    user_scaled = scaler.transform(user_data)
    cluster = int(kmeans.predict(user_scaled)[0])

    labels = {
        0: '🎯 Target Customer',
        1: '💎 High Income • High Spending',
        2: '🧾 Average Customer',
        3: '💰 High Income • Low Spending',
        4: '🛒 Budget Shopper'
    }

    st.info(f'Customer Type: {labels.get(cluster, "Customer Segment")}')
    st.markdown('</div>', unsafe_allow_html=True)
