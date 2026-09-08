import streamlit as st
import pickle
import pandas as pd
from sklearn.datasets import load_wine

st.set_page_config(page_title="Wine PCA", page_icon="🍷")

st.title("🍷 Wine Dataset PCA")

# Load saved models
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("pca_model.pkl", "rb") as f:
    pca = pickle.load(f)

# Load dataset
wine = load_wine()
x = wine.data
y = wine.target

# Transform using saved models
x_scaled = scaler.transform(x)
x_pca = pca.transform(x_scaled)

df = pd.DataFrame(x_pca, columns=["PC1", "PC2"])

st.subheader("PCA Output")
st.dataframe(df)

st.subheader("Explained Variance Ratio")
st.write(pca.explained_variance_ratio_)